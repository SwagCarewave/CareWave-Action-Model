"""Evaluate a trained Pose-only model: window metrics + event metrics (recall, false alarms/hour, latency)."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from build_pose_windows import load_intervals
from models.pose_encoder import PoseOnlyClassifier
from train_pose_classifier import Standardizer, load_split, predict, window_metrics

WIN = 30
MATCH_PRE, MATCH_POST = 0.5, 2.0


def load_run(run: Path) -> tuple[PoseOnlyClassifier, dict, Standardizer, Standardizer]:
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = PoseOnlyClassifier(ck["in_dim"], ck["q_channels"], hidden=cfg["hidden"], dropout=cfg["dropout"],
                               use_gate=cfg["gate"], use_quality_input=cfg["quality_input"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    fs, qs = Standardizer.__new__(Standardizer), Standardizer.__new__(Standardizer)
    fs.mean, fs.std = ck["feature_scaler"]["mean"], ck["feature_scaler"]["std"]
    qs.mean, qs.std = ck["quality_scaler"]["mean"], ck["quality_scaler"]["std"]
    return model, ck, fs, qs


def stream_probs(model, fs, qs, feat: dict, WIN: int = WIN) -> tuple[np.ndarray, np.ndarray]:
    """p_fall at every frame whose trailing WIN-frame window is valid (NaN elsewhere)."""
    x, q, valid, t = fs(feat["features"]), qs(feat["quality"]), feat["valid_mask"], feat["time_sec"]
    n = len(x)
    p = np.full(n, np.nan, np.float32)
    if n < WIN:
        return p, t
    xw = np.lib.stride_tricks.sliding_window_view(x, WIN, axis=0).transpose(0, 2, 1)
    qw = np.lib.stride_tricks.sliding_window_view(q, WIN, axis=0).transpose(0, 2, 1)
    vw = np.lib.stride_tricks.sliding_window_view(valid.astype(np.float32), WIN).mean(1)
    gw = np.lib.stride_tricks.sliding_window_view(t, WIN).__array__()
    ok = (vw >= 0.95) & (np.diff(gw, axis=1).max(1) <= 0.15 + 1e-6)
    if ok.any():
        probs, _ = predict(model, torch.from_numpy(np.ascontiguousarray(xw[ok])), torch.from_numpy(np.ascontiguousarray(qw[ok])))
        p[np.where(ok)[0] + WIN - 1] = probs
    return p, t


def state_machine(p: np.ndarray, t: np.ndarray, thr: float, alpha: float = 0.6, k: int = 3, of: int = 5,
                  cooldown: float = 5.0) -> list[float]:
    """Exponential smoothing -> candidate -> k of last `of` -> confirmed alarm times (with cooldown)."""
    alarms, smooth, recent, last_alarm = [], None, [], -1e9
    for pi, ti in zip(p, t):
        if np.isnan(pi):
            recent.clear(); smooth = None
            continue
        smooth = pi if smooth is None else alpha * pi + (1 - alpha) * smooth
        recent = (recent + [smooth >= thr])[-of:]
        if sum(recent) >= k and ti - last_alarm >= cooldown:
            alarms.append(float(ti)); last_alarm = ti
    return alarms


def event_metrics(streams: list[dict], params: dict) -> dict:
    n_events = matched = alarms_total = alarms_true = dup = 0
    latencies, neg_hours = [], 0.0
    for s in streams:
        alarms = state_machine(s["p"], s["t"], **params)
        used = np.zeros(len(alarms), bool)
        zone_sec = 0.0
        for on, im, post_end in s["events"]:
            n_events += 1
            end = max(im + MATCH_POST, post_end)
            zone_sec += end - (on - MATCH_PRE)
            hit = [i for i, a in enumerate(alarms) if on - MATCH_PRE <= a <= end]
            if hit:
                matched += 1
                latencies.append(alarms[hit[0]] - on)
                dup += len(hit) - 1
                used[hit] = True
        alarms_total += len(alarms)
        alarms_true += int(used.sum())
        neg_hours += max(s["duration"] - zone_sec, 0.0) / 3600.0
    false_alarms = alarms_total - alarms_true
    precision = alarms_true / max(alarms_total, 1)
    recall = matched / max(n_events, 1)
    return {"events": n_events, "matched": matched, "event_recall": recall, "event_precision": precision,
            "event_f1": 2 * precision * recall / max(precision + recall, 1e-9), "alarms": alarms_total,
            "false_alarms": false_alarms, "false_alarms_per_hour": false_alarms / max(neg_hours, 1e-9),
            "negative_hours": neg_hours, "mean_latency_sec": float(np.mean(latencies)) if latencies else float("nan"),
            "duplicate_alarms": dup}


def build_streams(model, fs, qs, window_dir: Path, feature_dir: Path, split: str, labels_dir: Path,
                  index: pd.DataFrame, win_len: int = WIN) -> list[dict]:
    manifest = pd.read_csv(window_dir / "split_manifest_binary.csv", encoding="utf-8-sig")
    intervals = load_intervals(labels_dir)
    action = dict(zip(index.sample_id, index.action))
    streams = []
    for sid in manifest[manifest.split == split].sample_id:
        with np.load(feature_dir / f"{sid}.npz", allow_pickle=True) as z:
            feat = {k: z[k] for k in z.files}
        iv = intervals.get(sid)
        events = []
        for j, (a, b, lab) in enumerate(sorted(iv or [])):
            if lab != "falling":
                continue
            post_end, k = b, j + 1
            ordered = sorted(iv)
            while k < len(ordered) and ordered[k][2] in {"lying", "getting_up", "transition"}:
                post_end = ordered[k][1]; k += 1
            events.append((a, b, post_end))
        end = max(b for _, b, _ in iv) if iv else float(feat["time_sec"][-1])
        keep = feat["time_sec"] < end + 1e-6
        feat = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(keep) else v) for k, v in feat.items()}
        p, t = stream_probs(model, fs, qs, feat, win_len)
        streams.append({"sample_id": sid, "p": p, "t": t, "events": events, "duration": float(t[-1] - t[0]),
                        "action": action.get(sid, "")})
    return streams


def tune_event_params(streams: list[dict], floor: float) -> tuple[dict, pd.DataFrame]:
    rows = []
    for thr, alpha, k in itertools.product([0.5, 0.6, 0.7, 0.8, 0.9], [0.4, 0.6, 0.8], [2, 3, 4]):
        prm = {"thr": thr, "alpha": alpha, "k": k}
        rows.append({**prm, **event_metrics(streams, prm)})
    df = pd.DataFrame(rows)
    ok = df[df.event_recall >= floor]
    pool = ok if len(ok) else df
    best = pool.sort_values(["false_alarms_per_hour", "event_f1"], ascending=[True, False]).iloc[0]
    return {"thr": float(best.thr), "alpha": float(best.alpha), "k": int(best.k)}, df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/pose_branch.yaml"))
    ap.add_argument("--final-test", action="store_true", help="score the sealed test split once, with frozen params")
    ap.add_argument("--recall-floor", type=float, default=0.8)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model, ck, fs, qs = load_run(args.run)
    window_dir = Path(cfg["paths"]["window_dir"]) / ck["tag"]
    hashes = json.loads(str(ck["source_hashes"]))
    variant, fmode = hashes["variant"], hashes["feature_mode"]
    feature_dir = Path(cfg["paths"]["feature_dir"]) / f"{variant}_{fmode}"
    index = pd.read_csv(Path(cfg["paths"]["pose_cache_dir"]) / variant / "index.csv", encoding="utf-8-sig")
    split = "test" if args.final_test else "validation"
    params_path = args.run / "event_params.json"

    if args.final_test:
        if (args.run / "test_metrics.json").exists():
            raise SystemExit("test_metrics.json exists: the test split was already scored for this run (score it once).")
        if not params_path.exists():
            raise SystemExit("Run without --final-test first so post-processing params are frozen on validation.")

    data = load_split(window_dir, split)
    x = torch.from_numpy(fs(data["X_pose"])); q = torch.from_numpy(qs(data["X_quality"]))
    prob, gate = predict(model, x, q)
    win = window_metrics(data["y"].astype(int), prob, ck["threshold"])
    pd.DataFrame({"sample_id": data["sample_id"], "start_sec": data["start_sec"], "end_sec": data["end_sec"],
                  "subtype": data["subtype"] if "subtype" in data else "", "y": data["y"], "p_fall": prob, "gate": gate}
                 ).to_csv(args.run / f"{split}_window_predictions.csv", index=False)
    sub = pd.DataFrame({"subtype": data["subtype"], "pred": prob >= ck["threshold"], "y": data["y"]})
    per_subtype = sub[sub.y == 0].groupby("subtype").pred.mean().round(3).to_dict()

    WIN_LEN = int(data["X_pose"].shape[1])
    streams = build_streams(model, fs, qs, window_dir, feature_dir, split, Path(cfg["paths"]["labels_dir"]), index, WIN_LEN)
    if args.final_test:
        params = json.loads(params_path.read_text())
        grid = None
    else:
        params, grid = tune_event_params(streams, args.recall_floor)
        params_path.write_text(json.dumps(params), encoding="utf-8")
        grid.to_csv(args.run / "event_grid_validation.csv", index=False)
    events = event_metrics(streams, params)
    result = {"split": split, "window": win, "event_params": params, "event": events,
              "false_alarm_rate_by_nonfall_subtype": per_subtype, "mean_gate": float(gate.mean()), "run": str(args.run)}
    (args.run / f"{'test' if args.final_test else 'val'}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    cm = np.array(win["confusion"])
    ax.imshow(cm, cmap="Blues")
    for i, j in itertools.product(range(2), range(2)):
        ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks([0, 1], ["standing", "fall"]); ax.set_yticks([0, 1], ["standing", "fall"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{split} confusion")
    fig.tight_layout(); fig.savefig(args.run / f"confusion_matrix_{split}.png", dpi=120); plt.close(fig)

    shown = [s for s in streams if s["events"]][:4]
    if shown:
        fig, axes = plt.subplots(len(shown), 1, figsize=(9, 2.2 * len(shown)), squeeze=False)
        for ax, s in zip(axes[:, 0], shown):
            ax.plot(s["t"], s["p"], lw=1, label="p_fall")
            for on, im, _ in s["events"]:
                ax.axvspan(on, im, color="tab:red", alpha=0.25)
            for a in state_machine(s["p"], s["t"], **params):
                ax.axvline(a, color="k", ls="--", lw=1)
            ax.set_ylim(0, 1); ax.set_title(s["sample_id"], fontsize=8)
        fig.tight_layout(); fig.savefig(args.run / f"event_timeline_{split}.png", dpi=110); plt.close(fig)

    print(f"[{split}] window: AUPRC={win['auprc']:.3f} AUROC={win['auroc']:.3f} P={win['precision']:.3f} R={win['recall']:.3f} "
          f"F1={win['f1']:.3f} spec={win['specificity']:.3f} (n={win['n']}, fall={win['n_fall']})")
    print(f"[{split}] event {params}: recall={events['event_recall']:.2f} ({events['matched']}/{events['events']}) "
          f"precision={events['event_precision']:.2f} FA/h={events['false_alarms_per_hour']:.1f} "
          f"latency={events['mean_latency_sec']:.2f}s over {events['negative_hours']:.2f} h")


if __name__ == "__main__":
    main()
