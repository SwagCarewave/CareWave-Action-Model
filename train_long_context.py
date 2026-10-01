"""Train and evaluate the dense long-context fall detector (long_context.py).

1. Stage-1 motion-GBM probabilities: date-fold (out-of-date) models for non-test recordings,
   the saved final model for test recordings.
2. Long-context rows every 0.5 s; leave-one-date-out GBM -> OOF scores.
3. Alarm threshold chosen on OOF alarm F1 (test never used); the full recall/false-alarm
   curve is saved as well.
4. Final model on all non-test rows; test recordings scored once.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

import long_context as lc  # noqa: E402
from csi_dataset import labels_for_times, load_csi_features, load_labels, load_scaler  # noqa: E402
from data.build_action_labels import assign_splits, discover_pairs  # noqa: E402
from evaluate_date_folds import build_raw_windows, recording_date  # noqa: E402
from evaluate_stream_events import recording_probabilities  # noqa: E402
from models.loading import build_model  # noqa: E402
from models.motion_gbm import make_gbm, motion_stats  # noqa: E402
from train_motion_gbm import checkpoint_dict  # noqa: E402

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def make_model(seed: int):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                          l2_regularization=1.0, class_weight="balanced", random_state=seed)


def score_alarms(rec: dict, alarms: list[float]) -> dict:
    events = rec["events"]
    detected, matched, latency = 0, set(), []
    for e in events.itertuples(index=False):
        hits = [i for i, t in enumerate(alarms) if e.onset_sec - 1.0 <= t <= e.impact_sec + 8.0]
        if hits:
            detected += 1; latency.append(alarms[hits[0]] - e.onset_sec)
        matched.update(hits)
    false = [alarms[i] for i in range(len(alarms)) if i not in matched]
    labels = []
    for t in false:
        near = rec["frame_labels"][(rec["frame_times"] > t - 3.0) & (rec["frame_times"] <= t)]
        labels.append(pd.Series(near).mode().iat[0] if len(near) else "unknown")
    return {"events": len(events), "detected": detected, "alarms": len(alarms), "false_alarms": len(false),
            "false_alarm_times": false, "false_alarm_labels": labels, "latency": latency}


def total(results: list[dict], minutes: float) -> dict:
    n = sum(r["events"] for r in results); d = sum(r["detected"] for r in results)
    a = sum(r["alarms"] for r in results); f = sum(r["false_alarms"] for r in results)
    precision = (a - f) / a if a else 0.0; recall = d / n if n else 0.0
    lat = [x for r in results for x in r["latency"]]
    return {"events": n, "detected": d, "recall": recall, "false_alarms": f, "false_alarms_per_hour": f / minutes * 60,
            "alarm_precision": precision, "alarm_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mean_latency_sec": float(np.mean(lat)) if lat else None, "minutes": minutes,
            "false_alarm_labels": pd.Series([l for r in results for l in r["false_alarm_labels"]]).value_counts().to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_stream_split_v4_events.yaml"))
    parser.add_argument("--stage1", type=Path, default=Path("outputs/motion_gbm_v1/best.pt"))
    parser.add_argument("--window-cache", type=Path, default=Path("outputs/date_folds/cache/action_stream_split_v4"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/long_context_v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-weight", type=float, default=1.0, help="weight for move-then-stand decisions")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data, prep, post = cfg["data"], cfg["preprocessing"], cfg["postprocess"]
    fps = int(data["target_fps"])

    x_raw, y, w, meta = build_raw_windows(cfg, args.window_cache)
    median, iqr = load_scaler(Path(data["scaler_path"]), x_raw.shape[-1])
    x = np.clip((x_raw - median) / iqr, float(prep["clip_min"]), float(prep["clip_max"])) * iqr + median
    feats = motion_stats(x.astype(np.float32), data); del x, x_raw
    usable = ~meta["split"].eq("test").to_numpy()
    dates = sorted(meta.loc[usable, "date"].astype(str).unique())
    stage1 = {"final": build_model(torch.load(args.stage1, map_location="cpu", weights_only=False))}
    for date in dates:
        fit = usable & ~meta["date"].astype(str).eq(date).to_numpy()
        stage1[date] = build_model(checkpoint_dict(make_gbm(args.seed).fit(feats[fit], y[fit], sample_weight=w[fit]),
                                                   cfg, median, iqr, 0.5, args.seed))

    events_all = pd.read_csv(data["events_path"])
    pairs, _, _ = discover_pairs(Path(data["raw_csi_dir"]), Path(data["labels_dir"]), data["split"].get("test_dir_name", "test"))
    pairs = assign_splits(pairs, data["split"], int(cfg.get("seed", 42)))
    stride = max(1, round(float(post.get("inference_stride_seconds", data["stride_seconds"])) * fps))
    recs = []
    for pair in pairs:
        date = recording_date(Path(pair["raw_path"]))
        is_test = pair["split"] == "test"
        frame, raw, _ = load_csi_features(Path(pair["raw_path"]), data)
        model = stage1["final" if is_test else date]
        starts, probs = recording_probabilities(model, torch.device("cpu"), raw, data, prep, median, iqr, stride)
        frame_times = frame["time_sec"].to_numpy() + 1.0 / fps
        s1_times = frame["time_sec"].to_numpy()[starts] + float(data["window_seconds"])
        signals = lc.recording_signals(raw, data)
        times, rows = lc.recording_rows(frame_times, signals, s1_times, probs)
        events = events_all[events_all["sample_id"].astype(str).eq(pair["sample_id"])]
        recs.append({"sample_id": pair["sample_id"], "date": date, "test": is_test, "times": times, "x": rows,
                     "y": lc.labels_for(times, events), "events": events,
                     "w": lc.stop_weights(times, load_labels(Path(pair["label_path"])), args.stop_weight), "frame_times": frame_times,
                     "frame_labels": labels_for_times(frame["time_sec"].to_numpy(), load_labels(Path(pair["label_path"]))),
                     "minutes": float(frame_times[-1]) / 60})
        print(f"{pair['sample_id']} [{date}{', test' if is_test else ''}]: {len(times)} decisions, {int((recs[-1]['y'] == 1).sum())} positive", flush=True)

    train = [r for r in recs if not r["test"]]
    for date in dates:
        fit = [r for r in train if r["date"] != date]
        xs = np.concatenate([r["x"] for r in fit]); ys = np.concatenate([r["y"] for r in fit])
        ws = np.concatenate([r["w"] for r in fit])
        model = make_model(args.seed).fit(xs[ys >= 0], ys[ys >= 0], sample_weight=ws[ys >= 0])
        for r in train:
            if r["date"] == date:
                r["oof"] = model.predict_proba(r["x"])[:, 1]

    minutes = sum(r["minutes"] for r in train)
    curve = {}
    for t in THRESHOLDS:
        curve[float(t)] = total([score_alarms(r, lc.alarms_from_scores(r["times"], r["oof"], float(t))) for r in train], minutes)
        c = curve[float(t)]
        print(f"OOF threshold {t:.2f}: detected {c['detected']}/{c['events']} ({100 * c['recall']:.1f}%), "
              f"false alarms {c['false_alarms']} ({c['false_alarms_per_hour']:.1f}/h), F1 {c['alarm_f1']:.3f}", flush=True)
    threshold = max(curve, key=lambda t: curve[t]["alarm_f1"])
    per_date = {d: total([score_alarms(r, lc.alarms_from_scores(r["times"], r["oof"], threshold)) for r in train if r["date"] == d],
                         sum(r["minutes"] for r in train if r["date"] == d)) for d in dates}
    print(f"\nSelected threshold {threshold:.2f}")
    for d, c in per_date.items():
        print(f"  OOF {d}: {c['detected']}/{c['events']}, false alarms {c['false_alarms']} ({c['false_alarms_per_hour']:.1f}/h)")

    xs = np.concatenate([r["x"] for r in train]); ys = np.concatenate([r["y"] for r in train])
    ws = np.concatenate([r["w"] for r in train])
    final = make_model(args.seed).fit(xs[ys >= 0], ys[ys >= 0], sample_weight=ws[ys >= 0])
    groups = {"final4": ["hoyeon_walk_stop_fall_01", "csi_test_01", "yena_test_02", "sujin_test_03"],
              "dev5": ["sujin_walk_stop_fall_01", "yena_fall_normal_18", "yena_test_03", "yena_test_04", "hoyeon_test_04"]}
    groups["all9"] = groups["final4"] + groups["dev5"]
    test_results = {}
    for r in recs:
        if r["test"]:
            r["score"] = final.predict_proba(r["x"])[:, 1]
            r["result"] = score_alarms(r, lc.alarms_from_scores(r["times"], r["score"], threshold))
    for name, samples in groups.items():
        chosen = [r for r in recs if r["sample_id"] in samples]
        test_results[name] = total([r["result"] for r in chosen], sum(r["minutes"] for r in chosen))
        c = test_results[name]
        print(f"TEST {name}: detected {c['detected']}/{c['events']} ({100 * c['recall']:.1f}%), false alarms {c['false_alarms']} "
              f"({c['false_alarms_per_hour']:.1f}/h), precision {100 * c['alarm_precision']:.1f}%, F1 {c['alarm_f1']:.3f}, "
              f"latency {c['mean_latency_sec']}, FA labels {c['false_alarm_labels']}")
    for r in recs:
        if r["test"] and (r["result"]["false_alarms"] or r["result"]["detected"] < r["result"]["events"]):
            print(f"  {r['sample_id']}: detected {r['result']['detected']}/{r['result']['events']}, "
                  f"FA at {r['result']['false_alarm_times']} {r['result']['false_alarm_labels']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "long_context", "model": final, "threshold": threshold, "features": lc.FEATURE_NAMES,
                "stage1": str(args.stage1), "config": cfg}, args.output_dir / "model.pt")
    (args.output_dir / "report.json").write_text(json.dumps(
        {"threshold": threshold, "oof_curve": curve, "oof_per_date": per_date, "test": test_results},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
