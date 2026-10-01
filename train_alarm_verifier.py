"""Compare alarm post-processing variants on leave-one-date-out predictions and train the
stage-2 context verifier (context_verifier.py).

Stage-1 probabilities come from the motion-GBM fold model that did not see the recording's
date. Variants (chosen by out-of-fold alarm F1; the test folder is never used):
  V0  fall_then_still (config postprocess)             V1  V0 + floor state
  VR  post_drop <= 0.5 rule + floor state               V2  logistic verifier + floor state
  V3  logistic verifier without floor state
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

import context_verifier as cv  # noqa: E402
from csi_dataset import load_csi_features, load_scaler  # noqa: E402
from data.build_action_labels import assign_splits, discover_pairs  # noqa: E402
from evaluate_date_folds import build_raw_windows, recording_date  # noqa: E402
from evaluate_stream_events import frame_motion, recording_probabilities  # noqa: E402
from models.motion_gbm import make_gbm, motion_stats  # noqa: E402
from postprocess_state_machine import detect_fall_alarms  # noqa: E402
from train_motion_gbm import checkpoint_dict  # noqa: E402
from models.loading import build_model  # noqa: E402

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def match(alarm_times: list[float], events: pd.DataFrame) -> tuple[int, int, int]:
    """(events, detected, false alarms) with the evaluate_stream_events window (-1 s, +8 s)."""
    detected, matched = 0, set()
    for event in events.itertuples(index=False):
        hits = [i for i, t in enumerate(alarm_times) if event.onset_sec - 1.0 <= t <= event.impact_sec + 8.0]
        detected += bool(hits); matched.update(hits)
    return len(events), detected, len(alarm_times) - len(matched)


def summarize(per_recording: list[tuple]) -> dict:
    out, total = {}, np.zeros(3, dtype=int)
    per_sample = {}
    for key, n, d, f in per_recording:
        date, sample = key.split("|")
        out.setdefault(date, np.zeros(3, dtype=int))
        out[date] += (n, d, f); total += (n, d, f)
        per_sample[sample] = [n, d, f]
    precision = total[1] / max(1, total[1] + total[2]); recall = total[1] / max(1, total[0])
    return {"events": int(total[0]), "detected": int(total[1]), "false_alarms": int(total[2]),
            "alarm_f1": float(2 * precision * recall / max(1e-9, precision + recall)),
            "per_date": {k: {"events": int(v[0]), "detected": int(v[1]), "false_alarms": int(v[2])} for k, v in out.items()},
            "per_recording": per_sample}


def make_verifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_stream_split_v4_events.yaml"))
    parser.add_argument("--stage1", type=Path, default=Path("outputs/motion_gbm_v1/best.pt"))
    parser.add_argument("--window-cache", type=Path, default=Path("outputs/date_folds/cache/action_stream_split_v4"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/alarm_verifier_v1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data, prep = cfg["data"], cfg["preprocessing"]
    stage1 = torch.load(args.stage1, map_location="cpu", weights_only=False)
    post = dict(cfg["postprocess"], fall_threshold=float(stage1["threshold"]))
    fps = int(data["target_fps"])

    # Stage-1 fold models (same recipe as train_motion_gbm.py).
    x_raw, y, w, meta = build_raw_windows(cfg, args.window_cache)
    median, iqr = load_scaler(Path(data["scaler_path"]), x_raw.shape[-1])
    x = np.clip((x_raw - median) / iqr, float(prep["clip_min"]), float(prep["clip_max"])) * iqr + median
    features = motion_stats(x.astype(np.float32), data); del x, x_raw
    usable = ~meta["split"].eq("test").to_numpy()
    dates = sorted(meta.loc[usable, "date"].astype(str).unique())
    modules = {}
    for date in dates:
        fit = usable & ~meta["date"].astype(str).eq(date).to_numpy()
        gbm = make_gbm(args.seed).fit(features[fit], y[fit], sample_weight=w[fit])
        modules[date] = build_model(checkpoint_dict(gbm, cfg, median, iqr, post["fall_threshold"], args.seed))

    events_all = pd.read_csv(data["events_path"])
    pairs, _, _ = discover_pairs(Path(data["raw_csi_dir"]), Path(data["labels_dir"]), data["split"].get("test_dir_name", "test"))
    pairs = [p for p in assign_splits(pairs, data["split"], int(cfg.get("seed", 42))) if p["split"] != "test"]
    stride = max(1, round(float(post.get("inference_stride_seconds", data["stride_seconds"])) * fps))
    recs = []
    for pair in pairs:
        date = recording_date(Path(pair["raw_path"]))
        frame, raw, _ = load_csi_features(Path(pair["raw_path"]), data)
        starts, probs = recording_probabilities(modules[date], torch.device("cpu"), raw, data, prep, median, iqr, stride)
        times = frame["time_sec"].to_numpy()
        rec = {"sample_id": pair["sample_id"], "date": date, "decision": times[starts] + float(data["window_seconds"]),
               "probs": probs, "frame_times": times + 1.0 / fps, "scaled_motion": frame_motion(raw, data, median, iqr),
               "events": events_all[events_all["sample_id"].astype(str).eq(pair["sample_id"])]}
        motion = cv.relative_motion(raw, data, float(post.get("quiet_window_sec", 60.0)))
        rec["segments"] = cv.floor_segments(rec["frame_times"], motion, post)
        rec["candidates"] = cv.candidates(rec["decision"], probs, rec["frame_times"], motion, raw, data, post)
        for row in rec["candidates"]:
            row["label"] = int(any(e.onset_sec - 1.0 <= row["decision_sec"] <= e.impact_sec + 8.0 for e in rec["events"].itertuples()))
        recs.append(rec)
        print(f"{pair['sample_id']} [{date}]: {len(rec['candidates'])} candidates", flush=True)

    results = {}
    for name, floor in (("V0_fall_then_still", False), ("V1_fall_then_still_floor", True)):
        rule_post = dict(post, floor_state=floor)
        results[name] = summarize([(r["date"] + "|" + r["sample_id"], *match([a["time_sec"] for a in detect_fall_alarms(
            r["decision"], r["probs"], r["frame_times"], r["scaled_motion"], rule_post, r["segments"])], r["events"])) for r in recs])
    rule_post = dict(post, floor_state=True, verifier_threshold=0.5)
    results["VR_post_drop_rule_floor"] = summarize([(r["date"] + "|" + r["sample_id"], *match([a["time_sec"] for a in cv.select_alarms(
        r["candidates"], cv.rule_scores(r["candidates"], rule_post), rule_post, r["segments"])], r["events"])) for r in recs])

    # Verifier: leave-one-date-out over candidates, then choose the threshold on OOF alarm F1.
    for rec in recs:
        rec["scores"] = np.zeros(len(rec["candidates"]))
    for date in dates:
        train_rows = [row for r in recs if r["date"] != date for row in r["candidates"]]
        medians = np.nanmedian(cv.feature_matrix(train_rows), axis=0)
        model = make_verifier().fit(cv.feature_matrix(train_rows, medians), [row["label"] for row in train_rows])
        for rec in recs:
            if rec["date"] == date and rec["candidates"]:
                rec["scores"] = model.predict_proba(cv.feature_matrix(rec["candidates"], medians))[:, 1]
    tables = {}
    for name, floor in (("V2_verifier_floor", True), ("V3_verifier", False)):
        tables[name] = {}
        for t in THRESHOLDS:
            vpost = dict(post, floor_state=floor, verifier_threshold=float(t))
            tables[name][float(t)] = summarize([(r["date"] + "|" + r["sample_id"], *match([a["time_sec"] for a in cv.select_alarms(
                r["candidates"], r["scores"], vpost, r["segments"])], r["events"])) for r in recs])
        best = max(tables[name], key=lambda t: tables[name][t]["alarm_f1"])
        results[name] = dict(tables[name][best], verifier_threshold=best)

    labels = np.array([row["label"] for r in recs for row in r["candidates"]])
    print(f"\ncandidates {len(labels)}, positive {labels.sum()}")
    for name, r in results.items():
        extra = f" (threshold {r['verifier_threshold']:.2f})" if "verifier_threshold" in r else ""
        print(f"{name:28s} detected {r['detected']}/{r['events']}, false alarms {r['false_alarms']}, F1 {r['alarm_f1']:.3f}{extra} "
              + " | ".join(f"{d}: {v['detected']}/{v['events']} FA {v['false_alarms']}" for d, v in r["per_date"].items()))

    all_rows = [row for r in recs for row in r["candidates"]]
    medians = np.nanmedian(cv.feature_matrix(all_rows), axis=0)
    final = make_verifier().fit(cv.feature_matrix(all_rows, medians), labels)
    coef = dict(zip(cv.FEATURES, final[-1].coef_[0].round(3).tolist()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "context_verifier", "model": final, "medians": medians, "features": cv.FEATURES,
                "threshold": results["V2_verifier_floor"]["verifier_threshold"], "stage1": str(args.stage1)},
               args.output_dir / "verifier.pt")
    (args.output_dir / "oof_report.json").write_text(json.dumps(
        {"stage1_threshold": post["fall_threshold"], "variants": results, "verifier_threshold_tables": tables,
         "verifier_coefficients": coef, "candidates": int(len(labels)), "positive_candidates": int(labels.sum())},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("coefficients", coef)
    print(f"Saved {args.output_dir / 'verifier.pt'}")


if __name__ == "__main__":
    main()
