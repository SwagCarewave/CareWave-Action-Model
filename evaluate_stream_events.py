"""Event-level streaming evaluation: run the model over whole recordings and score alarms.

Unlike evaluate_action_classifier.py (which scores only the labelled windows kept by
build_action_windows.py), every window of each recording is scored in time order, as
in real-time use, so lying/getting-up/transition periods are included.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from csi_dataset import labels_for_times, load_csi_features, load_labels, load_scaler, transform_window
from data.build_action_labels import assign_splits, discover_pairs
from models.loading import build_model
from postprocess_state_machine import detect_fall_alarms


def recording_probabilities(model, device, raw_features, data, prep, median, iqr, stride):
    window = round(float(data["window_seconds"]) * int(data["target_fps"]))
    starts = list(range(0, len(raw_features) - window + 1, stride))
    batch = np.stack([
        np.clip((transform_window(raw_features[s:s + window], data) - median) / iqr,
                float(prep["clip_min"]), float(prep["clip_max"]))
        for s in starts
    ]).astype(np.float32)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(batch).to(device))).cpu().numpy()
    return np.asarray(starts), probabilities.astype(np.float64)


def frame_motion(raw_features, data, median, iqr):
    """Mean scaled packet-motion per 10 Hz frame (stream_split features only)."""
    if data.get("feature_mode") != "stream_split":
        return None
    dims = slice(raw_features.shape[1] - 6, raw_features.shape[1])
    return ((raw_features[:, dims] - median[dims]) / iqr[dims]).mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match-before-sec", type=float, default=1.0)
    parser.add_argument("--match-after-sec", type=float, default=8.0)
    parser.add_argument("--save-probabilities", type=Path, help="Optional CSV of per-window probabilities")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data, prep, post = cfg["data"], cfg["preprocessing"], dict(cfg.get("postprocess", {}))
    fps = int(data["target_fps"])
    stride = max(1, round(float(post.get("inference_stride_seconds", data["stride_seconds"])) * fps))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint).to(device)
    post["fall_threshold"] = max(float(checkpoint.get("threshold", 0.5)), float(post.get("fall_threshold", 0.0)))

    split_cfg = data["split"]
    pairs, _, _ = discover_pairs(Path(data["raw_csi_dir"]), Path(data["labels_dir"]), split_cfg.get("test_dir_name", "test"))
    pairs = [p for p in assign_splits(pairs, split_cfg, int(cfg.get("seed", 42))) if p["split"] == args.split]
    events = pd.read_csv(data["events_path"])
    median = iqr = None

    rows, event_rows, alarm_rows = [], [], []
    for pair in pairs:
        frame, raw_features, _ = load_csi_features(Path(pair["raw_path"]), data)
        if median is None or len(median) != raw_features.shape[1]:
            median, iqr = load_scaler(Path(data["scaler_path"]), raw_features.shape[1])
        starts, probabilities = recording_probabilities(model, device, raw_features, data, prep, median, iqr, stride)
        window_sec = float(data["window_seconds"])
        times = frame["time_sec"].to_numpy()
        decision_times = times[starts] + window_sec  # causal: decision at window end
        motion = frame_motion(raw_features, data, median, iqr)
        if post.get("mode", "confirm") != "confirm" and motion is None:
            raise SystemExit("fall_then_still post-processing needs stream_split features")
        alarms = detect_fall_alarms(decision_times, probabilities, times + 1.0 / fps,
                                    motion if motion is not None else np.zeros(len(times)), post)
        frame_labels = labels_for_times(times, load_labels(Path(pair["label_path"])))
        sample_events = events[events["sample_id"].astype(str).eq(pair["sample_id"])]
        matched = set()
        for event in sample_events.itertuples(index=False):
            lo, hi = float(event.onset_sec) - args.match_before_sec, float(event.impact_sec) + args.match_after_sec
            hits = [a for a in alarms if lo <= a["time_sec"] <= hi]
            matched.update(id(a) for a in hits)
            event_rows.append({"sample_id": pair["sample_id"], "event_id": event.event_id,
                               "onset_sec": float(event.onset_sec), "detected": bool(hits),
                               "latency_sec": (hits[0]["time_sec"] - float(event.onset_sec)) if hits else None})
        for alarm in alarms:
            at = (times >= alarm["time_sec"] - window_sec) & (times < alarm["time_sec"])
            context = pd.Series(frame_labels[at]).value_counts()
            alarm_rows.append({"sample_id": pair["sample_id"], **alarm, "true_alarm": id(alarm) in matched,
                               "window_label": str(context.index[0]) if len(context) else "unknown"})
        if args.save_probabilities:
            rows.append(pd.DataFrame({"sample_id": pair["sample_id"], "decision_sec": decision_times,
                                      "fall_probability": probabilities}))
        print(f"{pair['sample_id']}: {len(sample_events)} events, {len(alarms)} alarms")

    events_df, alarms_df = pd.DataFrame(event_rows), pd.DataFrame(alarm_rows)
    false_alarms = alarms_df[~alarms_df["true_alarm"]] if len(alarms_df) else alarms_df
    report = {
        "split": args.split, "checkpoint": str(args.checkpoint), "postprocess": post,
        "recordings": len(pairs), "events": int(len(events_df)),
        "events_detected": int(events_df["detected"].sum()) if len(events_df) else 0,
        "events_missed": int((~events_df["detected"]).sum()) if len(events_df) else 0,
        "false_alarms": int(len(false_alarms)),
        "false_alarms_by_label": false_alarms["window_label"].value_counts().to_dict() if len(false_alarms) else {},
        "false_alarms_by_recording": false_alarms["sample_id"].value_counts().to_dict() if len(false_alarms) else {},
        "mean_latency_sec": float(events_df["latency_sec"].dropna().mean()) if len(events_df) and events_df["detected"].any() else None,
        "match_window_sec": [-args.match_before_sec, args.match_after_sec],
        "events_detail": event_rows, "alarms": alarm_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_probabilities and rows:
        pd.concat(rows).to_csv(args.save_probabilities, index=False, encoding="utf-8-sig")
    print(json.dumps({k: report[k] for k in ("events", "events_detected", "events_missed", "false_alarms",
                                             "false_alarms_by_label", "mean_latency_sec")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
