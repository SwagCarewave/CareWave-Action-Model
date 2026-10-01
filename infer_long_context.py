"""Fall alarms for one raw CSI recording with the long-context detector (train_long_context.py).

Stage 1: motion-GBM window probabilities every 0.5 s. Stage 2: every 0.5 s, the
long-context model scores "a fall happened 2-6 s ago" from the last 30 s; an alarm fires
when 2 of the last 3 scores reach the saved threshold (10 s cooldown).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

import long_context as lc
from csi_dataset import load_csi_features, load_scaler
from evaluate_stream_events import recording_probabilities
from models.loading import build_model


def detect(raw_csv: Path, checkpoint: dict) -> tuple[pd.DataFrame, list[float]]:
    cfg = checkpoint["config"]; data, prep, post = cfg["data"], cfg["preprocessing"], cfg["postprocess"]
    fps = int(data["target_fps"])
    stage1 = torch.load(checkpoint["stage1"], map_location="cpu", weights_only=False)
    frame, raw, _ = load_csi_features(raw_csv, data)
    median, iqr = load_scaler(Path(data["scaler_path"]), raw.shape[1])
    stride = max(1, round(float(post.get("inference_stride_seconds", data["stride_seconds"])) * fps))
    starts, probs = recording_probabilities(build_model(stage1), torch.device("cpu"), raw, data, prep, median, iqr, stride)
    frame_times = frame["time_sec"].to_numpy() + 1.0 / fps
    s1_times = frame["time_sec"].to_numpy()[starts] + float(data["window_seconds"])
    times, rows = lc.recording_rows(frame_times, lc.recording_signals(raw, data), s1_times, probs)
    scores = checkpoint["model"].predict_proba(rows)[:, 1]
    alarms = lc.alarms_from_scores(times, scores, float(checkpoint["threshold"]))
    return pd.DataFrame({"decision_sec": times, "fall_score": scores}), alarms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/long_context_v1_stopw5/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/long_context_v1_stopw5/inference.csv"))
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    scores, alarms = detect(args.raw_csv, checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.assign(alarm=scores["decision_sec"].isin(alarms)).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"{len(alarms)} fall alarms at {[round(t, 1) for t in alarms]} s (threshold {checkpoint['threshold']:.2f}); saved {args.output}")


if __name__ == "__main__":
    main()
