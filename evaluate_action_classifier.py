"""Evaluate a trained CSI Stream checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

from models.csi_encoder import CSIActionClassifier
from postprocess_state_machine import ActionStateMachine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/csi_stream/best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/csi_stream"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", type=Path, default=Path("outputs/csi_stream/test_metrics.json"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CSIActionClassifier(
        num_classes=len(checkpoint["class_names"]), **checkpoint["model_config"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    x = np.load(args.data_dir / "X.npy").astype(np.float32)
    y = np.load(args.data_dir / "y.npy").astype(np.int64)
    meta = pd.read_csv(args.data_dir / "metadata.csv")
    idx = np.flatnonzero(meta["split"].to_numpy() == args.split)
    if not len(idx):
        raise SystemExit(f"Split {args.split!r} is empty")

    x = x[idx]
    preprocessing = checkpoint.get("preprocessing", {})
    if preprocessing.get("window_center", False):
        x = x - x.mean(axis=1, keepdims=True)
    x = (
        x - checkpoint["mean"][None, None, :]
    ) / checkpoint["std"][None, None, :]

    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()
        pred = probabilities.argmax(1)

    names = checkpoint["class_names"]
    labels = list(range(len(names)))
    report = classification_report(
        y[idx], pred, labels=labels, target_names=names,
        output_dict=True, zero_division=0,
    )
    fall_id = names.index("falling")
    breakdown = {}
    selected_meta = meta.iloc[idx].reset_index(drop=True)
    for column in ("subject_id", "sample_id"):
        breakdown[column] = {}
        for value, rows in selected_meta.groupby(column).groups.items():
            rows = np.asarray(list(rows), dtype=int)
            breakdown[column][str(value)] = classification_report(
                y[idx][rows], pred[rows], labels=labels, target_names=names,
                output_dict=True, zero_division=0,
            )

    post_cfg = checkpoint.get(
        "postprocess", checkpoint.get("config", {}).get("postprocess", {})
    )
    true_events: set[str] = set()
    detected_events: set[str] = set()
    false_events = 0
    unknown_count = 0
    latencies = []
    duration_sec = 0.0
    for _, rows in selected_meta.groupby("sample_id").groups.items():
        rows = np.asarray(list(rows), dtype=int)
        rows = rows[np.argsort(selected_meta.iloc[rows]["center_sec"].to_numpy())]
        sample_meta = selected_meta.iloc[rows]
        machine = ActionStateMachine(names, post_cfg)
        sample_true = {
            str(e)
            for e in sample_meta.loc[sample_meta["label"] == "falling", "event_id"]
            if str(e)
        }
        true_events.update(sample_true)
        if len(sample_meta):
            duration_sec += float(
                sample_meta["end_sec"].max() - sample_meta["start_sec"].min()
            )
        for local_row in rows:
            row = selected_meta.iloc[local_row]
            state = machine.update(float(row["center_sec"]), probabilities[local_row])
            unknown_count += int(state["pred_smoothed"] == "unknown")
            if state["event_id"]:
                truth_event = str(row.get("event_id", ""))
                if truth_event and truth_event in sample_true:
                    if truth_event not in detected_events:
                        onset = float(
                            sample_meta.loc[
                                sample_meta["event_id"].astype(str) == truth_event,
                                "start_sec",
                            ].min()
                        )
                        latencies.append(float(row["center_sec"]) - onset)
                    detected_events.add(truth_event)
                else:
                    false_events += 1

    event_recall = len(detected_events) / len(true_events) if true_events else 0.0
    result = {
        "split": args.split,
        "window_count": int(len(idx)),
        "fall_recall": recall_score(y[idx] == fall_id, pred == fall_id, zero_division=0),
        "fall_precision": precision_score(y[idx] == fall_id, pred == fall_id, zero_division=0),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y[idx], pred, labels=labels).tolist(),
        "breakdown": breakdown,
        "event_metrics": {
            "true_fall_events": len(true_events),
            "detected_fall_events": len(detected_events),
            "event_recall": event_recall,
            "false_alarms_per_hour": false_events / max(duration_sec / 3600.0, 1e-9),
            "false_alarms_per_hour_is_reference_only": True,
            "mean_detection_latency_sec": float(np.mean(latencies)) if latencies else None,
            "unknown_rate": unknown_count / len(selected_meta) if len(selected_meta) else 0.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        classification_report(
            y[idx], pred, labels=labels, target_names=names, zero_division=0
        )
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
