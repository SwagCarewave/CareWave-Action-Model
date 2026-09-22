"""Evaluate raw and temporally confirmed CSI-only binary predictions."""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from models.csi_encoder import CSIActionClassifier


def make_report(truth: np.ndarray, prediction: np.ndarray) -> dict:
    report = classification_report(
        truth, prediction, labels=[0, 1], target_names=["standing", "falling"],
        output_dict=True, zero_division=0,
    )
    report["confusion_matrix"] = confusion_matrix(truth, prediction, labels=[0, 1]).tolist()
    return report


def apply_temporal_confirmation(metadata, probabilities, alpha, threshold, confirm_count, confirm_window):
    smoothed = np.zeros(len(probabilities), dtype=np.float64)
    confirmed = np.zeros(len(probabilities), dtype=np.int64)
    work = metadata.copy()
    work["_position"] = np.arange(len(work))
    work["_probability"] = probabilities
    for _, group in work.groupby("sample_id", sort=False):
        group = group.sort_values("center_sec")
        previous_ema = None
        recent = deque(maxlen=confirm_window)
        for position, probability in zip(group["_position"], group["_probability"]):
            position, probability = int(position), float(probability)
            ema = probability if previous_ema is None else alpha * probability + (1.0 - alpha) * previous_ema
            previous_ema = ema
            smoothed[position] = ema
            recent.append(int(ema >= threshold))
            confirmed[position] = int(len(recent) >= confirm_window and sum(recent) >= confirm_count)
    return smoothed, confirmed


def source_reports(metadata, truth, prediction):
    if "binary_source" not in metadata.columns:
        return {}
    result = {}
    sources = metadata["binary_source"].fillna("unknown").astype(str).to_numpy()
    for source in sorted(set(sources)):
        mask = sources == source
        result[source] = {
            "support": int(mask.sum()),
            "correct": int((truth[mask] == prediction[mask]).sum()),
            "predicted_falling": int(prediction[mask].sum()),
            "accuracy": float((truth[mask] == prediction[mask]).mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_fusion.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/csi_stream/best.pt"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", type=Path, default=Path("outputs/csi_stream/evaluation.json"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_dir = Path(cfg["data"]["processed_dir"])
    x = np.load(data_dir / "X.npy").astype(np.float32)
    y = np.load(data_dir / "y.npy").astype(np.int64)
    metadata = pd.read_csv(data_dir / "metadata.csv")
    indices = np.flatnonzero(metadata["split"].to_numpy() == args.split)
    if not len(indices):
        raise SystemExit(f"No {args.split} windows")

    split_x, split_y = x[indices], y[indices]
    split_metadata = metadata.iloc[indices].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CSIActionClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(split_x)),
        batch_size=int(cfg["training"].get("batch_size", 64)), shuffle=False,
    )
    probabilities = []
    with torch.no_grad():
        for (batch_x,) in loader:
            probabilities.extend(torch.sigmoid(model(batch_x.to(device))).cpu().tolist())
    probabilities = np.asarray(probabilities, dtype=np.float64)

    checkpoint_threshold = float(checkpoint.get("threshold", 0.5))
    raw_prediction = (probabilities >= checkpoint_threshold).astype(np.int64)
    raw_report = make_report(split_y, raw_prediction)
    raw_report["threshold"] = checkpoint_threshold
    raw_report["source_breakdown"] = source_reports(split_metadata, split_y, raw_prediction)

    post_cfg = cfg.get("postprocess", {})
    if bool(post_cfg.get("enabled", True)):
        alpha = float(post_cfg.get("ema_alpha", 0.6))
        post_threshold = max(checkpoint_threshold, float(post_cfg.get("fall_threshold", checkpoint_threshold)))
        confirm_count = int(post_cfg.get("confirm_count", 3))
        confirm_window = int(post_cfg.get("confirm_window", 5))
        if not 1 <= confirm_count <= confirm_window:
            raise SystemExit("postprocess.confirm_count must be between 1 and confirm_window")
        smoothed_probability, post_prediction = apply_temporal_confirmation(
            split_metadata, probabilities, alpha, post_threshold, confirm_count, confirm_window
        )
        post_report = make_report(split_y, post_prediction)
        post_report.update({
            "ema_alpha": alpha, "threshold": post_threshold, "confirm_count": confirm_count,
            "confirm_window": confirm_window,
            "source_breakdown": source_reports(split_metadata, split_y, post_prediction),
        })
    else:
        smoothed_probability, post_prediction = probabilities.copy(), raw_prediction.copy()
        post_report = None

    rows = split_metadata.copy()
    rows["true_label_id"] = split_y
    rows["fall_probability"] = probabilities
    rows["raw_prediction"] = raw_prediction
    rows["smoothed_probability"] = smoothed_probability
    rows["postprocessed_prediction"] = post_prediction
    prediction_path = args.output.with_name(args.output.stem + "_predictions.csv")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    report = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "raw_window_metrics": raw_report,
        "postprocessed_window_metrics": post_report,
        "prediction_csv": str(prediction_path),
        "warning": "Threshold was selected on validation data; test data was not used for selection. Validation has few hard negatives.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
