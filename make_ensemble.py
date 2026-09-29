"""Combine seed checkpoints into one probability-averaging ensemble checkpoint.

The decision threshold is re-selected on validation windows with the same rule as
train_action_classifier.py (best F1 among thresholds meeting the recall floor).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from models.loading import build_model
from train_action_classifier import choose_threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--members", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_dir = Path(cfg["data"]["processed_dir"])
    metadata = pd.read_csv(data_dir / "metadata.csv")
    val = np.flatnonzero(metadata["split"].to_numpy() == "val")
    x = torch.from_numpy(np.load(data_dir / "X.npy", mmap_mode="r")[val].astype(np.float32))
    y = np.load(data_dir / "y.npy")[val].astype(np.int64)

    members = [torch.load(path, map_location="cpu", weights_only=False) for path in args.members]
    checkpoint = {
        "ensemble_members": [{"model_config": m["model_config"], "model_state": m["model_state"],
                              "seed": m.get("seed"), "source": str(p)} for m, p in zip(members, args.members)],
        "class_names": members[0]["class_names"], "config": cfg,
    }
    model = build_model(checkpoint)
    with torch.no_grad():
        probability = torch.sigmoid(torch.cat([model(x[i:i + 256]) for i in range(0, len(x), 256)])).numpy()
    training = cfg["training"]
    threshold, selected, table = choose_threshold(y, probability, [float(v) for v in training["threshold_candidates"]],
                                                  float(training.get("recall_floor", 0.85)))
    checkpoint["threshold"] = threshold
    checkpoint["threshold_selection"] = {"recall_floor": float(training.get("recall_floor", 0.85)),
                                         "candidates": table, "selected": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Saved {len(members)}-member ensemble to {args.output}; val threshold={threshold:.2f} "
          f"precision={selected['precision']:.4f} recall={selected['recall']:.4f} f1={selected['f1']:.4f}")


if __name__ == "__main__":
    main()
