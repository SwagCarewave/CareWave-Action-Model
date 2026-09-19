"""Run offline sliding-window inference for one raw CSI CSV."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.csi_dataset import FEATURE_COLUMNS, preprocess_raw
from models.csi_encoder import CSIActionClassifier
from postprocess_state_machine import ActionStateMachine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/csi_stream/best.pt"))
    parser.add_argument("--stride-seconds", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/csi_stream/predictions.csv"))
    args = parser.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fps = int(ckpt["target_fps"])
    window = int(round(float(ckpt["window_seconds"]) * fps))
    post_cfg = ckpt.get("postprocess", ckpt.get("config", {}).get("postprocess", {}))
    stride_seconds = args.stride_seconds if args.stride_seconds is not None else float(post_cfg.get("inference_stride_seconds", 0.2))
    stride = max(1, int(round(stride_seconds * fps)))
    frame_df, _ = preprocess_raw(args.raw_csv, fps)
    values = frame_df[FEATURE_COLUMNS].to_numpy(np.float32)
    starts = list(range(0, len(values) - window + 1, stride))
    if not starts:
        raise SystemExit(f"Recording needs at least {window} frames")
    x = np.stack([values[s : s + window] for s in starts])
    x = (x - ckpt["mean"][None, None, :]) / ckpt["std"][None, None, :]
    model = CSIActionClassifier(num_classes=len(ckpt["class_names"]), **ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"]); model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()
    rows = []
    machine = ActionStateMachine(ckpt["class_names"], post_cfg)
    for i, start in enumerate(starts):
        pred = int(probabilities[i].argmax())
        center_sec = (start + window / 2) / fps
        state = machine.update(center_sec, probabilities[i])
        row = {
            "time_sec": center_sec, "start_sec": start / fps,
            "end_sec": (start + window) / fps,
            **{f"prob_{name}": float(probabilities[i, j]) for j, name in enumerate(ckpt["class_names"])},
            **state,
        }
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(rows)} predictions: {args.output}")


if __name__ == "__main__":
    main()
