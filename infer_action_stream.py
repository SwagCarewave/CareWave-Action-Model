"""Run CSI-stream inference for one raw recording."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from csi_dataset import load_csi_10hz, scale_features
from models.csi_encoder import CSIActionClassifier


def main():
    p = argparse.ArgumentParser(); p.add_argument("raw_csv", type=Path); p.add_argument("--config", type=Path, default=Path("configs/action_fusion.yaml")); p.add_argument("--checkpoint", type=Path, default=Path("outputs/csi_stream/best.pt")); p.add_argument("--output", type=Path, default=Path("outputs/csi_stream/inference.csv")); args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); data, prep = cfg["data"], cfg["preprocessing"]
    frame, raw_features, _ = load_csi_10hz(args.raw_csv, data); features = scale_features(raw_features, Path(data["scaler_path"]), float(prep["clip_min"]), float(prep["clip_max"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False); model = CSIActionClassifier(**checkpoint["model_config"]).to(device); model.load_state_dict(checkpoint["model_state"]); model.eval()
    window = round(data["window_seconds"] * data["target_fps"]); stride = max(1, round(cfg.get("postprocess", {}).get("inference_stride_seconds", .2) * data["target_fps"])); rows = []
    with torch.no_grad():
        for start in range(0, len(features) - window + 1, stride):
            probability = float(torch.sigmoid(model(torch.from_numpy(features[start:start + window][None]).to(device))).item()); rows.append({"start_sec": float(frame.time_sec.iloc[start]), "end_sec": float(frame.time_sec.iloc[start + window - 1] + 1 / data["target_fps"]), "fall_probability": probability, "prediction": "falling" if probability >= checkpoint.get("threshold", .5) else "standing"})
    args.output.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig"); print(f"Saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__": main()
