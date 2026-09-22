"""Reject recording/subject leakage and malformed processed arrays."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import yaml


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", type=Path, default=Path("configs/action_fusion.yaml")); args = p.parse_args(); cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); root = Path(cfg["data"]["processed_dir"]); meta = pd.read_csv(root / "metadata.csv"); x = np.load(root / "X.npy", mmap_mode="r"); y = np.load(root / "y.npy", mmap_mode="r")
    if not (len(x) == len(y) == len(meta)): raise SystemExit("Length mismatch")
    counts = meta.groupby("sample_id").split.nunique(); leaked = counts[counts > 1]
    if len(leaked): raise SystemExit(f"Recording leakage: {leaked.index.tolist()}")
    if cfg["data"]["split"].get("mode") == "subject":
        counts = meta.groupby("subject_id").split.nunique(); leaked = counts[counts > 1]
        if len(leaked): raise SystemExit(f"Subject leakage: {leaked.index.tolist()}")
    expected = (round(cfg["data"]["window_seconds"] * cfg["data"]["target_fps"]), cfg["data"]["input_dim"])
    if x.shape[1:] != expected or not np.isfinite(x).all(): raise SystemExit(f"Bad X: shape={x.shape}, finite={np.isfinite(x).all()}")
    print("OK", x.shape); print(meta.groupby(["split", "label"]).size())


if __name__ == "__main__": main()
