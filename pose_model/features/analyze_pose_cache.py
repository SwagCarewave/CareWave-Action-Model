"""Diagnostic: cached first-stage pose error vs MediaPipe pose, by first-stage lineage."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--variant", default="v2")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/pose_predictions"))
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    cache_dir = args.cache_dir / args.variant
    index = pd.read_csv(cache_dir / "index.csv", encoding="utf-8-sig")

    rows = []
    for rec in index.itertuples():
        sync = list((args.data_root / "synchronized_315").glob(f"*/{rec.sample_id}_synchronized_315.npz"))[0]
        with np.load(sync, allow_pickle=True) as g, np.load(cache_dir / rec.path, allow_pickle=True) as c:
            t_c, valid = c["time_sec"].astype(np.float64), c["valid_mask"]
            pos = np.clip(np.searchsorted(t_c, g["csi_time_sec"]), 0, len(t_c) - 1)
            ok = valid[pos] & (np.abs(t_c[pos] - g["csi_time_sec"]) < 0.05)
            vis = g["visibility"][ok] >= 0.3
            err = np.linalg.norm(c["pred_raw_xy"][pos][ok] - g["pose_xy"][ok], axis=-1)
            rows.append({"sample_id": rec.sample_id, "person": rec.person, "action": rec.action,
                         "first_stage_split": rec.first_stage_split, "frames": int(ok.sum()),
                         "mpjpe": float(err.mean()), "mpjpe_visible": float(err[vis].mean()) if vis.any() else np.nan,
                         "pck_005": float((err <= 0.05).mean())})
    df = pd.DataFrame(rows)
    out = args.output or cache_dir / "error_vs_mediapipe.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[{args.variant}] per-lineage error vs MediaPipe (recording mean)")
    print(df.groupby("first_stage_split")[["mpjpe", "mpjpe_visible", "pck_005"]].agg(["mean", "median"]).round(4).to_string())
    print("worst 5 recordings:", df.nlargest(5, "mpjpe")[["sample_id", "first_stage_split", "mpjpe"]].round(3).values.tolist())


if __name__ == "__main__":
    main()
