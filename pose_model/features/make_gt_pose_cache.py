"""DIAGNOSTIC ONLY: pose cache built from MediaPipe pose (not the first-stage prediction)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.cache_pose_predictions import QUALITY_NAMES
from features.pose_geometry import normalize_pose


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--source-index", type=Path, default=Path("data/pose_predictions/v2_oofb/index.csv"))
    ap.add_argument("--output", type=Path, default=Path("data/pose_predictions/gt_diag"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.source_index, encoding="utf-8-sig")
    rows = []
    for rec in index.itertuples():
        f = list((args.data_root / "synchronized_315").glob(f"*/{rec.sample_id}_synchronized_315.npz"))[0]
        with np.load(f, allow_pickle=True) as z:
            raw, t = z["pose_xy"].astype(np.float32), z["time_sec"].astype(np.float64)
        valid = np.isfinite(raw).all(axis=(1, 2))
        raw = np.nan_to_num(raw)
        norm, center, scale = normalize_pose(raw)
        n = len(t)
        q = np.zeros((n, len(QUALITY_NAMES)), np.float32)
        q[1:, 3] = np.linalg.norm(np.diff(raw, axis=0), axis=-1).mean(1)
        q[~valid] = 0.0
        np.savez_compressed(
            args.output / f"{rec.sample_id}.npz", sample_id=rec.sample_id, time_sec=t.astype(np.float32),
            frame_index=np.rint(t * 10).astype(np.int32), pred_raw_xy=raw, pred_norm_xy=norm, pred_center=center,
            pred_log_scale=np.log(scale)[:, None], pred_var_xy=np.zeros_like(raw), pose_quality=q,
            pose_quality_names=np.array(QUALITY_NAMES), valid_mask=valid, source_checkpoint="mediapipe_gt_diagnostic",
            scaler_id="none", variant="gt_diag", first_stage_split="oof")
        rows.append({**index.loc[index.sample_id == rec.sample_id].iloc[0].to_dict(), "path": f"{rec.sample_id}.npz",
                     "frames": n, "valid_frames": int(valid.sum()), "variant": "gt_diag",
                     "source_checkpoint": "mediapipe_gt_diagnostic", "scaler_id": "none"})
    pd.DataFrame(rows).to_csv(args.output / "index.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(rows)} diagnostic GT caches to {args.output}")


if __name__ == "__main__":
    main()
