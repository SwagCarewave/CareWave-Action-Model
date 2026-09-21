"""Reference check: pose cache from the separate carewaveTest CSI->pose model (read-only use)."""

from __future__ import annotations

import argparse
import sys

sys.dont_write_bytecode = True

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.cache_pose_predictions import QUALITY_NAMES, pose_quality
from features.pose_geometry import normalize_pose

TEST6 = {"yena_stand_normal_08", "sujin_stand_normal_06", "yena_fall_normal_11", "sujin_fall_stay_down_01",
         "yena_slow_lie_down_01", "sujin_lie_down_normal_01"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--carewavetest", type=Path, default=Path("D:/project/carewaveTest"))
    ap.add_argument("--raw-csi", type=Path, default=Path("D:/project/carewaveAI/carewave-csi-pose-estimation/carewave_dataset/raw/csi"))
    ap.add_argument("--index", type=Path, default=Path("data/pose_predictions/v2_oofb/index.csv"))
    ap.add_argument("--output", type=Path, default=Path("data/pose_predictions/cwt"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(args.carewavetest))
    import os
    os.chdir(args.carewavetest)
    from infer_realtime import RealtimePosePredictor, stream_ticks
    os.chdir(Path(__file__).resolve().parents[1])

    info = pd.read_csv(args.carewavetest / "processed/dataset/dataset_info.csv", encoding="utf-8-sig")
    split_of = info.drop_duplicates("sample_id").set_index("sample_id")["split"].to_dict()
    index = pd.read_csv(args.index, encoding="utf-8-sig")
    if args.limit:
        index = index.head(args.limit)
    ckpt_path = args.carewavetest / "checkpoints" / "csi_pose_final.pth"
    ckpt_id = f"{ckpt_path.name}#{hashlib.sha256(ckpt_path.read_bytes()).hexdigest()[:16]}"
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for rec in index.itertuples():
        raw = list(args.raw_csi.glob(f"*/{rec.sample_id}_csi_raw.csv"))
        if len(raw) != 1:
            print(f"[SKIP] {rec.sample_id}: raw csi files found = {len(raw)}")
            continue
        os.chdir(args.carewavetest)
        pred = RealtimePosePredictor("csi_pose_final.pth")
        ticks, pts = [], []
        for k, vec in stream_ticks(raw[0]):
            p = pred.push_tick(vec)
            if p is not None:
                ticks.append(k); pts.append(p)
        os.chdir(Path(__file__).resolve().parents[1])
        if not ticks:
            print(f"[SKIP] {rec.sample_id}: no prediction")
            continue
        ticks = np.array(ticks)
        n = int(ticks.max()) + 1
        raw_xy = np.zeros((n, 33, 2), np.float32)
        valid = np.zeros(n, bool)
        raw_xy[ticks] = np.stack(pts).astype(np.float32)
        raw_xy[..., 1] = 1.0 - raw_xy[..., 1]
        valid[ticks] = True
        norm, center, scale = normalize_pose(raw_xy)
        out = {"raw_xy": raw_xy, "var_xy": np.zeros_like(raw_xy), "valid_mask": valid}
        quality = pose_quality(out, 0)
        t = (np.arange(n) * 0.1).astype(np.float64)
        lineage = "test6" if rec.sample_id in TEST6 else {"train": "train", "val": "val", "test": "train"}.get(split_of.get(rec.sample_id, "train"), "train")
        np.savez_compressed(
            args.output / f"{rec.sample_id}.npz", sample_id=rec.sample_id, time_sec=t.astype(np.float32),
            frame_index=np.arange(n, dtype=np.int32), pred_raw_xy=raw_xy, pred_norm_xy=norm, pred_center=center,
            pred_log_scale=np.log(scale)[:, None], pred_var_xy=out["var_xy"], pose_quality=quality,
            pose_quality_names=np.array(QUALITY_NAMES), valid_mask=valid, source_checkpoint=ckpt_id, scaler_id="in_checkpoint",
            variant="cwt", first_stage_split=lineage)
        rows.append({"sample_id": rec.sample_id, "person": rec.person, "action": rec.action, "first_stage_split": lineage,
                     "path": f"{rec.sample_id}.npz", "frames": n, "valid_frames": int(valid.sum()), "variant": "cwt",
                     "source_checkpoint": ckpt_id, "scaler_id": "in_checkpoint"})
        print(f"[OK] {rec.sample_id} [{lineage}] frames={n} valid={int(valid.sum())}", flush=True)
    pd.DataFrame(rows).to_csv(args.output / "index.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(rows)} recordings to {args.output}")


if __name__ == "__main__":
    main()
