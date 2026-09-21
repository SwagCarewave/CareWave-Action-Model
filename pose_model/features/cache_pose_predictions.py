"""Cache first-stage predicted poses at 10 Hz (frozen model, sliding stride 1)."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.first_stage_pose import SEQ_LEN, FirstStagePose, apply_scaler

CHECKPOINTS = {"v1": "trained_model_315/carewave_315_best.pt", "v2": "trained_model_315_v2/carewave_315_v2_best.pt"}
SCALER = "model_ready_315/csi_robust_scaler_train_only.npz"
FPS = 10
QUALITY_NAMES = ["mc_var_mean", "mc_var_p90", "out_of_range_ratio", "jitter"]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def split_segments(time_sec: np.ndarray, max_gap: float) -> list[tuple[int, int]]:
    """Half-open [start, end) index ranges with consecutive gaps <= max_gap."""
    breaks = np.where(np.diff(time_sec) > max_gap + 1e-6)[0] + 1
    edges = [0, *breaks.tolist(), len(time_sec)]
    return list(zip(edges[:-1], edges[1:]))


def predict_recording(model: FirstStagePose, scaled: np.ndarray, time_sec: np.ndarray, max_gap: float,
                      stride: int, mc_samples: int) -> dict[str, np.ndarray]:
    n = len(scaled)
    out = {"raw_xy": np.zeros((n, 33, 2), np.float32), "norm_xy": np.zeros((n, 33, 2), np.float32),
           "center": np.zeros((n, 2), np.float32), "log_scale": np.zeros((n, 1), np.float32),
           "var_xy": np.zeros((n, 33, 2), np.float32)}
    if model.variant == "v2":
        out["embedding"] = np.zeros((n, 128), np.float32)
    valid = np.zeros(n, bool)
    windows, targets = [], []
    for start, end in split_segments(time_sec, max_gap):
        if end - start < SEQ_LEN:
            continue
        view = np.lib.stride_tricks.sliding_window_view(scaled[start:end], SEQ_LEN, axis=0)
        view = view.transpose(0, 2, 1)[::stride]
        windows.append(view)
        targets.append(start + np.arange(len(view)) * stride + SEQ_LEN - 1)
    if windows:
        idx = np.concatenate(targets)
        pred = model.predict(np.concatenate(windows), mc_samples=mc_samples)
        for key in out:
            out[key][idx] = pred[key]
        valid[idx] = True
    out["valid_mask"] = valid
    return out


def pose_quality(pred: dict[str, np.ndarray], mc_samples: int) -> np.ndarray:
    n, valid = len(pred["raw_xy"]), pred["valid_mask"]
    q = np.zeros((n, len(QUALITY_NAMES)), np.float32)
    if mc_samples > 0:
        v = pred["var_xy"].reshape(n, -1)
        q[:, 0], q[:, 1] = v.mean(1), np.percentile(v, 90, axis=1)
    raw = pred["raw_xy"].reshape(n, -1)
    q[:, 2] = ((raw < 0) | (raw > 1)).mean(1)
    step = np.linalg.norm(np.diff(pred["raw_xy"], axis=0), axis=-1).mean(1)
    both = valid[1:] & valid[:-1]
    q[1:, 3] = np.where(both, step, 0.0)
    q[~valid] = 0.0
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True, help="round4_v1_final directory")
    ap.add_argument("--variant", choices=sorted(CHECKPOINTS), default="v2")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--inference-stride", type=int, default=1)
    ap.add_argument("--max-gap", type=float, default=0.15)
    ap.add_argument("--mc-samples", type=int, default=0, help="MC-dropout passes for pred_var_xy (0 = off)")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--oof-dir", type=Path, default=None,
                    help="folds.csv + fold*/best.pt from train_first_stage_oof.py; each recording is predicted by the "
                         "model that did not train on it (v2 architecture, first_stage_split='oof')")
    args = ap.parse_args()

    root = args.data_root
    ckpt = args.checkpoint or root / CHECKPOINTS[args.variant]
    manifest = pd.read_csv(args.manifest or root / "model_split_315" / "all_split_manifest.csv", encoding="utf-8-sig")
    fold_of: dict[str, int] = {}
    if args.oof_dir:
        assert args.variant == "v2", "OOF retraining reproduces the v2 recipe"
        fold_of = dict(pd.read_csv(args.oof_dir / "folds.csv").values)
        manifest = manifest[manifest.sample_id.isin(fold_of)]
    out_dir = args.output or Path("data/pose_predictions") / (f"{args.variant}_oof" if args.oof_dir else args.variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.limit:
        manifest = manifest.head(args.limit)

    fold_models: dict[int, FirstStagePose] = {}
    model = None if args.oof_dir else FirstStagePose(args.variant, ckpt)
    scaler_path = root / SCALER
    source_checkpoint = f"{ckpt.name}#{file_hash(ckpt)}" if not args.oof_dir else f"oof:{args.oof_dir.name}"
    scaler_id = f"{scaler_path.name}#{file_hash(scaler_path)}"
    print(f"variant={args.variant} checkpoint={source_checkpoint} scaler={scaler_id} stride={args.inference_stride}")

    rows = []
    for rec in manifest.itertuples(index=False):
        matches = list((root / "features_315").glob(f"*/{rec.sample_id}_features_315.npz"))
        assert len(matches) == 1, f"{rec.sample_id}: expected one features file, found {len(matches)}"
        with np.load(matches[0], allow_pickle=True) as z:
            feats, time_sec = z["features"].astype(np.float32), z["time_sec"].astype(np.float64)
        assert np.all(np.diff(time_sec) > 0), f"{rec.sample_id}: timestamps not increasing"
        if args.oof_dir:
            fold = int(fold_of[rec.sample_id])
            if fold not in fold_models:
                fold_models[fold] = FirstStagePose("v2", args.oof_dir / f"fold{fold}" / "best.pt")
            model = fold_models[fold]
        pred = predict_recording(model, apply_scaler(feats, scaler_path), time_sec, args.max_gap,
                                 args.inference_stride, args.mc_samples)
        quality = pose_quality(pred, args.mc_samples)
        valid = pred["valid_mask"]
        for key in ("raw_xy", "norm_xy", "center", "log_scale", "var_xy"):
            assert np.isfinite(pred[key]).all(), f"{rec.sample_id}: non-finite {key}"
        np.savez_compressed(
            out_dir / f"{rec.sample_id}.npz",
            sample_id=rec.sample_id, time_sec=time_sec.astype(np.float32),
            frame_index=np.rint(time_sec * FPS).astype(np.int32),
            pred_raw_xy=pred["raw_xy"], pred_norm_xy=pred["norm_xy"], pred_center=pred["center"],
            pred_log_scale=pred["log_scale"], pred_var_xy=pred["var_xy"], pose_quality=quality,
            pose_quality_names=np.array(QUALITY_NAMES), valid_mask=valid,
            **({"pred_embedding": pred["embedding"]} if "embedding" in pred else {}),
            source_checkpoint=source_checkpoint, scaler_id=scaler_id, variant=args.variant,
            first_stage_split="oof" if args.oof_dir else rec.model_split,
        )
        rows.append({
            "sample_id": rec.sample_id, "person": rec.person, "action": rec.action,
            "first_stage_split": "oof" if args.oof_dir else rec.model_split, "path": f"{rec.sample_id}.npz", "frames": len(time_sec),
            "valid_frames": int(valid.sum()), "segments": len(split_segments(time_sec, args.max_gap)),
            "duration_sec": float(time_sec[-1] - time_sec[0]),
            "out_of_range_ratio": float(quality[valid, 2].mean()) if valid.any() else float("nan"),
            "variant": args.variant, "source_checkpoint": source_checkpoint, "scaler_id": scaler_id,
        })
        print(f"[OK] {rec.sample_id} [{rec.model_split}] frames={len(time_sec)} valid={int(valid.sum())}")

    pd.DataFrame(rows).to_csv(out_dir / "index.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(rows)} recordings to {out_dir}")


if __name__ == "__main__":
    main()
