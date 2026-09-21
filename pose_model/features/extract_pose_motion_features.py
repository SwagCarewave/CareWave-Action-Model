"""Pose motion features from cached first-stage predictions (per-frame, 10 Hz)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.pose_geometry import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, normalize_pose

CORE_JOINTS = [0, 11, 12, 23, 24, 25, 26, 27, 28, 31, 32, 15]
BONES = [(11, 12), (23, 24), (11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28)]
LR_PAIRS = [(2, 3), (4, 5), (6, 7)]
MAX_GAP = 0.15
QUALITY_WINDOW = 10


def _dt(time_sec: np.ndarray) -> np.ndarray:
    return np.diff(time_sec, prepend=time_sec[0]).astype(np.float32)


def _diff_with_mask(x: np.ndarray, time_sec: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First derivative of x [L,...] using real dt; returns (dx/dt, ok[L])."""
    dt = _dt(time_sec)
    ok = np.zeros(len(x), bool)
    ok[1:] = valid[1:] & valid[:-1] & (dt[1:] > 0) & (dt[1:] <= MAX_GAP)
    d = np.zeros_like(x)
    safe_dt = np.where(dt > 0, dt, 1.0).reshape((-1,) + (1,) * (x.ndim - 1))
    d[1:] = (x[1:] - x[:-1]) / safe_dt[1:]
    d[~ok] = 0.0
    return d, ok


def _trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    csum = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - window, 0)
    return (csum[idx] - csum[lo]) / (idx - lo).reshape((-1,) + (1,) * (x.ndim - 1))


def ema_smooth(raw: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    """Causal exponential smoothing s_t = a*x_t + (1-a)*s_{t-1}; restarts after invalid frames (past-only, streaming-safe)."""
    out, state = raw.copy(), None
    for i in range(len(raw)):
        if not valid[i]:
            state = None
            continue
        state = raw[i] if state is None else alpha * raw[i] + (1 - alpha) * state
        out[i] = state
    return out


def extract(cache: dict[str, np.ndarray], mode: str = "full", ema: float = 0.0) -> dict[str, np.ndarray]:
    raw = cache["pred_raw_xy"].astype(np.float32)
    norm = cache["pred_norm_xy"].astype(np.float32)
    if ema > 0:
        raw = ema_smooth(raw, cache["valid_mask"].astype(bool), ema)
        norm = normalize_pose(raw)[0]
    time_sec = cache["time_sec"].astype(np.float64)
    valid = cache["valid_mask"].astype(bool)
    joints = list(range(33)) if mode in ("full", "full_emb", "emb") else CORE_JOINTS
    n = len(raw)

    pose = norm[:, joints, :].reshape(n, -1)
    vel, vel_ok = _diff_with_mask(pose, time_sec, valid)
    acc, acc_ok = _diff_with_mask(vel, time_sec, vel_ok)

    pelvis = 0.5 * (raw[:, L_HIP] + raw[:, R_HIP])
    pelvis_vel, _ = _diff_with_mask(pelvis, time_sec, valid)
    shoulder = 0.5 * (raw[:, L_SHOULDER] + raw[:, R_SHOULDER])
    trunk = shoulder - pelvis
    trunk_len = np.linalg.norm(trunk, axis=-1).clip(min=1e-4)
    angle = np.stack([trunk[:, 0] / trunk_len, -trunk[:, 1] / trunk_len], axis=-1)

    used = raw[:, joints, :]
    width = used[..., 0].max(1) - used[..., 0].min(1)
    height = (used[..., 1].max(1) - used[..., 1].min(1)).clip(min=1e-4)
    box = np.stack([np.clip(width / height, 0, 5), height], axis=-1)

    bones = np.stack([np.linalg.norm(raw[:, a] - raw[:, b], axis=-1) for a, b in BONES], axis=-1)

    feats = np.concatenate([pose, vel, acc, pelvis, pelvis_vel, angle, box, bones], axis=1).astype(np.float32)
    feats[~valid] = 0.0
    embedding = cache["pred_embedding"].astype(np.float32) if mode in ("emb", "full_emb") else None
    jn = [f"j{j}" for j in joints]
    names = ([f"norm_{j}_{c}" for j in jn for c in "xy"] + [f"vel_{j}_{c}" for j in jn for c in "xy"]
             + [f"acc_{j}_{c}" for j in jn for c in "xy"]
             + ["pelvis_x", "pelvis_y", "pelvis_vx", "pelvis_vy", "torso_sin", "torso_cos", "box_aspect", "box_height"]
             + [f"bone_{a}_{b}" for a, b in BONES])

    if embedding is not None:
        emb_names = [f"emb_{i}" for i in range(embedding.shape[1])]
        if mode == "emb":
            feats, names = embedding.copy(), emb_names
        else:
            feats, names = np.concatenate([feats, embedding], axis=1), names + emb_names
        feats[~valid] = 0.0

    q_cache = cache["pose_quality"].astype(np.float32)
    bone_mean = _trailing_mean(bones, QUALITY_WINDOW)
    bone_var = np.clip(_trailing_mean(bones ** 2, QUALITY_WINDOW) - bone_mean ** 2, 0, None)
    bone_cv = (np.sqrt(bone_var) / bone_mean.clip(min=1e-4)).mean(1)
    asym = np.mean([np.abs(bones[:, a] - bones[:, b]) / (0.5 * (bones[:, a] + bones[:, b])).clip(min=1e-4)
                    for a, b in LR_PAIRS], axis=0)
    valid_ratio = _trailing_mean(valid.astype(np.float32), QUALITY_WINDOW)
    quality = np.column_stack([q_cache, bone_cv, asym, valid_ratio]).astype(np.float32)
    quality[~valid] = 0.0
    quality_names = [*[str(x) for x in cache["pose_quality_names"]], "bone_cv", "lr_asymmetry", "valid_ratio"]

    return {"features": feats, "feature_names": np.array(names), "quality": quality,
            "quality_names": np.array(quality_names), "valid_mask": valid, "vel_ok": vel_ok, "acc_ok": acc_ok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True, help="directory produced by cache_pose_predictions.py")
    ap.add_argument("--mode", choices=["full", "core12", "emb", "full_emb"], default="full")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--ema", type=float, default=0.0, help="causal EMA alpha on the predicted pose (0 = off)")
    args = ap.parse_args()
    out_dir = args.output or Path("data/processed/pose_features") / f"{args.cache.name}_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in args.cache.glob("*.npz"))
    for path in files:
        with np.load(path, allow_pickle=True) as z:
            cache = {k: z[k] for k in z.files}
        feat = extract(cache, args.mode, args.ema)
        assert np.isfinite(feat["features"]).all() and np.isfinite(feat["quality"]).all(), path.name
        np.savez_compressed(out_dir / path.name, sample_id=cache["sample_id"], time_sec=cache["time_sec"],
                            first_stage_split=cache["first_stage_split"], **feat)
    print(f"Saved {len(files)} feature files to {out_dir} (F={feat['features'].shape[1]}, Q={feat['quality'].shape[1]})")


if __name__ == "__main__":
    main()
