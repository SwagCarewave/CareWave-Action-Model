"""Pose geometry helpers shared by the pose cache and motion features."""

from __future__ import annotations

import numpy as np

L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
MIN_SCALE = 0.05


def pelvis_center_scale(raw_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return pelvis center [...,2] and body scale [...] for raw [...,33,2] pose."""
    hip_center = 0.5 * (raw_xy[..., L_HIP, :] + raw_xy[..., R_HIP, :])
    shoulder_center = 0.5 * (raw_xy[..., L_SHOULDER, :] + raw_xy[..., R_SHOULDER, :])
    shoulder_width = np.linalg.norm(raw_xy[..., L_SHOULDER, :] - raw_xy[..., R_SHOULDER, :], axis=-1)
    hip_width = np.linalg.norm(raw_xy[..., L_HIP, :] - raw_xy[..., R_HIP, :], axis=-1)
    torso_length = np.linalg.norm(shoulder_center - hip_center, axis=-1)
    scale = np.maximum.reduce([shoulder_width, hip_width, 2.0 * torso_length, np.full_like(hip_width, MIN_SCALE)])
    return hip_center, scale


def normalize_pose(raw_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pelvis-centered, scale-normalized pose: returns (norm_xy, center, scale)."""
    center, scale = pelvis_center_scale(raw_xy)
    norm_xy = (raw_xy - center[..., None, :]) / scale[..., None, None]
    return norm_xy.astype(np.float32), center.astype(np.float32), scale.astype(np.float32)
