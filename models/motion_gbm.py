"""Gradient-boosting fall detector on within-window motion statistics.

The deep classifier sees the absolute stream spectra and learns recording-date
fingerprints (leave-one-date-out AUC 0.43). These features only describe how much and
when the signal changes inside the 3 s window, so the static room spectrum is not an input.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from csi_dataset import stream_shape_dims


def motion_stats(x: np.ndarray, data_cfg: dict) -> np.ndarray:
    """(windows, frames, 318) unscaled stream_split windows -> (windows, 66) features."""
    dims = stream_shape_dims(data_cfg)
    n_sub = int(data_cfg.get("subcarriers_per_rx", 52))
    motion = x[:, :, dims:]
    shape = x[:, :, :dims].reshape(len(x), x.shape[1], -1, n_sub)
    k = x.shape[1] // 3
    shape_shift = np.linalg.norm(shape[:, -k:].mean(1) - shape[:, :k].mean(1), axis=-1)
    shape_spread = shape.std(axis=1).mean(axis=-1)
    parts = [motion.mean(1), motion.std(1), motion.max(1), motion.min(1),
             motion[:, :k].mean(1), motion[:, k:2 * k].mean(1), motion[:, -k:].mean(1), shape_shift, shape_spread]
    return np.concatenate(parts, axis=1)


def make_gbm(seed: int = 42):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=seed)


class MotionGBMModule(nn.Module):
    """Drop-in for CSIActionClassifier: takes the scaled windows the pipeline already
    produces, undoes the robust scaler, and returns fall logits (use torch.sigmoid)."""

    def __init__(self, checkpoint: dict):
        super().__init__()
        self.gbm = checkpoint["gbm"]
        self.data_cfg = checkpoint["data_config"]
        self.median = np.asarray(checkpoint["scaler_median"], dtype=np.float32)
        self.iqr = np.asarray(checkpoint["scaler_iqr"], dtype=np.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x.detach().cpu().numpy() * self.iqr + self.median
        probability = self.gbm.predict_proba(motion_stats(raw, self.data_cfg))[:, 1].clip(1e-6, 1 - 1e-6)
        return torch.logit(torch.from_numpy(probability).to(x.device, torch.float32))
