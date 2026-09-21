"""Frozen first-stage CSI -> 2D pose models (ported from the Round4 notebook)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.pose_geometry import normalize_pose

FEATURE_DIM = 315
SEQ_LEN = 20
FEATURE_CLIP = 10.0


class ReceiverSubcarrierEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def _receiver_features(encoder: ReceiverSubcarrierEncoder, x: torch.Tensor) -> torch.Tensor:
    """(B,T,315) -> (B,T,195): 3 receivers x 64 + 3 receiver medians."""
    b, t, d = x.shape
    assert d == FEATURE_DIM
    centered = x[:, :, 0:156].reshape(b, t, 3, 52)
    delta = x[:, :, 156:312].reshape(b, t, 3, 52)
    medians = x[:, :, 312:315]
    rx_input = torch.stack([centered, delta], dim=3).reshape(b * t * 3, 2, 52)
    emb = encoder(rx_input).reshape(b, t, 3 * 64)
    return torch.cat([emb, medians], dim=-1)


def _attention_pool(attention: nn.Module, seq: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(attention(seq).squeeze(-1), dim=1)
    return torch.sum(seq * weights.unsqueeze(-1), dim=1)


class CareWavePoseModel(nn.Module):
    """v1: predicts normalized pose, pelvis center and log body scale."""

    def __init__(self, lstm_hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.receiver_encoder = ReceiverSubcarrierEncoder()
        self.frame_projection = nn.Sequential(
            nn.Linear(195, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(192, 128), nn.LayerNorm(128), nn.GELU(),
        )
        self.temporal_model = nn.LSTM(128, lstm_hidden, num_layers=2, batch_first=True,
                                      bidirectional=True, dropout=dropout)
        td = lstm_hidden * 2
        self.attention = nn.Sequential(nn.Linear(td, 128), nn.Tanh(), nn.Linear(128, 1))
        self.context_network = nn.Sequential(nn.Linear(td, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout))
        self.pose_head = nn.Linear(256, 66)
        self.center_head = nn.Linear(256, 2)
        self.log_scale_head = nn.Linear(256, 1)

    def forward(self, x):
        frames = self.frame_projection(_receiver_features(self.receiver_encoder, x))
        seq, _ = self.temporal_model(frames)
        ctx = self.context_network(_attention_pool(self.attention, seq))
        return {"pose": self.pose_head(ctx), "center": self.center_head(ctx), "log_scale": self.log_scale_head(ctx)}


class CareWaveDirectPoseModel(nn.Module):
    """v2: predicts raw 0-1 pose directly (sigmoid) with a normalized auxiliary head."""

    def __init__(self, lstm_hidden: int = 96, dropout: float = 0.35):
        super().__init__()
        self.receiver_encoder = ReceiverSubcarrierEncoder()
        self.frame_projection = nn.Sequential(
            nn.Linear(195, 160), nn.LayerNorm(160), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(160, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(dropout),
        )
        self.temporal_model = nn.LSTM(96, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        td = lstm_hidden * 2
        self.attention = nn.Sequential(nn.Linear(td, 96), nn.Tanh(), nn.Linear(96, 1))
        self.context_network = nn.Sequential(
            nn.Linear(td, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(192, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
        )
        self.raw_pose_head = nn.Linear(128, 66)
        self.normalized_pose_head = nn.Linear(128, 66)

    def forward(self, x):
        frames = self.frame_projection(_receiver_features(self.receiver_encoder, x))
        seq, _ = self.temporal_model(frames)
        ctx = self.context_network(_attention_pool(self.attention, seq))
        return {"raw_pose": torch.sigmoid(self.raw_pose_head(ctx)), "normalized_pose": self.normalized_pose_head(ctx),
                "context": ctx}


VARIANTS = {"v1": CareWavePoseModel, "v2": CareWaveDirectPoseModel}


def apply_scaler(features: np.ndarray, scaler_path: Path) -> np.ndarray:
    """Train-only robust scaler: clip((x - median) / iqr, -10, 10). Never re-fit."""
    with np.load(scaler_path, allow_pickle=True) as s:
        median, iqr = s["median"].astype(np.float32), s["iqr"].astype(np.float32)
    assert features.shape[-1] == FEATURE_DIM == len(median)
    return np.clip((features - median) / iqr, -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32)


class FirstStagePose:
    """Frozen first-stage model. predict() maps scaled (N,20,315) to pose arrays."""

    def __init__(self, variant: str, checkpoint: Path, device: str = "cpu"):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
        self.variant = variant
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = VARIANTS[variant]()
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.epoch = ckpt.get("epoch")

    def _forward(self, x: np.ndarray, batch_size: int, stochastic: bool) -> dict[str, np.ndarray]:
        self.model.train(stochastic)
        chunks: dict[str, list[np.ndarray]] = {}
        with torch.inference_mode():
            for i in range(0, len(x), batch_size):
                out = self.model(torch.from_numpy(x[i:i + batch_size]).to(self.device))
                for k, v in out.items():
                    chunks.setdefault(k, []).append(v.cpu().numpy())
        self.model.eval()
        return {k: np.concatenate(v) for k, v in chunks.items()}

    def _to_pose(self, out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.variant == "v1":
            n = len(out["pose"])
            norm = out["pose"].reshape(n, 33, 2)
            raw = norm * np.exp(out["log_scale"]).reshape(n, 1, 1) + out["center"].reshape(n, 1, 2)
            return {"raw_xy": raw.astype(np.float32), "norm_xy": norm.astype(np.float32),
                    "center": out["center"].astype(np.float32), "log_scale": out["log_scale"].astype(np.float32)}
        raw = out["raw_pose"].reshape(-1, 33, 2)
        norm, center, scale = normalize_pose(raw)
        return {"raw_xy": raw.astype(np.float32), "norm_xy": norm, "center": center,
                "log_scale": np.log(scale)[:, None].astype(np.float32)}

    def predict(self, x: np.ndarray, batch_size: int = 512, mc_samples: int = 0) -> dict[str, np.ndarray]:
        assert x.ndim == 3 and x.shape[1:] == (SEQ_LEN, FEATURE_DIM), x.shape
        x = np.ascontiguousarray(x, dtype=np.float32)
        raw_out = self._forward(x, batch_size, stochastic=False)
        result = self._to_pose(raw_out)
        if "context" in raw_out:
            result["embedding"] = raw_out["context"].astype(np.float32)
        if mc_samples > 0:
            draws = np.stack([self._to_pose(self._forward(x, batch_size, True))["raw_xy"] for _ in range(mc_samples)])
            result["var_xy"] = draws.var(axis=0).astype(np.float32)
        else:
            result["var_xy"] = np.zeros_like(result["raw_xy"])
        return result
