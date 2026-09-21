"""Pose Motion branch: BiGRU encoder, quality gate and the Pose-only classifier."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(seq).squeeze(-1), dim=1)
        return (seq * weights.unsqueeze(-1)).sum(dim=1)


class PoseEncoder(nn.Module):
    """(B,T,F) -> (B,embed): Linear F->128 -> BiGRU(hidden) -> attention pooling."""

    def __init__(self, in_dim: int, hidden: int = 64, embed: int = 128, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, embed), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(embed, hidden, batch_first=True, bidirectional=True)
        self.pool = AttentionPool(2 * hidden)
        self.out_dim = 2 * hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.gru(self.proj(x))
        return self.pool(seq)


def summarize_quality(quality: torch.Tensor) -> torch.Tensor:
    """(B,T,Q) -> (B,2Q): temporal mean and max of every quality signal."""
    return torch.cat([quality.mean(dim=1), quality.amax(dim=1)], dim=-1)


class QualityGate(nn.Module):
    """Quality summary -> g in [0,1]; low pose quality shrinks the pose embedding."""

    def __init__(self, q_dim: int, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(q_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, q_summary: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(q_summary))


class PoseBranch(nn.Module):
    def __init__(self, in_dim: int, q_channels: int, hidden: int = 64, dropout: float = 0.3, use_gate: bool = True):
        super().__init__()
        self.encoder = PoseEncoder(in_dim, hidden=hidden, dropout=dropout)
        self.use_gate = use_gate
        self.gate = QualityGate(2 * q_channels) if use_gate else None
        self.embed_dim = self.encoder.out_dim
        self.q_summary_dim = 2 * q_channels

    def forward(self, pose: torch.Tensor, quality: torch.Tensor):
        emb = self.encoder(pose)
        q = summarize_quality(quality)
        gate = self.gate(q) if self.use_gate else torch.ones(len(emb), 1, device=emb.device)
        return gate * emb, gate, q


class PoseOnlyClassifier(nn.Module):
    """E2 baseline: pose motion + quality -> one fall logit."""

    def __init__(self, in_dim: int, q_channels: int, hidden: int = 64, dropout: float = 0.3,
                 use_gate: bool = True, use_quality_input: bool = True):
        super().__init__()
        self.branch = PoseBranch(in_dim, q_channels, hidden, dropout, use_gate)
        self.use_quality_input = use_quality_input
        head_in = self.branch.embed_dim + (self.branch.q_summary_dim if use_quality_input else 0)
        self.head = nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, pose: torch.Tensor, quality: torch.Tensor):
        emb, gate, q = self.branch(pose, quality)
        feats = torch.cat([emb, q], dim=-1) if self.use_quality_input else emb
        return self.head(feats).squeeze(-1), {"gate": gate.squeeze(-1)}
