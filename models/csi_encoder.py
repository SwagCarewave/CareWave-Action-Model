"""CSI Stream: temporal CNN, BiLSTM and attention pooling."""

from __future__ import annotations

import torch
from torch import nn


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class CSIActionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 156,
        num_classes: int = 3,
        conv_channels: int = 128,
        kernel_size: int = 5,
        tcn_blocks: int = 2,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        attention_hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # Conv1d runs over time while treating the 156 CSI values as channels.
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length")
        layers: list[nn.Module] = [
            nn.Conv1d(input_dim, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        layers.extend(
            ResidualTCNBlock(conv_channels, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(tcn_blocks)
        )
        self.temporal_cnn = nn.Sequential(*layers)
        self.bilstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        encoded_dim = lstm_hidden * 2
        self.attention = TemporalAttention(encoded_dim, attention_hidden)
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoded_dim),
            nn.Dropout(dropout),
            nn.Linear(encoded_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.temporal_cnn(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.bilstm(x)
        features, attention = self.attention(sequence)
        logits = self.classifier(features)
        if return_features:
            return {"logits": logits, "features": features, "attention": attention}
        return logits
