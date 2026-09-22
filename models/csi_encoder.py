"""Design-aligned CSI encoder: LayerNorm, TCN, BiGRU, attention."""
from __future__ import annotations

import torch
from torch import nn


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__(); padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation), nn.BatchNorm1d(channels))
        self.activation = nn.ReLU()

    def forward(self, x): return self.activation(x + self.net(x))


class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(); self.score = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False))

    def forward(self, x):
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1), weights


class CSIActionClassifier(nn.Module):
    def __init__(self, input_dim=315, conv_channels=128, kernel_size=5, tcn_blocks=2, gru_hidden=96,
                 gru_layers=1, attention_hidden=64, dropout=.3):
        super().__init__()
        if kernel_size % 2 == 0: raise ValueError("kernel_size must be odd")
        self.input_norm = nn.LayerNorm(input_dim)
        layers = [nn.Conv1d(input_dim, conv_channels, kernel_size, padding=kernel_size // 2), nn.BatchNorm1d(conv_channels), nn.ReLU(), nn.Dropout(dropout)]
        layers += [ResidualTCNBlock(conv_channels, kernel_size, 2 ** i, dropout) for i in range(tcn_blocks)]
        self.temporal_cnn = nn.Sequential(*layers)
        self.bigru = nn.GRU(conv_channels, gru_hidden, gru_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if gru_layers > 1 else 0.0)
        self.attention = TemporalAttention(gru_hidden * 2, attention_hidden)
        self.classifier = nn.Sequential(nn.LayerNorm(gru_hidden * 2), nn.Dropout(dropout), nn.Linear(gru_hidden * 2, 1))

    def forward(self, x, return_features=False):
        x = self.input_norm(x); x = self.temporal_cnn(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.bigru(x); features, attention = self.attention(sequence); logits = self.classifier(features).squeeze(-1)
        return {"logits": logits, "features": features, "attention": attention} if return_features else logits
