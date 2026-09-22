"""Binary mean-MLP sanity-check baseline."""
from torch import nn


class MeanMLP(nn.Module):
    def __init__(self, input_dim: int = 315):
        super().__init__(); self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128, 1))

    def forward(self, x): return self.net(x.mean(dim=1)).squeeze(-1)
