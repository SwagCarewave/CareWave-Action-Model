"""Small baseline used to verify that the action dataset is learnable."""

from torch import nn


class MeanMLP(nn.Module):
    def __init__(self, input_dim: int = 156, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.net(x.mean(dim=1))
