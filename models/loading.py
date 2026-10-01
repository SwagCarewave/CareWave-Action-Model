"""Build a classifier from a single-model or seed-ensemble checkpoint."""
from __future__ import annotations

import torch
from torch import nn

from .csi_encoder import CSIActionClassifier


class ProbabilityAverageEnsemble(nn.Module):
    """Averages member fall probabilities; returns the logit of that mean so callers can
    keep using torch.sigmoid(model(x))."""

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        probability = torch.stack([torch.sigmoid(member(x)) for member in self.members]).mean(dim=0)
        return torch.logit(probability.clamp(1e-6, 1 - 1e-6))


def build_model(checkpoint: dict) -> nn.Module:
    if checkpoint.get("kind") == "motion_gbm":
        from .motion_gbm import MotionGBMModule
        return MotionGBMModule(checkpoint).eval()
    if "ensemble_members" in checkpoint:
        members = []
        for member in checkpoint["ensemble_members"]:
            model = CSIActionClassifier(**member["model_config"])
            model.load_state_dict(member["model_state"])
            members.append(model)
        model = ProbabilityAverageEnsemble(members)
    else:
        model = CSIActionClassifier(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
    return model.eval()
