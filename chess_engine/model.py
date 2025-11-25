"""PyTorch model definition for evaluating chess positions."""

from __future__ import annotations

import torch
from torch import nn


class SimpleEvaluator(nn.Module):
    """Small feed-forward network producing a scalar evaluation in [-1, 1]."""

    def __init__(self, input_dim: int = 773, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_model(path: str, device: torch.device | str = "cpu") -> SimpleEvaluator:
    model = SimpleEvaluator()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

