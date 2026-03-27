from __future__ import annotations

import torch
import torch.nn as nn


class InstanceClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, instance_feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(instance_feat)
