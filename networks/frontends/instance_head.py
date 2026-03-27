from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class InstanceHeadOutput:
    mask_logits: torch.Tensor
    score_logits: torch.Tensor
    class_logits: torch.Tensor | None = None


class InstanceSegmentationHead(nn.Module):
    def __init__(
        self, in_dim: int, num_classes: int = 0, hidden_dim: int = 256
    ) -> None:
        super().__init__()
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.score_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, 1),
        )
        self.class_head = None
        if num_classes > 0:
            self.class_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_dim, num_classes),
            )

    def forward(self, feature_map: torch.Tensor) -> InstanceHeadOutput:
        class_logits = None if self.class_head is None else self.class_head(feature_map)
        return InstanceHeadOutput(
            mask_logits=self.mask_head(feature_map),
            score_logits=self.score_head(feature_map),
            class_logits=class_logits,
        )
