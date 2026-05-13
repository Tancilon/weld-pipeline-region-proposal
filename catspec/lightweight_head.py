"""Spec-conditioned lightweight prediction head for the mini PoC."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from catspec.mini_dataset import PATH_COUNT_CLASSES, SEGMENT_SEQUENCE_CLASSES, TOPOLOGY_CLASSES


class SpecConditionedLightweightHead(nn.Module):
    """Small MLP with topology, path-count, and segment-sequence heads."""

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.topology_head = nn.Linear(hidden_dim, len(TOPOLOGY_CLASSES))
        self.path_count_head = nn.Linear(hidden_dim, len(PATH_COUNT_CLASSES))
        self.segment_sequence_head = nn.Linear(hidden_dim, len(SEGMENT_SEQUENCE_CLASSES))

    def forward(self, spec_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(spec_embedding)
        return {
            "topology_logits": self.topology_head(hidden),
            "path_count_logits": self.path_count_head(hidden),
            "segment_sequence_logits": self.segment_sequence_head(hidden),
        }


def mini_head_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return summed mini-head loss and scalar loss parts."""

    topology_loss = F.cross_entropy(outputs["topology_logits"], targets["topology_label"])
    path_count_loss = F.cross_entropy(outputs["path_count_logits"], targets["path_count_label"])
    segment_sequence_loss = F.cross_entropy(
        outputs["segment_sequence_logits"],
        targets["segment_sequence_label"],
    )
    total = topology_loss + path_count_loss + segment_sequence_loss
    return total, {
        "topology_loss": float(topology_loss.detach().cpu()),
        "path_count_loss": float(path_count_loss.detach().cpu()),
        "segment_sequence_loss": float(segment_sequence_loss.detach().cpu()),
    }


def logits_to_predictions(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert logits to integer prediction labels."""

    return {
        "topology_label": outputs["topology_logits"].argmax(dim=-1),
        "path_count_label": outputs["path_count_logits"].argmax(dim=-1),
        "segment_sequence_label": outputs["segment_sequence_logits"].argmax(dim=-1),
    }


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    if len(target) == 0:
        return 0.0
    return float((pred == target).to(dtype=torch.float32).mean().detach().cpu())


def batch_accuracy(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, float]:
    preds = logits_to_predictions(outputs)
    return {
        "topology_accuracy": accuracy(preds["topology_label"], targets["topology_label"]),
        "path_count_accuracy": accuracy(preds["path_count_label"], targets["path_count_label"]),
        "segment_sequence_accuracy": accuracy(preds["segment_sequence_label"], targets["segment_sequence_label"]),
    }


def state_config(embedding_dim: int, hidden_dim: int) -> dict[str, Any]:
    return {"embedding_dim": embedding_dim, "hidden_dim": hidden_dim}
