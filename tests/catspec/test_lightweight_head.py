import torch

from catspec.lightweight_head import SpecConditionedLightweightHead, mini_head_loss
from catspec.mini_dataset import PATH_COUNT_CLASSES, SEGMENT_SEQUENCE_CLASSES, TOPOLOGY_CLASSES


def test_lightweight_head_forward_shapes():
    head = SpecConditionedLightweightHead(embedding_dim=32, hidden_dim=24)
    embeddings = torch.randn(3, 32)

    outputs = head(embeddings)

    assert outputs["topology_logits"].shape == (3, len(TOPOLOGY_CLASSES))
    assert outputs["path_count_logits"].shape == (3, len(PATH_COUNT_CLASSES))
    assert outputs["segment_sequence_logits"].shape == (3, len(SEGMENT_SEQUENCE_CLASSES))


def test_mini_head_loss_is_finite_and_backpropagates():
    head = SpecConditionedLightweightHead(embedding_dim=32, hidden_dim=24)
    embeddings = torch.randn(4, 32)
    targets = {
        "topology_label": torch.tensor([0, 1, 1, 2], dtype=torch.long),
        "path_count_label": torch.tensor([0, 1, 1, 1], dtype=torch.long),
        "segment_sequence_label": torch.tensor([0, 1, 1, 2], dtype=torch.long),
    }

    outputs = head(embeddings)
    loss, parts = mini_head_loss(outputs, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert set(parts) == {"topology_loss", "path_count_loss", "segment_sequence_loss"}
    assert any(param.grad is not None for param in head.parameters())
