import torch
import torch.nn as nn

from networks.dino_wrapper import DINOv2WithQueries


class AddBlock(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(float(delta)))

    def forward(self, tokens):
        return tokens + self.delta.view(1, 1, 1)


class FakeDINO(nn.Module):
    embed_dim = 4

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            AddBlock(1.0),
            AddBlock(10.0),
            AddBlock(100.0),
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.register_tokens = None
        self.norm = nn.Identity()

    def prepare_tokens_with_masks(self, x):
        bs = x.shape[0]
        cls = self.cls_token.expand(bs, -1, -1)
        patches = torch.zeros(bs, 4, self.embed_dim)
        return torch.cat([cls, patches], dim=1)


def test_forward_with_queries_returns_pose_tokens_after_pose_tail():
    wrapper = DINOv2WithQueries(FakeDINO(), num_query_tokens=2, query_inject_layer=2)

    pose_tokens, query_tokens, seg_tokens = wrapper.forward_with_queries(
        torch.zeros(1, 3, 28, 28)
    )

    assert pose_tokens.shape == (1, 4, 4)
    assert query_tokens.shape == (1, 2, 4)
    assert seg_tokens.shape == (1, 4, 4)
    assert torch.allclose(pose_tokens, torch.full_like(pose_tokens, 111.0))


def test_wrapper_clones_a_dedicated_segmentation_tail():
    wrapper = DINOv2WithQueries(FakeDINO(), num_query_tokens=2, query_inject_layer=2)

    assert hasattr(wrapper, "seg_blocks")
    assert wrapper.seg_blocks[0] is not wrapper.dino.blocks[2]
    assert torch.allclose(
        wrapper.seg_blocks[0].delta.detach(),
        wrapper.dino.blocks[2].delta.detach(),
    )

