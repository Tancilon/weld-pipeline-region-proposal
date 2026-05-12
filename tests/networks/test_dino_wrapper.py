from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from networks.dino_wrapper import DINOv2WithQueries


def _non_lora_dino_params(wrapper):
    return [
        param
        for name, param in wrapper.dino.named_parameters()
        if ".lora_" not in name
    ]


def _lora_params(wrapper):
    return [
        param
        for name, param in wrapper.lora_adapters.named_parameters()
        if "lora_" in name
    ]


class AddBlock(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(float(delta)))

    def forward(self, tokens):
        return tokens + self.delta.view(1, 1, 1)


class CountingBlock(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(float(delta)))
        self.calls = 0

    def forward(self, tokens):
        self.calls += 1
        return tokens + self.delta.view(1, 1, 1)


class LinearBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, tokens):
        return self.proj(tokens)


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


class FakeDINOWithNorm(FakeDINO):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(self.embed_dim)


def test_forward_with_queries_returns_pose_tokens_after_pose_tail():
    wrapper = DINOv2WithQueries(FakeDINO(), num_query_tokens=2, query_inject_layer=2)

    pose_tokens, query_tokens, seg_tokens = wrapper.forward_with_queries(
        torch.zeros(1, 3, 28, 28)
    )

    assert pose_tokens.shape == (1, 4, 4)
    assert query_tokens.shape == (1, 2, 4)
    assert seg_tokens.shape == (1, 4, 4)
    assert torch.allclose(pose_tokens, torch.full_like(pose_tokens, 111.0))


def test_wrapper_uses_lora_tail_without_cloning_blocks():
    dino = FakeDINO()
    dino.blocks = nn.ModuleList([
        AddBlock(1.0),
        AddBlock(10.0),
        LinearBlock(),
    ])
    wrapper = DINOv2WithQueries(
        dino,
        num_query_tokens=2,
        query_inject_layer=2,
        lora_rank=1,
        lora_alpha=1.0,
    )
    with torch.no_grad():
        wrapper.query_embed.weight.zero_()
        adapter = wrapper.lora_adapters[0]
        adapter.lora_A.zero_()
        adapter.lora_B.zero_()
        adapter.lora_A[0, 0] = 1.0
        adapter.lora_B[:, 0] = 1.0

    pose_tokens, query_tokens, seg_tokens = wrapper.forward_with_queries(
        torch.zeros(1, 3, 28, 28)
    )

    assert not hasattr(wrapper, "seg_blocks")
    assert wrapper.dino.blocks[2] is dino.blocks[2]
    assert torch.allclose(pose_tokens, torch.full_like(pose_tokens, 11.0))
    assert torch.allclose(seg_tokens, torch.full_like(seg_tokens, 22.0))
    assert query_tokens.shape == (1, 2, 4)
    assert wrapper.query_embed.weight.requires_grad
    assert all(param.requires_grad for param in _lora_params(wrapper))
    assert not any(param.requires_grad for param in _non_lora_dino_params(wrapper))


def test_forward_segmentation_reuses_frozen_tail_with_lora_enabled():
    dino = FakeDINO()
    dino.blocks = nn.ModuleList([
        CountingBlock(1.0),
        CountingBlock(10.0),
        LinearBlock(),
    ])
    wrapper = DINOv2WithQueries(
        dino,
        num_query_tokens=2,
        query_inject_layer=2,
        lora_rank=1,
        lora_alpha=1.0,
    )

    with torch.no_grad():
        wrapper.query_embed.weight.zero_()
        adapter = wrapper.lora_adapters[0]
        adapter.lora_A.zero_()
        adapter.lora_B.zero_()
        adapter.lora_A[0, 0] = 1.0
        adapter.lora_B[:, 0] = 1.0

    query_tokens, seg_tokens = wrapper.forward_segmentation(torch.zeros(1, 3, 28, 28))

    assert query_tokens.shape == (1, 2, 4)
    assert seg_tokens.shape == (1, 4, 4)
    assert torch.allclose(seg_tokens, torch.full_like(seg_tokens, 22.0))
    assert dino.blocks[2].proj.lora_active is False


def test_wrapper_keeps_only_query_and_lora_trainable_when_input_dino_is_frozen():
    dino = FakeDINOWithNorm()
    dino.blocks = nn.ModuleList([
        AddBlock(1.0),
        AddBlock(10.0),
        LinearBlock(),
    ])
    dino.requires_grad_(False)

    wrapper = DINOv2WithQueries(
        dino,
        num_query_tokens=2,
        query_inject_layer=2,
        lora_rank=1,
    )

    assert all(not param.requires_grad for param in wrapper.dino.norm.parameters())
    assert wrapper.query_embed.weight.requires_grad
    assert all(param.requires_grad for param in _lora_params(wrapper))
