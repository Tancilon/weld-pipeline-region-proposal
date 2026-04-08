# Single-Agent Segmentation And Pose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new single-agent segmentation-and-pose runtime that uses one new `seg_ckpt` for both mask prediction and 6D pose inference, while keeping `energy_ckpt` and `scale_ckpt` separate.

**Architecture:** Split the segmentation-enabled DINO wrapper into a frozen shared stem, a frozen original pose tail, and a trainable cloned segmentation tail. Keep segmentation training initialized from an existing pose checkpoint, but remove `pose_ckpt` from runtime inference by routing both segmentation and pose through one `enable_segmentation=True` main agent.

**Tech Stack:** Python, PyTorch, pytest, existing GenPose2 `PoseNet`/`GFObjectPose` stack, DINOv2 wrapper, EoMT segmentation head.

---

## File Structure

- Modify: `networks/dino_wrapper.py`
  Responsibility: implement the dual-tail DINO wrapper with a frozen original pose tail and a cloned trainable segmentation tail.

- Modify: `networks/posenet.py`
  Responsibility: consume the wrapper’s new output contract so pose uses final frozen pose-tail patch tokens while segmentation keeps using query and segmentation-tail patch tokens.

- Modify: `runners/trainer.py`
  Responsibility: require a pose checkpoint when training segmentation, and freeze all parameters except the segmentation tail, query embedding, and EoMT head.

- Modify: `configs/config.py`
  Responsibility: clarify CLI semantics for segmentation training and mark `unfreeze_dino_last_n` as legacy in the new dual-tail design.

- Modify: `runners/visualize_seg.py`
  Responsibility: align the inline segmentation config with the new “original DINO stays frozen” semantics.

- Create: `runners/infer_nuclear_full_lib.py`
  Responsibility: hold testable parser/config/checkpoint/runtime helpers for the nuclear single-agent inference path without importing `PoseNet` eagerly, and own the moved `build_instance_batch`/`aggregate_pose`/`estimate_size_from_geometry` helpers used by `infer_pose_and_size`.

- Modify: `runners/infer_nuclear_full.py`
  Responsibility: keep the CLI entrypoint, but route runtime setup through the new helper module and remove the separate pose-agent path.

- Create: `tests/networks/test_dino_wrapper.py`
  Responsibility: prove the wrapper returns pose tokens after the frozen pose tail and owns a cloned segmentation tail.

- Create: `tests/runners/test_trainer_segmentation_setup.py`
  Responsibility: prove the segmentation training setup only leaves segmentation modules trainable and rejects missing pose initialization checkpoints.

- Create: `tests/runners/test_infer_nuclear_full_lib.py`
  Responsibility: prove the runtime parser has no `pose_ckpt` argument, old checkpoints are rejected clearly, and the runtime only constructs one main agent plus optional EnergyNet/ScaleNet agents.

## Task 1: Dual-Tail DINO Wrapper And Model Integration

**Files:**
- Create: `tests/networks/test_dino_wrapper.py`
- Modify: `networks/dino_wrapper.py:1-125`
- Modify: `networks/posenet.py:44-74`
- Modify: `networks/posenet.py:123-279`

- [ ] **Step 1: Write the failing wrapper tests**

```python
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
```

- [ ] **Step 2: Run the wrapper tests to verify they fail**

Run: `pytest tests/networks/test_dino_wrapper.py -v`

Expected: FAIL because the current wrapper returns pose tokens from before the final DINO tail and does not expose a cloned `seg_blocks` tail.

- [ ] **Step 3: Implement the dual-tail wrapper and keep `GFObjectPose` on the same public method**

```python
# networks/dino_wrapper.py
import copy
import torch
import torch.nn as nn


class DINOv2WithQueries(nn.Module):
    def __init__(self, dino_model, num_query_tokens=50, query_inject_layer=-4):
        super().__init__()
        self.dino = dino_model
        self.num_query_tokens = num_query_tokens
        self.embed_dim = dino_model.embed_dim

        num_blocks = len(dino_model.blocks)
        if query_inject_layer < 0:
            query_inject_layer = num_blocks + query_inject_layer
        assert 0 < query_inject_layer < num_blocks, (
            f"query_inject_layer={query_inject_layer} out of range [1, {num_blocks - 1}]"
        )
        self.inject_layer = query_inject_layer

        self.query_embed = nn.Embedding(num_query_tokens, self.embed_dim)
        nn.init.trunc_normal_(self.query_embed.weight, std=0.02)

        self.seg_blocks = copy.deepcopy(
            nn.ModuleList(dino_model.blocks[self.inject_layer:])
        )
        self.seg_norm = (
            copy.deepcopy(dino_model.norm)
            if hasattr(dino_model, "norm") and dino_model.norm is not None
            else None
        )

    def _apply_blocks(self, tokens, blocks):
        for blk in blocks:
            tokens = blk(tokens)
        return tokens

    def forward_with_queries(self, x):
        bs = x.shape[0]
        tokens = self._prepare_tokens(x)
        num_prefix = self._count_prefix_tokens()

        stem_tokens = self._apply_blocks(tokens, self.dino.blocks[:self.inject_layer])

        pose_tokens = self._apply_blocks(
            stem_tokens, self.dino.blocks[self.inject_layer:]
        )
        if hasattr(self.dino, "norm") and self.dino.norm is not None:
            pose_tokens = self.dino.norm(pose_tokens)
        patch_tokens_for_pose = pose_tokens[:, num_prefix:, :].detach().clone()

        query_tokens = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)
        seg_tokens = torch.cat([stem_tokens, query_tokens], dim=1)
        seg_tokens = self._apply_blocks(seg_tokens, self.seg_blocks)
        if self.seg_norm is not None:
            seg_tokens = self.seg_norm(seg_tokens)

        num_queries = self.num_query_tokens
        query_out = seg_tokens[:, -num_queries:, :]
        patch_out = seg_tokens[:, num_prefix:-num_queries, :]
        return patch_tokens_for_pose, query_out, patch_out
```

```python
# networks/posenet.py
if self.enable_segmentation:
    self.dino_wrapper = DINOv2WithQueries(
        raw_dino,
        num_query_tokens=cfg.num_queries,
        query_inject_layer=cfg.query_inject_layer,
    )
    self.eomt_head = EoMTHead(
        embed_dim=self.dino_dim,
        num_classes=cfg.num_object_classes,
        num_queries=cfg.num_queries,
    )
    raw_dino.requires_grad_(False)
    self.dino = raw_dino
```

```python
# networks/posenet.py
if self.enable_segmentation:
    pose_patch_tokens, query_out, patch_after_query = (
        self.dino_wrapper.forward_with_queries(roi_rgb)
    )
    feat = pose_patch_tokens
    data["_query_tokens"] = query_out
    data["_patch_tokens_seg"] = patch_after_query
else:
    feat = self.dino.get_intermediate_layers(roi_rgb)[0]
```

- [ ] **Step 4: Run the wrapper tests again**

Run: `pytest tests/networks/test_dino_wrapper.py -v`

Expected: PASS

- [ ] **Step 5: Commit the wrapper and integration changes**

```bash
git add tests/networks/test_dino_wrapper.py networks/dino_wrapper.py networks/posenet.py
git commit -m "feat: split dino wrapper into pose and segmentation tails"
```

### Task 2: Segmentation Training Init And Freeze Policy

**Files:**
- Create: `tests/runners/test_trainer_segmentation_setup.py`
- Modify: `runners/trainer.py:220-354`
- Modify: `configs/config.py:40-67`
- Modify: `runners/visualize_seg.py:267-305`

- [ ] **Step 1: Write the failing training-setup tests**

```python
import importlib
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

@pytest.fixture
def trainer_module(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["trainer.py"])
    return importlib.import_module("runners.trainer")


class DummyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_embed = nn.Embedding(2, 4)
        self.seg_blocks = nn.ModuleList([nn.Linear(4, 4)])
        self.seg_norm = nn.LayerNorm(4)
        self.dino = nn.Module()
        self.dino.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.eomt_head = nn.Linear(4, 2)
        self.dino_wrapper = DummyWrapper()
        self.pose_score_net = nn.Linear(4, 4)


class DummyAgent:
    def __init__(self):
        self.net = DummyNet()
        self.cfg = SimpleNamespace(unfreeze_dino_last_n=4)


def test_freeze_pose_params_only_keeps_segmentation_modules_trainable(trainer_module):
    agent = DummyAgent()

    trainer_module.freeze_pose_params(agent)

    trainable = {name for name, param in agent.net.named_parameters() if param.requires_grad}
    assert trainable == {
        "eomt_head.weight",
        "eomt_head.bias",
        "dino_wrapper.query_embed.weight",
        "dino_wrapper.seg_blocks.0.weight",
        "dino_wrapper.seg_blocks.0.bias",
        "dino_wrapper.seg_norm.weight",
        "dino_wrapper.seg_norm.bias",
    }


def test_validate_segmentation_pose_init_requires_checkpoint(trainer_module):
    cfg = SimpleNamespace(pretrained_score_model_path=None)

    with pytest.raises(
        ValueError,
        match="Segmentation training requires pretrained_score_model_path",
    ):
        trainer_module.validate_segmentation_pose_init(cfg)
```

- [ ] **Step 2: Run the training-setup tests to verify they fail**

Run: `pytest tests/runners/test_trainer_segmentation_setup.py -v`

Expected: FAIL because `validate_segmentation_pose_init()` does not exist and `freeze_pose_params()` does not currently unfreeze `seg_blocks`/`seg_norm`.

- [ ] **Step 3: Add the segmentation init guard, tighten the freeze policy, and clarify the config**

```python
# runners/trainer.py
def validate_segmentation_pose_init(cfg):
    if not cfg.pretrained_score_model_path:
        raise ValueError(
            "Segmentation training requires pretrained_score_model_path pointing to a pose checkpoint"
        )


def freeze_pose_params(agent):
    net = agent.net.module if isinstance(agent.net, torch.nn.DataParallel) else agent.net
    for param in net.parameters():
        param.requires_grad = False

    if hasattr(net, "eomt_head"):
        for param in net.eomt_head.parameters():
            param.requires_grad = True

    if hasattr(net, "dino_wrapper"):
        for param in net.dino_wrapper.query_embed.parameters():
            param.requires_grad = True
        for param in net.dino_wrapper.seg_blocks.parameters():
            param.requires_grad = True
        if getattr(net.dino_wrapper, "seg_norm", None) is not None:
            for param in net.dino_wrapper.seg_norm.parameters():
                param.requires_grad = True
```

```python
# runners/trainer.py
elif cfg.agent_type == 'segmentation':
    cfg.enable_segmentation = True
    validate_segmentation_pose_init(cfg)
    seg_agent = PoseNet(cfg)
    seg_agent.load_ckpt(
        model_dir=cfg.pretrained_score_model_path,
        model_path=True,
        load_model_only=True,
    )
    freeze_pose_params(seg_agent)
    tr_agent = seg_agent
```

```python
# configs/config.py
parser.add_argument(
    '--pretrained_score_model_path',
    type=str,
    help='score checkpoint path; required when agent_type=segmentation to seed the frozen pose path',
)
parser.add_argument(
    '--unfreeze_dino_last_n',
    type=int,
    default=0,
    help='legacy option; ignored in dual-tail segmentation because the original DINO backbone stays frozen',
)
```

```python
# runners/visualize_seg.py
cfg.unfreeze_dino_last_n = 0
```

- [ ] **Step 4: Run the training-setup tests again**

Run: `pytest tests/runners/test_trainer_segmentation_setup.py -v`

Expected: PASS

- [ ] **Step 5: Commit the training-setup changes**

```bash
git add tests/runners/test_trainer_segmentation_setup.py runners/trainer.py configs/config.py runners/visualize_seg.py
git commit -m "feat: require pose init and freeze only seg tail for segmentation training"
```

### Task 3: Single-Agent Nuclear Runtime Library And CLI Wiring

**Files:**
- Create: `tests/runners/test_infer_nuclear_full_lib.py`
- Create: `runners/infer_nuclear_full_lib.py`
- Modify: `runners/infer_nuclear_full.py:1-58`
- Modify: `runners/infer_nuclear_full.py:148-188`
- Modify: `runners/infer_nuclear_full.py:376-413`
- Modify: `runners/infer_nuclear_full.py:499-727`

- [ ] **Step 1: Write the failing runtime tests**

```python
from types import SimpleNamespace

import pytest
import torch

import runners.infer_nuclear_full_lib as lib


def test_build_arg_parser_has_no_pose_checkpoint_argument():
    parser = lib.build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--nuclear_data_path", "data",
            "--seg_ckpt_path", "seg.pth",
            "--pose_ckpt_path", "pose.pth",
        ])


def test_validate_single_agent_seg_state_dict_rejects_old_checkpoint():
    old_state = {
        "dino_wrapper.query_embed.weight": torch.zeros(2, 4),
        "eomt_head.class_head.weight": torch.zeros(7, 4),
        "pose_score_net.fc.weight": torch.zeros(4, 4),
    }

    with pytest.raises(ValueError, match="single-agent segmentation checkpoint"):
        lib.validate_single_agent_seg_state_dict(old_state)


def test_init_pipeline_agents_builds_one_main_agent(monkeypatch):
    created = []

    class DummyNet:
        def eval(self):
            return self

    class FakePoseNet:
        def __init__(self, cfg):
            created.append((cfg.agent_type, cfg.enable_segmentation))
            self.cfg = cfg
            self.net = DummyNet()

    monkeypatch.setattr(lib, "load_posenet_cls", lambda: FakePoseNet)
    monkeypatch.setattr(lib, "load_main_agent_checkpoint", lambda agent, ckpt_path: None)
    monkeypatch.setattr(lib, "load_model_only", lambda agent, ckpt_path, name: None)

    args = SimpleNamespace(
        nuclear_data_path="data",
        seg_ckpt_path="seg.pth",
        energy_ckpt_path="energy.pth",
        scale_ckpt_path="scale.pth",
        split="val",
        output_dir="out",
        num_vis=1,
        score_threshold=0.5,
        repeat_num=8,
        num_points=1024,
        img_size=224,
        device="cpu",
    )

    main_cfg, main_agent, energy_agent, scale_agent = lib.init_pipeline_agents(args)

    assert created == [("score", True), ("energy", False), ("scale", False)]
    assert main_agent is not None
    assert energy_agent is not None
    assert scale_agent is not None
```

- [ ] **Step 2: Run the runtime tests to verify they fail**

Run: `pytest tests/runners/test_infer_nuclear_full_lib.py -v`

Expected: FAIL with `ModuleNotFoundError` because `runners.infer_nuclear_full_lib` does not exist yet.

- [ ] **Step 3: Add the runtime helper module and rewire the CLI to one main agent**

Move the existing `build_instance_batch()`, `aggregate_pose()`, and `estimate_size_from_geometry()` helpers from `runners/infer_nuclear_full.py` into `runners/infer_nuclear_full_lib.py` unchanged first, then add the new parser/checkpoint/runtime helpers below so `infer_pose_and_size()` has no hidden dependencies.

```python
# runners/infer_nuclear_full_lib.py
import argparse
import torch


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Nuclear workpiece full pipeline inference")
    parser.add_argument("--nuclear_data_path", type=str, required=True)
    parser.add_argument("--seg_ckpt_path", type=str, required=True)
    parser.add_argument("--energy_ckpt_path", type=str, default=None)
    parser.add_argument("--scale_ckpt_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output_dir", type=str, default="./results/full_pipeline")
    parser.add_argument("--num_vis", type=int, default=5)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--repeat_num", type=int, default=10)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def validate_single_agent_seg_state_dict(model_state_dict):
    checks = {
        "dino_wrapper.query_embed.weight": "missing query embedding weights",
        "eomt_head.class_head.weight": "missing segmentation head weights",
    }
    for key, reason in checks.items():
        if key not in model_state_dict:
            raise ValueError(
                f"Expected a single-agent segmentation checkpoint: {reason} ({key})"
            )

    if not any(key.startswith("dino_wrapper.seg_blocks.") for key in model_state_dict):
        raise ValueError(
            "Expected a single-agent segmentation checkpoint with dino_wrapper.seg_blocks.* keys"
        )
    if not any(key.startswith("dino_wrapper.dino.blocks.") for key in model_state_dict):
        raise ValueError(
            "Expected a single-agent segmentation checkpoint with frozen pose-path DINO blocks"
        )
    if not any(key.startswith("pose_score_net.") for key in model_state_dict):
        raise ValueError(
            "Expected a single-agent segmentation checkpoint carrying pose network weights"
        )


def load_posenet_cls():
    from networks.posenet_agent import PoseNet
    return PoseNet


def load_model_only(agent, ckpt_path, name):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = agent.net.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if missing or unexpected:
        raise ValueError(
            f"{name} checkpoint does not match the current model. "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )


def load_main_agent_checkpoint(agent, seg_ckpt_path):
    checkpoint = torch.load(seg_ckpt_path, map_location="cpu")
    model_state_dict = checkpoint["model_state_dict"]
    validate_single_agent_seg_state_dict(model_state_dict)
    missing, unexpected = agent.net.load_state_dict(model_state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            "Single-agent segmentation checkpoint does not match the current model. "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
```

```python
# runners/infer_nuclear_full_lib.py
def build_cfg(args, agent_type="score", enable_segmentation=False):
    class Cfg:
        pass

    cfg = Cfg()
    cfg.device = args.device if torch.cuda.is_available() else "cpu"
    cfg.dino = "pointwise"
    cfg.pts_encoder = "pointnet2"
    cfg.agent_type = agent_type
    cfg.pose_mode = "rot_matrix"
    cfg.regression_head = "Rx_Ry_and_T"
    cfg.sde_mode = "ve"
    cfg.num_points = args.num_points
    cfg.img_size = args.img_size
    cfg.pointnet2_params = "light"
    cfg.parallel = False
    cfg.is_train = False
    cfg.eval = False
    cfg.pred = False
    cfg.use_pretrain = False
    cfg.log_dir = "infer_nuclear"
    cfg.ema_rate = 0.999
    cfg.lr = 1e-4
    cfg.lr_decay = 0.99
    cfg.optimizer = "Adam"
    cfg.warmup = 50
    cfg.grad_clip = 1.0
    cfg.sampling_steps = 500
    cfg.sampler_mode = ["ode"]
    cfg.energy_mode = "IP"
    cfg.s_theta_mode = "score"
    cfg.norm_energy = "identical"
    cfg.scale_embedding = 180
    cfg.eval_repeat_num = 50
    cfg.repeat_num = args.repeat_num
    cfg.num_gpu = 1
    cfg.scale_batch_size = 64
    cfg.save_video = False
    cfg.retain_ratio = 0.4
    cfg.clustering = 1
    cfg.clustering_eps = 0.05
    cfg.clustering_minpts = 0.1667
    cfg.enable_segmentation = enable_segmentation
    cfg.num_queries = 50
    cfg.query_inject_layer = -4
    cfg.num_object_classes = 6
    cfg.unfreeze_dino_last_n = 0
    cfg.seg_loss_weight = 1.0
    cfg.cls_loss_weight = 2.0
    return cfg


def init_pipeline_agents(args):
    PoseNet = load_posenet_cls()

    main_cfg = build_cfg(args, agent_type="score", enable_segmentation=True)
    main_agent = PoseNet(main_cfg)
    load_main_agent_checkpoint(main_agent, args.seg_ckpt_path)
    main_agent.net.eval()

    energy_agent = None
    if args.energy_ckpt_path:
        energy_cfg = build_cfg(args, agent_type="energy", enable_segmentation=False)
        energy_agent = PoseNet(energy_cfg)
        load_model_only(energy_agent, args.energy_ckpt_path, "energy")
        energy_agent.net.eval()

    scale_agent = None
    if args.scale_ckpt_path:
        scale_cfg = build_cfg(args, agent_type="scale", enable_segmentation=False)
        scale_agent = PoseNet(scale_cfg)
        load_model_only(scale_agent, args.scale_ckpt_path, "scale")
        scale_agent.net.eval()

    return main_cfg, main_agent, energy_agent, scale_agent


def infer_pose_and_size(main_agent, energy_agent, scale_agent, cfg, pt_data, device, repeat_num):
    data = build_instance_batch(pt_data, device)

    pred_pose, _ = main_agent.pred_func(data, repeat_num=repeat_num)
    pred_energy = None
    if energy_agent is not None:
        pred_energy = energy_agent.get_energy(
            data=data,
            pose_samples=pred_pose,
            T=None,
            mode="test",
            extract_feature=True,
        )

    aggregated_pose = aggregate_pose(cfg, pred_pose, pred_energy)
    final_pose = aggregated_pose.clone()

    if scale_agent is not None:
        scale_input = {
            "pts_feat": data["pts_feat"],
            "rgb_feat": data["rgb_feat"],
            "axes": aggregated_pose[:, :3, :3],
        }
        cal_mat, pred_size = scale_agent.pred_scale_func(scale_input)
        final_pose[:, :3, :3] = cal_mat
        size_source = "scale_net"
    else:
        pred_size = estimate_size_from_geometry(data["pcl_in"], aggregated_pose)
        size_source = "geometry"

    return {
        "R": final_pose[0, :3, :3].detach().cpu().numpy(),
        "t": final_pose[0, :3, 3].detach().cpu().numpy(),
        "size": pred_size[0].detach().cpu().numpy(),
        "size_source": size_source,
    }
```

```python
# runners/infer_nuclear_full.py
from runners.infer_nuclear_full_lib import (
    build_arg_parser,
    init_pipeline_agents,
    infer_pose_and_size,
)

_p = build_arg_parser()
_args = _p.parse_args()
sys.argv = sys.argv[:1]
```

```python
# runners/infer_nuclear_full.py
main_cfg, main_agent, energy_agent, scale_agent = init_pipeline_agents(args)

with torch.no_grad():
    class_logits, mask_logits = main_agent.net(batch, mode="segmentation")

pose_res = infer_pose_and_size(
    main_agent=main_agent,
    energy_agent=energy_agent,
    scale_agent=scale_agent,
    cfg=main_cfg,
    pt_data=pt_data,
    device=device,
    repeat_num=args.repeat_num,
)
```

- [ ] **Step 4: Run the runtime tests again**

Run: `pytest tests/runners/test_infer_nuclear_full_lib.py -v`

Expected: PASS

- [ ] **Step 5: Commit the runtime changes**

```bash
git add tests/runners/test_infer_nuclear_full_lib.py runners/infer_nuclear_full_lib.py runners/infer_nuclear_full.py
git commit -m "feat: run nuclear segmentation and pose through one main agent"
```

## Final Verification

- [ ] Run the focused unit tests together.

Run: `pytest tests/networks/test_dino_wrapper.py tests/runners/test_trainer_segmentation_setup.py tests/runners/test_infer_nuclear_full_lib.py -v`

Expected: PASS

- [ ] Check the CLI surface for the new runtime.

Run: `python runners/infer_nuclear_full.py --help`

Expected: usage includes `--seg_ckpt_path`, `--energy_ckpt_path`, `--scale_ckpt_path` and does not include `--pose_ckpt_path`.

- [ ] Run a one-image nuclear inference smoke test with the new runtime surface.

Run: `python runners/infer_nuclear_full.py --nuclear_data_path ./data/aiws5.2_nuclear_workpieces --seg_ckpt_path ./results/ckpts/SegNetSingleAgent/ckpt_epoch1.pth --energy_ckpt_path ./results/ckpts/EnergyNet/energynet.pth --scale_ckpt_path ./results/ckpts/ScaleNet/scalenet.pth --split val --num_vis 1 --output_dir ./results/full_pipeline_single_agent_smoke`

Expected: one visualization image is written under `./results/full_pipeline_single_agent_smoke` and the console ends with `Done. Results saved to: ./results/full_pipeline_single_agent_smoke`.

- [ ] Run a short segmentation-training smoke test to confirm the new init path and trainable-parameter policy.

Run: `python runners/trainer.py --agent_type segmentation --is_train --dataset_type nuclear --nuclear_data_path ./data/aiws5.2_nuclear_workpieces --pretrained_score_model_path ./results/ckpts/ScoreNet/scorenet.pth --n_epochs 1 --eval_freq 1 --batch_size 2 --num_workers 0 --log_dir SegNetSingleAgentSmoke`

Expected: startup prints the `freeze_pose_params` trainable-count summary, training runs for one epoch without loading a separate segmentation-only backbone, and writes a new checkpoint under `./results/ckpts/SegNetSingleAgentSmoke`.
