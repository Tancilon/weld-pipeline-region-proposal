import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
        )
        self.eomt_head = nn.Linear(4, 2)
        self.dino_wrapper = nn.Module()
        self.dino_wrapper.dino = nn.Sequential(
            nn.Linear(4, 4),
            nn.Linear(4, 4),
        )
        self.dino_wrapper.query_embed = nn.Embedding(3, 4)
        self.dino_wrapper.seg_blocks = nn.ModuleList(
            [nn.Linear(4, 4), nn.Linear(4, 4)]
        )
        self.dino_wrapper.seg_norm = nn.LayerNorm(4)


class FakeAgent:
    def __init__(self):
        self.net = FakeNet()
        self.cfg = SimpleNamespace(unfreeze_dino_last_n=99)


def _install_module(monkeypatch, name, package=False, **attrs):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def trainer_module(monkeypatch):
    original_argv0 = sys.argv[0]
    monkeypatch.setattr(sys, "argv", [original_argv0])
    monkeypatch.delitem(sys.modules, "runners.trainer", raising=False)

    cv2 = _install_module(monkeypatch, "cv2")
    cv2.INTER_LINEAR = 1
    cv2.RETR_EXTERNAL = 0
    cv2.CHAIN_APPROX_SIMPLE = 0
    cv2.COLOR_BGR2RGB = 0
    cv2.COLOR_RGB2BGR = 0
    cv2.imread = lambda *args, **kwargs: None
    cv2.imwrite = lambda *args, **kwargs: True
    cv2.cvtColor = lambda img, code: img
    cv2.resize = lambda img, size, interpolation=None: img
    cv2.addWeighted = lambda src1, alpha, src2, beta, gamma: src1
    cv2.findContours = lambda *args, **kwargs: ([], None)
    cv2.drawContours = lambda *args, **kwargs: None
    cv2.moments = lambda *args, **kwargs: {"m00": 0, "m10": 0, "m01": 0}

    ipdb = _install_module(monkeypatch, "ipdb")
    ipdb.set_trace = lambda *args, **kwargs: None

    tensorboardx = _install_module(monkeypatch, "tensorboardX")

    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

    tensorboardx.SummaryWriter = SummaryWriter

    matplotlib = _install_module(monkeypatch, "matplotlib")
    pyplot = _install_module(monkeypatch, "matplotlib.pyplot")
    matplotlib.rc = lambda *args, **kwargs: None
    pyplot.figure = lambda *args, **kwargs: None
    pyplot.imshow = lambda *args, **kwargs: None
    pyplot.show = lambda *args, **kwargs: None
    pyplot.savefig = lambda *args, **kwargs: None
    pyplot.close = lambda *args, **kwargs: None
    pyplot.subplots = lambda *args, **kwargs: (None, None)
    pyplot.axis = lambda *args, **kwargs: None
    matplotlib.pyplot = pyplot

    scipy = _install_module(monkeypatch, "scipy")
    spatial = _install_module(monkeypatch, "scipy.spatial")
    transform = _install_module(monkeypatch, "scipy.spatial.transform")

    class Rotation:
        @classmethod
        def from_matrix(cls, *args, **kwargs):
            return cls()

        def as_matrix(self):
            return None

    transform.Rotation = Rotation
    spatial.transform = transform
    scipy.spatial = spatial

    _install_module(monkeypatch, "datasets", package=True)
    _install_module(monkeypatch, "networks", package=True)
    _install_module(monkeypatch, "utils", package=True)
    _install_module(monkeypatch, "cutoop", package=True)

    _install_module(
        monkeypatch,
        "datasets.datasets_omni6dpose",
        get_data_loaders_from_cfg=lambda *args, **kwargs: None,
        process_batch=lambda *args, **kwargs: None,
        array_to_SymLabel=lambda *args, **kwargs: None,
    )
    _install_module(
        monkeypatch,
        "datasets.datasets_nuclear",
        get_nuclear_data_loaders=lambda *args, **kwargs: None,
        process_batch_seg=lambda *args, **kwargs: None,
    )
    _install_module(monkeypatch, "networks.posenet_agent", PoseNet=type("PoseNet", (), {}))
    _install_module(
        monkeypatch,
        "utils.misc",
        exists_or_mkdir=lambda *args, **kwargs: None,
        get_pose_representation=lambda *args, **kwargs: None,
        average_quaternion_batch=lambda *args, **kwargs: None,
        parallel_setup=lambda *args, **kwargs: None,
        parallel_cleanup=lambda *args, **kwargs: None,
    )
    _install_module(monkeypatch, "utils.genpose_utils", merge_results=lambda *args, **kwargs: None)
    _install_module(
        monkeypatch,
        "utils.metrics",
        get_metrics=lambda *args, **kwargs: None,
        get_rot_matrix=lambda *args, **kwargs: None,
    )
    _install_module(monkeypatch, "utils.so3_visualize", visualize_so3=lambda *args, **kwargs: None)
    _install_module(monkeypatch, "utils.visualize", create_grid_image=lambda *args, **kwargs: None)
    _install_module(monkeypatch, "utils.transforms")
    _install_module(monkeypatch, "cutoop.utils", draw_3d_bbox=lambda *args, **kwargs: None)
    _install_module(monkeypatch, "cutoop.transform")
    _install_module(monkeypatch, "cutoop.data_types")
    _install_module(monkeypatch, "cutoop.eval_utils")

    return importlib.import_module("runners.trainer")


def test_freeze_pose_params_only_keeps_segmentation_modules_trainable(trainer_module):
    agent = FakeAgent()

    trainer_module.freeze_pose_params(agent)

    trainable = {name for name, param in agent.net.named_parameters() if param.requires_grad}

    assert trainable == {
        "eomt_head.weight",
        "eomt_head.bias",
        "dino_wrapper.query_embed.weight",
        "dino_wrapper.seg_blocks.0.weight",
        "dino_wrapper.seg_blocks.0.bias",
        "dino_wrapper.seg_blocks.1.weight",
        "dino_wrapper.seg_blocks.1.bias",
        "dino_wrapper.seg_norm.weight",
        "dino_wrapper.seg_norm.bias",
    }
    assert not any(param.requires_grad for param in agent.net.dino_wrapper.dino.parameters())
    assert not any(param.requires_grad for param in agent.net.backbone.parameters())


@pytest.mark.parametrize("value", [None, ""])
def test_validate_segmentation_pose_init_rejects_falsy_checkpoint(trainer_module, value):
    cfg = SimpleNamespace(agent_type="segmentation", pretrained_score_model_path=value)

    with pytest.raises(ValueError, match="pretrained_score_model_path"):
        trainer_module.validate_segmentation_pose_init(cfg)


def test_validate_segmentation_pose_init_accepts_checkpoint(trainer_module):
    cfg = SimpleNamespace(
        agent_type="segmentation",
        pretrained_score_model_path="/tmp/pose-init.pth",
    )

    assert trainer_module.validate_segmentation_pose_init(cfg) == "/tmp/pose-init.pth"


def test_freeze_pose_params_raises_when_required_segmentation_modules_missing(trainer_module):
    class IncompleteSegNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dino_wrapper = nn.Module()
            self.dino_wrapper.query_embed = nn.Embedding(3, 4)

    agent = SimpleNamespace(net=IncompleteSegNet())

    with pytest.raises(RuntimeError, match="eomt_head|dino_wrapper.seg_blocks"):
        trainer_module.freeze_pose_params(agent)


def test_build_segmentation_training_agent_uses_score_style_config_and_preserves_training_cfg(
    monkeypatch, trainer_module
):
    created_cfgs = []
    load_calls = []
    freeze_calls = []

    class RecordingAgent:
        def __init__(self, cfg):
            created_cfgs.append(cfg)
            self.cfg = cfg
            self.net = FakeNet()
            self.clock = SimpleNamespace(epoch=0, step=0)

        def load_ckpt(self, **kwargs):
            load_calls.append(kwargs)

    def fake_freeze_pose_params(agent):
        freeze_calls.append(agent)

    monkeypatch.setattr(trainer_module, "PoseNet", RecordingAgent)
    monkeypatch.setattr(trainer_module, "freeze_pose_params", fake_freeze_pose_params)

    cfg = SimpleNamespace(
        agent_type="segmentation",
        enable_segmentation=False,
        pretrained_score_model_path="/tmp/pose-init.pth",
    )

    agent = trainer_module.build_segmentation_training_agent(cfg)

    assert cfg.agent_type == "segmentation"
    assert cfg.enable_segmentation is False
    assert len(created_cfgs) == 1
    assert created_cfgs[0] is agent.cfg
    assert agent.cfg is not cfg
    assert agent.cfg.agent_type == "score"
    assert agent.cfg.enable_segmentation is True
    assert load_calls == [
        {
            "model_dir": "/tmp/pose-init.pth",
            "model_path": True,
            "load_model_only": True,
        }
    ]
    assert freeze_calls == [agent]
