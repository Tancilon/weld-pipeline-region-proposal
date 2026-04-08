import importlib
import json
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
    _install_module(
        monkeypatch,
        "utils.experiment_logger",
        update_summary_json=lambda *args, **kwargs: None,
    )
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


def test_resolve_full_checkpoint_path_rejects_segmentation_resume_without_dedicated_checkpoint(
    trainer_module,
):
    cfg = SimpleNamespace(
        agent_type="segmentation",
        use_pretrain=True,
        eval=False,
        pred=False,
        pretrained_score_model_path="/tmp/pose-init.pth",
    )

    with pytest.raises(ValueError, match="segmentation.*full checkpoint"):
        trainer_module.resolve_full_checkpoint_path(cfg)


def test_train_segmentation_updates_latest_and_best_from_val_mask_iou(
    monkeypatch, trainer_module
):
    processed_batches = []
    monkeypatch.setattr(
        trainer_module,
        "process_batch_seg",
        lambda batch, device: processed_batches.append((batch, device)) or batch,
    )

    train_loader = [[{"train_epoch": 0}], [{"train_epoch": 1}]]
    val_loader = [{"val_batch": 0}, {"val_batch": 1}]

    eval_results = iter(
        [
            {"cls_loss": torch.tensor(1.0), "mask_loss": torch.tensor(2.0), "dice_loss": torch.tensor(3.0), "total_loss": torch.tensor(6.0), "mask_iou": torch.tensor(0.50), "mask_dice": torch.tensor(0.60), "matched_count": torch.tensor(2.0)},
            {"cls_loss": torch.tensor(1.0), "mask_loss": torch.tensor(2.0), "dice_loss": torch.tensor(3.0), "total_loss": torch.tensor(6.0), "mask_iou": torch.tensor(0.70), "mask_dice": torch.tensor(0.80), "matched_count": torch.tensor(2.0)},
            {"cls_loss": torch.tensor(1.5), "mask_loss": torch.tensor(2.5), "dice_loss": torch.tensor(3.5), "total_loss": torch.tensor(7.5), "mask_iou": torch.tensor(0.60), "mask_dice": torch.tensor(0.70), "matched_count": torch.tensor(2.0)},
            {"cls_loss": torch.tensor(1.5), "mask_loss": torch.tensor(2.5), "dice_loss": torch.tensor(3.5), "total_loss": torch.tensor(7.5), "mask_iou": torch.tensor(0.60), "mask_dice": torch.tensor(0.90), "matched_count": torch.tensor(2.0)},
        ]
    )

    class FakeClock:
        def __init__(self):
            self.epoch = 0
            self.step = 0

        def tick(self):
            self.step += 1

        def tock(self):
            self.epoch += 1

    class FakeAgent:
        def __init__(self):
            self.clock = FakeClock()
            self.saved = []
            self.recorded = []

        def update_learning_rate(self):
            return None

        def train_func(self, data, gf_mode):
            return {"total_loss": torch.tensor(1.0)}

        def eval_func(self, data, data_mode, gf_mode):
            assert data_mode == "val"
            assert gf_mode == "segmentation"
            return next(eval_results)

        def record_losses(self, loss_dict, mode):
            self.recorded.append((mode, {k: float(v) for k, v in loss_dict.items()}))

        def save_ckpt(self, name=None):
            self.saved.append(name)

    cfg = SimpleNamespace(
        device="cpu",
        n_epochs=2,
        warmup=0,
        eval_freq=1,
        log_dir="seg-exp",
        nuclear_data_path="/tmp/nuclear",
        pretrained_score_model_path="/tmp/pose-init.pth",
        img_size=224,
        num_queries=50,
        query_injection_layer=6,
    )
    seg_agent = FakeAgent()

    trainer_module.train_segmentation(cfg, train_loader, val_loader, seg_agent)

    assert len(processed_batches) == 8
    assert seg_agent.saved == ["latest", "best", "latest", "best"]
    assert len(seg_agent.recorded) == 2
    assert seg_agent.recorded[0][0] == "val"
    assert seg_agent.recorded[0][1]["cls_loss"] == pytest.approx(1.0)
    assert seg_agent.recorded[0][1]["mask_loss"] == pytest.approx(2.0)
    assert seg_agent.recorded[0][1]["dice_loss"] == pytest.approx(3.0)
    assert seg_agent.recorded[0][1]["total_loss"] == pytest.approx(6.0)
    assert seg_agent.recorded[0][1]["mask_iou"] == pytest.approx(0.6, rel=1e-6)
    assert seg_agent.recorded[0][1]["mask_dice"] == pytest.approx(0.7, rel=1e-6)

    assert seg_agent.recorded[1][0] == "val"
    assert seg_agent.recorded[1][1]["cls_loss"] == pytest.approx(1.5)
    assert seg_agent.recorded[1][1]["mask_loss"] == pytest.approx(2.5)
    assert seg_agent.recorded[1][1]["dice_loss"] == pytest.approx(3.5)
    assert seg_agent.recorded[1][1]["total_loss"] == pytest.approx(7.5)
    assert seg_agent.recorded[1][1]["mask_iou"] == pytest.approx(0.6, rel=1e-6)
    assert seg_agent.recorded[1][1]["mask_dice"] == pytest.approx(0.8, rel=1e-6)


def test_train_segmentation_weights_mask_metrics_by_matched_instances(
    monkeypatch, trainer_module
):
    monkeypatch.setattr(trainer_module, "process_batch_seg", lambda batch, device: batch)

    eval_results = iter(
        [
            {
                "cls_loss": torch.tensor(1.0),
                "mask_loss": torch.tensor(2.0),
                "dice_loss": torch.tensor(3.0),
                "total_loss": torch.tensor(6.0),
                "mask_iou": torch.tensor(1.0),
                "mask_dice": torch.tensor(1.0),
                "matched_count": torch.tensor(1.0),
            },
            {
                "cls_loss": torch.tensor(1.0),
                "mask_loss": torch.tensor(2.0),
                "dice_loss": torch.tensor(3.0),
                "total_loss": torch.tensor(6.0),
                "mask_iou": torch.tensor(0.0),
                "mask_dice": torch.tensor(0.0),
                "matched_count": torch.tensor(10.0),
            },
        ]
    )

    class FakeClock:
        def __init__(self):
            self.epoch = 1
            self.step = 0

        def tick(self):
            self.step += 1

        def tock(self):
            self.epoch += 1

    class FakeAgent:
        def __init__(self):
            self.clock = FakeClock()
            self.recorded = []

        def update_learning_rate(self):
            return None

        def train_func(self, data, gf_mode):
            return {"total_loss": torch.tensor(1.0)}

        def eval_func(self, data, data_mode, gf_mode):
            return next(eval_results)

        def record_losses(self, loss_dict, mode):
            self.recorded.append((mode, {k: float(v) for k, v in loss_dict.items()}))

        def save_ckpt(self, name=None):
            return None

    cfg = SimpleNamespace(
        device="cpu",
        n_epochs=2,
        warmup=0,
        eval_freq=1,
        log_dir="seg-exp",
        nuclear_data_path="/tmp/nuclear",
        pretrained_score_model_path="/tmp/pose-init.pth",
        img_size=224,
        num_queries=50,
        query_injection_layer=6,
    )

    seg_agent = FakeAgent()
    trainer_module.train_segmentation(
        cfg,
        [[{"train": 0}]],
        [{"val": 0}, {"val": 1}],
        seg_agent,
    )

    assert seg_agent.recorded == [
        (
            "val",
            {
                "cls_loss": 1.0,
                "mask_loss": 2.0,
                "dice_loss": 3.0,
                "total_loss": 6.0,
                "mask_iou": pytest.approx(1.0 / 11.0, rel=1e-6),
                "mask_dice": pytest.approx(1.0 / 11.0, rel=1e-6),
                "cls_acc_matched": 0.0,
            },
        )
    ]


def test_train_segmentation_handles_zero_matched_instances_in_validation(
    monkeypatch, trainer_module
):
    monkeypatch.setattr(trainer_module, "process_batch_seg", lambda batch, device: batch)

    class FakeClock:
        def __init__(self):
            self.epoch = 1
            self.step = 0

        def tick(self):
            self.step += 1

        def tock(self):
            self.epoch += 1

    class FakeAgent:
        def __init__(self):
            self.clock = FakeClock()
            self.saved = []
            self.recorded = []

        def update_learning_rate(self):
            return None

        def train_func(self, data, gf_mode):
            return {"total_loss": torch.tensor(1.0)}

        def eval_func(self, data, data_mode, gf_mode):
            return {
                "cls_loss": torch.tensor(1.0),
                "mask_loss": torch.tensor(2.0),
                "dice_loss": torch.tensor(3.0),
                "total_loss": torch.tensor(6.0),
                "mask_iou": torch.tensor(0.0),
                "mask_dice": torch.tensor(0.0),
                "cls_acc_matched": torch.tensor(0.0),
                "matched_count": torch.tensor(0.0),
            }

        def record_losses(self, loss_dict, mode):
            self.recorded.append((mode, {k: float(v) for k, v in loss_dict.items()}))

        def save_ckpt(self, name=None):
            self.saved.append(name)

    cfg = SimpleNamespace(
        device="cpu",
        n_epochs=2,
        warmup=0,
        eval_freq=1,
        log_dir="seg-exp",
        nuclear_data_path="/tmp/nuclear",
        pretrained_score_model_path="/tmp/pose-init.pth",
        img_size=224,
        num_queries=50,
        query_injection_layer=6,
    )

    seg_agent = FakeAgent()
    trainer_module.train_segmentation(cfg, [[{"train": 0}]], [{"val": 0}], seg_agent)

    assert seg_agent.saved == ["latest", "best"]
    assert seg_agent.recorded == [
        (
            "val",
            {
                "cls_loss": 1.0,
                "mask_loss": 2.0,
                "dice_loss": 3.0,
                "total_loss": 6.0,
                "mask_iou": 0.0,
                "mask_dice": 0.0,
                "cls_acc_matched": 0.0,
            },
        )
    ]



def test_train_segmentation_writes_summary_with_best_and_latest(
    monkeypatch, tmp_path, trainer_module
):
    monkeypatch.setattr(trainer_module, "process_batch_seg", lambda batch, device: batch)

    def fake_update_summary_json(ckpt_dir, summary_update):
        summary_path = Path(ckpt_dir) / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:
            summary = {}
        summary.update(summary_update)
        summary_path.write_text(json.dumps(summary, sort_keys=True))
        return summary_path

    monkeypatch.setattr(trainer_module, "update_summary_json", fake_update_summary_json)

    class FakeClock:
        def __init__(self):
            self.epoch = 0
            self.step = 0

        def tick(self):
            self.step += 1

        def tock(self):
            self.epoch += 1

    class FakeAgent:
        def __init__(self):
            self.clock = FakeClock()
            self.model_dir = tmp_path.as_posix()

        def update_learning_rate(self):
            return None

        def train_func(self, data, gf_mode):
            return {"total_loss": torch.tensor(1.0)}

        def eval_func(self, data, data_mode, gf_mode):
            return {
                "cls_loss": torch.tensor(1.0),
                "mask_loss": torch.tensor(2.0),
                "dice_loss": torch.tensor(3.0),
                "total_loss": torch.tensor(6.0),
                "mask_iou": torch.tensor(0.75),
                "mask_dice": torch.tensor(0.80),
            }

        def record_losses(self, loss_dict, mode):
            return None

        def save_ckpt(self, name=None):
            return None

    cfg = SimpleNamespace(
        device="cpu",
        n_epochs=1,
        warmup=0,
        eval_freq=1,
        log_dir="seg-exp",
        nuclear_data_path="/tmp/nuclear",
        pretrained_score_model_path="/tmp/pose-init.pth",
        img_size=224,
        num_queries=50,
        query_injection_layer=6,
    )

    trainer_module.train_segmentation(cfg, [[{"train": 0}]], [{"val": 0}], FakeAgent())

    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload == {
        "best_epoch": 1,
        "best_mask_dice": 0.8,
        "best_mask_iou": 0.75,
        "dataset_path": "/tmp/nuclear",
        "experiment_name": "seg-exp",
        "image_size": 224,
        "latest_epoch": 1,
        "latest_mask_dice": 0.8,
        "latest_mask_iou": 0.75,
        "num_queries": 50,
        "pose_init_checkpoint": "/tmp/pose-init.pth",
        "query_injection_layer": 6,
    }
