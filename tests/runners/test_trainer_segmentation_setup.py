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


def _install_optional_module_stubs():
    def _install_module(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        return module

    def _install_package(name):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        return module

    if "cv2" not in sys.modules:
        cv2 = types.ModuleType("cv2")
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
        sys.modules["cv2"] = cv2

    if "ipdb" not in sys.modules:
        ipdb = types.ModuleType("ipdb")
        ipdb.set_trace = lambda *args, **kwargs: None
        sys.modules["ipdb"] = ipdb

    if "tensorboardX" not in sys.modules:
        tensorboardx = types.ModuleType("tensorboardX")

        class SummaryWriter:
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def close(self):
                pass

        tensorboardx.SummaryWriter = SummaryWriter
        sys.modules["tensorboardX"] = tensorboardx

    if "matplotlib" not in sys.modules:
        matplotlib = types.ModuleType("matplotlib")
        pyplot = types.ModuleType("matplotlib.pyplot")
        matplotlib.rc = lambda *args, **kwargs: None
        pyplot.figure = lambda *args, **kwargs: None
        pyplot.imshow = lambda *args, **kwargs: None
        pyplot.show = lambda *args, **kwargs: None
        pyplot.savefig = lambda *args, **kwargs: None
        pyplot.close = lambda *args, **kwargs: None
        pyplot.subplots = lambda *args, **kwargs: (None, None)
        pyplot.axis = lambda *args, **kwargs: None
        matplotlib.pyplot = pyplot
        sys.modules["matplotlib"] = matplotlib
        sys.modules["matplotlib.pyplot"] = pyplot

    if "scipy" not in sys.modules:
        scipy = types.ModuleType("scipy")
        spatial = types.ModuleType("scipy.spatial")
        transform = types.ModuleType("scipy.spatial.transform")

        class Rotation:
            @classmethod
            def from_matrix(cls, *args, **kwargs):
                return cls()

            def as_matrix(self):
                return None

        transform.Rotation = Rotation
        spatial.transform = transform
        scipy.spatial = spatial
        sys.modules["scipy"] = scipy
        sys.modules["scipy.spatial"] = spatial
        sys.modules["scipy.spatial.transform"] = transform

    _install_package("datasets")
    _install_package("networks")
    _install_package("utils")
    _install_package("cutoop")

    _install_module(
        "datasets.datasets_omni6dpose",
        get_data_loaders_from_cfg=lambda *args, **kwargs: None,
        process_batch=lambda *args, **kwargs: None,
        array_to_SymLabel=lambda *args, **kwargs: None,
    )
    _install_module(
        "datasets.datasets_nuclear",
        get_nuclear_data_loaders=lambda *args, **kwargs: None,
        process_batch_seg=lambda *args, **kwargs: None,
    )
    _install_module("networks.posenet_agent", PoseNet=type("PoseNet", (), {}))
    _install_module(
        "utils.misc",
        exists_or_mkdir=lambda *args, **kwargs: None,
        get_pose_representation=lambda *args, **kwargs: None,
        average_quaternion_batch=lambda *args, **kwargs: None,
        parallel_setup=lambda *args, **kwargs: None,
        parallel_cleanup=lambda *args, **kwargs: None,
    )
    _install_module("utils.genpose_utils", merge_results=lambda *args, **kwargs: None)
    _install_module(
        "utils.metrics",
        get_metrics=lambda *args, **kwargs: None,
        get_rot_matrix=lambda *args, **kwargs: None,
    )
    _install_module("utils.so3_visualize", visualize_so3=lambda *args, **kwargs: None)
    _install_module("utils.visualize", create_grid_image=lambda *args, **kwargs: None)
    _install_module("utils.transforms")
    _install_module("cutoop.utils", draw_3d_bbox=lambda *args, **kwargs: None)
    _install_module("cutoop.transform")
    _install_module("cutoop.data_types")
    _install_module("cutoop.eval_utils")


def import_trainer():
    original_argv = sys.argv[:]
    sys.argv = [original_argv[0]]
    try:
        _install_optional_module_stubs()
        sys.modules.pop("runners.trainer", None)
        return importlib.import_module("runners.trainer")
    finally:
        sys.argv = original_argv


def test_freeze_pose_params_only_keeps_segmentation_modules_trainable():
    trainer = import_trainer()
    agent = FakeAgent()

    trainer.freeze_pose_params(agent)

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
def test_validate_segmentation_pose_init_rejects_falsy_checkpoint(value):
    trainer = import_trainer()
    cfg = SimpleNamespace(agent_type="segmentation", pretrained_score_model_path=value)

    with pytest.raises(ValueError, match="pretrained_score_model_path"):
        trainer.validate_segmentation_pose_init(cfg)
