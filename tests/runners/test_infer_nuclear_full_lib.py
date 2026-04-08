import builtins
import importlib
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_module(monkeypatch, name, package=False, **attrs):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_runtime_deps(monkeypatch, include_sklearn=True):
    monkeypatch.delitem(sys.modules, "runners.infer_nuclear_full_lib", raising=False)

    cv2 = _install_module(monkeypatch, "cv2")
    cv2.INTER_LINEAR = 1
    cv2.INTER_NEAREST = 0
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

    _install_module(monkeypatch, "datasets", package=True)
    _install_module(monkeypatch, "networks", package=True)
    _install_module(monkeypatch, "utils", package=True)
    _install_module(monkeypatch, "cutoop", package=True)
    if include_sklearn:
        _install_module(monkeypatch, "sklearn", package=True)

    class Omni6DPoseDataSet:
        @staticmethod
        def depth_to_pcl(*args, **kwargs):
            return []

        @staticmethod
        def sample_points(points, num_points):
            return [], points

        @staticmethod
        def rgb_transform(rgb):
            return rgb

    _install_module(
        monkeypatch,
        "datasets.datasets_nuclear",
        CLASS_NAMES=["a", "b", "c", "d", "e", "f"],
        NUM_CLASSES=6,
        collate_nuclear=lambda batch: batch,
        process_batch_seg=lambda batch, device: batch,
        NuclearWorkpieceDataset=object,
    )
    _install_module(
        monkeypatch,
        "datasets.datasets_omni6dpose",
        Omni6DPoseDataSet=Omni6DPoseDataSet,
    )
    _install_module(
        monkeypatch,
        "cutoop.data_types",
        CameraIntrinsicsBase=type("CameraIntrinsicsBase", (), {}),
    )
    _install_module(
        monkeypatch,
        "cutoop.eval_utils",
        DetectMatch=type(
            "DetectMatch",
            (),
            {"_draw_image": staticmethod(lambda **kwargs: kwargs["vis_img"])},
        ),
    )
    _install_module(
        monkeypatch,
        "networks.reward",
        sort_poses_by_energy=lambda pred_pose, pred_energy: (pred_pose, pred_energy),
    )
    _install_module(
        monkeypatch,
        "utils.datasets_utils",
        aug_bbox_eval=lambda bbox, im_h, im_w: (bbox[:2], 1.0),
        crop_resize_by_warp_affine=lambda img, *args, **kwargs: img,
        get_2d_coord_np=lambda w, h: torch.zeros(2, h, w).numpy(),
    )
    _install_module(
        monkeypatch,
        "utils.misc",
        average_quaternion_batch=lambda quat: quat[:, 0, :],
    )
    _install_module(
        monkeypatch,
        "utils.metrics",
        get_rot_matrix=lambda rot, pose_mode: torch.eye(3).repeat(rot.shape[0], 1, 1),
    )
    _install_module(
        monkeypatch,
        "utils.transforms",
        matrix_to_quaternion=lambda matrix: torch.zeros(matrix.shape[0], 4),
        quaternion_to_matrix=lambda quat: torch.eye(3).repeat(quat.shape[0], 1, 1),
    )

    if include_sklearn:
        class FakeDBSCAN:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, x):
                self.labels_ = [-1] * len(x)
                return self

        _install_module(monkeypatch, "sklearn.cluster", DBSCAN=FakeDBSCAN)


@pytest.fixture
def runtime_lib(monkeypatch):
    _install_runtime_deps(monkeypatch)

    return importlib.import_module("runners.infer_nuclear_full_lib")


def test_runtime_lib_import_and_parser_work_without_runtime_dependencies(monkeypatch):
    blocked_roots = {"cv2", "datasets", "networks", "sklearn", "cutoop"}
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root in blocked_roots:
            raise AssertionError(f"Unexpected eager import of runtime module: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.delitem(sys.modules, "runners.infer_nuclear_full_lib", raising=False)

    module = importlib.import_module("runners.infer_nuclear_full_lib")

    parser = module.build_arg_parser()
    args = parser.parse_args(["--nuclear_data_path", "/tmp/data", "--seg_ckpt", "/tmp/seg.pth"])
    assert args.seg_ckpt == "/tmp/seg.pth"


def test_runner_help_exits_successfully():
    result = subprocess.run(
        [sys.executable, "runners/infer_nuclear_full.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "--seg_ckpt" in result.stdout


def test_build_arg_parser_has_no_pose_checkpoint_argument(runtime_lib):
    parser = runtime_lib.build_arg_parser()

    args = parser.parse_args(
        [
            "--nuclear_data_path",
            "/tmp/data",
            "--seg_ckpt",
            "/tmp/seg.pth",
            "--energy_ckpt",
            "/tmp/energy.pth",
            "--scale_ckpt",
            "/tmp/scale.pth",
        ]
    )

    assert args.seg_ckpt == "/tmp/seg.pth"
    assert args.energy_ckpt == "/tmp/energy.pth"
    assert args.scale_ckpt == "/tmp/scale.pth"
    assert not hasattr(args, "pose_ckpt")

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--nuclear_data_path",
                "/tmp/data",
                "--seg_ckpt",
                "/tmp/seg.pth",
                "--pose_ckpt_path",
                "/tmp/pose.pth",
            ]
        )


def test_validate_single_agent_seg_state_dict_rejects_legacy_checkpoint(runtime_lib):
    legacy_state_dict = {
        "dino_wrapper.dino.blocks.0.attn.qkv.weight": torch.zeros(1),
        "pose_score_net.fusion_tail.weight": torch.zeros(1),
    }

    with pytest.raises(ValueError, match="single-agent|legacy|query_embed|seg_blocks"):
        runtime_lib.validate_single_agent_seg_state_dict(legacy_state_dict)


def test_validate_single_agent_seg_state_dict_accepts_single_agent_checkpoint(runtime_lib):
    state_dict = {
        "dino_wrapper.query_embed.weight": torch.zeros(1),
        "eomt_head.class_head.weight": torch.zeros(1),
        "dino_wrapper.seg_blocks.0.weight": torch.zeros(1),
        "dino_wrapper.dino.blocks.0.attn.qkv.weight": torch.zeros(1),
        "pose_score_net.fusion_tail.weight": torch.zeros(1),
    }

    validated = runtime_lib.validate_single_agent_seg_state_dict(state_dict)

    assert validated is state_dict


def test_load_main_agent_checkpoint_loads_and_wraps_errors(runtime_lib, monkeypatch):
    state_dict = {
        "dino_wrapper.query_embed.weight": torch.zeros(1),
        "eomt_head.class_head.weight": torch.zeros(1),
        "dino_wrapper.seg_blocks.0.weight": torch.zeros(1),
        "dino_wrapper.dino.blocks.0.attn.qkv.weight": torch.zeros(1),
        "pose_score_net.fusion_tail.weight": torch.zeros(1),
    }

    class FakeNet:
        def __init__(self, should_fail=False):
            self.should_fail = should_fail
            self.load_calls = []

        def load_state_dict(self, loaded_state_dict, strict):
            self.load_calls.append((loaded_state_dict, strict))
            if self.should_fail:
                raise RuntimeError("size mismatch for head.weight")
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    class FakeAgent:
        def __init__(self, should_fail=False):
            self.net = FakeNet(should_fail=should_fail)

    monkeypatch.setattr(
        runtime_lib,
        "get_torch_module",
        lambda: SimpleNamespace(load=lambda path, map_location=None: {"model_state_dict": state_dict}),
    )

    ok_agent = FakeAgent()
    runtime_lib.load_main_agent_checkpoint(ok_agent, "/tmp/seg-ok.pth")
    assert ok_agent.net.load_calls == [(state_dict, True)]

    bad_agent = FakeAgent(should_fail=True)
    with pytest.raises(ValueError, match="seg-bad.pth.*size mismatch for head.weight"):
        runtime_lib.load_main_agent_checkpoint(bad_agent, "/tmp/seg-bad.pth")


def test_init_pipeline_agents_builds_one_main_agent_plus_optional_aux_agents(
    runtime_lib, monkeypatch
):
    created_cfgs = []
    main_loads = []
    aux_loads = []

    class FakeNet:
        def __init__(self):
            self.eval_called = False

        def eval(self):
            self.eval_called = True

    class FakePoseNet:
        def __init__(self, cfg):
            self.cfg = cfg
            self.net = FakeNet()
            created_cfgs.append(cfg)

    monkeypatch.setattr(runtime_lib, "get_posenet_class", lambda: FakePoseNet)
    monkeypatch.setattr(
        runtime_lib,
        "load_main_agent_checkpoint",
        lambda agent, ckpt_path: main_loads.append((agent.cfg.agent_type, agent.cfg.enable_segmentation, ckpt_path)),
    )
    monkeypatch.setattr(
        runtime_lib,
        "load_model_only",
        lambda agent, ckpt_path, name: aux_loads.append((name, agent.cfg.agent_type, agent.cfg.enable_segmentation, ckpt_path)),
    )

    args = SimpleNamespace(
        seg_ckpt="/tmp/seg-main.pth",
        energy_ckpt="/tmp/energy.pth",
        scale_ckpt="/tmp/scale.pth",
        device="cpu",
        num_points=1024,
        img_size=224,
        repeat_num=10,
    )

    main_cfg, main_agent, energy_agent, scale_agent = runtime_lib.init_pipeline_agents(args)

    assert main_cfg is main_agent.cfg
    assert main_cfg.enable_segmentation is True
    assert main_cfg.agent_type == "score"
    assert energy_agent.cfg.agent_type == "energy"
    assert scale_agent.cfg.agent_type == "scale"
    assert [cfg.agent_type for cfg in created_cfgs] == ["score", "energy", "scale"]
    assert [cfg.enable_segmentation for cfg in created_cfgs] == [True, False, False]
    assert [cfg.agent_type for cfg in created_cfgs].count("score") == 1
    assert main_loads == [("score", True, "/tmp/seg-main.pth")]
    assert aux_loads == [
        ("energy", "energy", False, "/tmp/energy.pth"),
        ("scale", "scale", False, "/tmp/scale.pth"),
    ]
    assert main_agent.net.eval_called is True
    assert energy_agent.net.eval_called is True
    assert scale_agent.net.eval_called is True
