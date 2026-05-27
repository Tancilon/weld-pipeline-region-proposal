from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

REGION_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REGION_ROOT.parent
if str(REGION_ROOT) not in sys.path:
    sys.path.insert(0, str(REGION_ROOT))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from runners.aiws_genpose_part import GenPosePartEstimator


def test_estimate_part_calls_infer_pose_and_size(monkeypatch, tmp_path):
    rgb_path = tmp_path / "rgb.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(rgb_path)
    Image.fromarray(np.ones((4, 5), dtype=np.uint8) * 255).save(mask_path)
    depth = np.ones((4, 5), dtype=np.float32) * 0.5
    calls = {}

    class FakeOmni:
        @staticmethod
        def depth_to_pcl(roi_depth, K_33, roi_coord_2d, valid_flat):
            return np.column_stack(
                [
                    np.linspace(0.0, 0.1, int(valid_flat.sum())),
                    np.linspace(0.0, 0.2, int(valid_flat.sum())),
                    np.ones(int(valid_flat.sum())) * 0.5,
                ]
            ).astype(np.float32)

        @staticmethod
        def sample_points(pcl, num_points):
            ids = np.arange(num_points) % len(pcl)
            return ids, pcl[ids]

        @staticmethod
        def rgb_transform(roi_rgb):
            return np.transpose(roi_rgb.astype(np.float32) / 255.0, (2, 0, 1))

    def fake_crop(data, bbox_center, scale, img_size, interpolation):
        if data.ndim == 2:
            return np.ones((img_size, img_size), dtype=data.dtype)
        return np.zeros((img_size, img_size, data.shape[2]), dtype=data.dtype)

    runtime = types.SimpleNamespace(
        cv2=types.SimpleNamespace(
            IMREAD_COLOR=1,
            COLOR_BGR2RGB=4,
            INTER_LINEAR=1,
            INTER_NEAREST=0,
            imread=lambda path, flags=None: np.zeros((4, 5, 3), dtype=np.uint8),
            cvtColor=lambda image, code: image,
        ),
        torch=__import__("torch"),
        Omni6DPoseDataSet=FakeOmni,
        aug_bbox_eval=lambda bbox, h, w: (np.array([2.0, 2.0]), 4.0),
        crop_resize_by_warp_affine=fake_crop,
        get_2d_coord_np=lambda width, height: np.zeros((2, height, width), dtype=np.float32),
        main_cfg=types.SimpleNamespace(device="cpu"),
        main_agent=object(),
        energy_agent=object(),
        scale_agent=object(),
        infer_pose_and_size=lambda **kwargs: calls.setdefault(
            "result",
            {
                "R": np.eye(3, dtype=np.float32),
                "t": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "size": np.array([0.101, 0.201, 0.101], dtype=np.float32),
                "size_source": "scale_net",
            },
        ),
    )

    estimator = GenPosePartEstimator(
        seg_ckpt="seg.pth",
        energy_ckpt="energy.pth",
        scale_ckpt="scale.pth",
        device="cpu",
        repeat_num=2,
        num_points=16,
        img_size=8,
    )
    monkeypatch.setattr(estimator, "_load_runtime_bundle", lambda: runtime)

    result = estimator.estimate_part(
        rgb_path=rgb_path,
        depth=depth,
        mask_path=mask_path,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 2.0, "cy": 2.0},
    )

    assert result["size_source"] == "scale_net"
    np.testing.assert_allclose(result["size"], [0.101, 0.201, 0.101])
    assert calls["result"]["t"].shape == (3,)
