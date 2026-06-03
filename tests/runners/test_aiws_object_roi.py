from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

REGION_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REGION_ROOT.parent
if str(REGION_ROOT) not in sys.path:
    sys.path.insert(0, str(REGION_ROOT))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from runners.aiws_object_roi import (
    ObjectRoiError,
    ObjectRoiEstimator,
    build_object_roi_from_mask,
    normalize_eomt_class_name,
)


def test_normalize_eomt_class_name_maps_supported_classes():
    assert normalize_eomt_class_name("盖板") == "cover_plate"
    assert normalize_eomt_class_name("方管") == "square_tube"
    assert normalize_eomt_class_name("喇叭口") == "bellmouth"


def test_normalize_eomt_class_name_rejects_unknown_class():
    with pytest.raises(ObjectRoiError, match="unsupported EoMT class"):
        normalize_eomt_class_name("未知")


def test_build_object_roi_from_mask_writes_masked_full_rgb_and_metadata(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "out" / "0034_depth.png"
    mask_path = tmp_path / "external_mask.png"
    rgb = np.zeros((12, 14, 3), dtype=np.uint8)
    rgb[:, :] = [10, 20, 30]
    rgb[3:10, 4:12] = [80, 90, 100]
    Image.fromarray(rgb).save(rgb_path)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ones((12, 14), dtype=np.uint16) * 500).save(depth_path)
    mask = np.zeros((12, 14), dtype=np.uint8)
    mask[3:10, 4:12] = 255
    Image.fromarray(mask, mode="L").save(mask_path)

    result = build_object_roi_from_mask(
        rgb_path=rgb_path,
        mask_path=mask_path,
        output_dir=tmp_path / "out",
        sample_id="0034",
        workpiece_type="square_tube",
        class_id=-1,
        class_confidence=1.0,
        source_depth_path=depth_path,
    )

    assert result.workpiece_type == "square_tube"
    assert result.bbox_xywh == [4, 3, 8, 7]
    assert result.object_mask.shape == (12, 14)
    assert result.object_mask_path.exists()
    assert result.masked_full_rgb_path.exists()
    masked_rgb = np.asarray(Image.open(result.masked_full_rgb_path).convert("RGB"))
    assert masked_rgb.shape == rgb.shape
    assert (masked_rgb[0, 0] == [255, 255, 255]).all()
    assert (masked_rgb[4, 5] == [80, 90, 100]).all()
    payload = json.loads(result.roi_metadata_path.read_text(encoding="utf-8"))
    assert payload["workpiece_type"] == "square_tube"
    assert payload["source_shape_hw"] == [12, 14]
    assert payload["source_depth_path"] == "0034_depth.png"
    assert payload["semantic_sam_input_mode"] == "masked_full"
    assert payload["masked_full_rgb_path"] == "object_roi/rgb_masked_full.png"
    assert payload["background_fill_rgb"] == [255, 255, 255]
    assert "crop_rgb_path" not in payload
    assert "crop_bbox_xywh" not in payload


def test_build_object_roi_from_mask_keeps_largest_component_and_records_filter(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    mask_path = tmp_path / "noisy_mask.png"
    rgb = np.zeros((12, 14, 3), dtype=np.uint8)
    rgb[:, :] = [10, 20, 30]
    Image.fromarray(rgb).save(rgb_path)
    mask = np.zeros((12, 14), dtype=np.uint8)
    mask[3:10, 4:12] = 255
    mask[0, 0] = 255
    mask[1, 12] = 255
    mask[11, 2] = 255
    Image.fromarray(mask, mode="L").save(mask_path)

    result = build_object_roi_from_mask(
        rgb_path=rgb_path,
        mask_path=mask_path,
        output_dir=tmp_path / "out",
        sample_id="0034",
        workpiece_type="square_tube",
        class_id=-1,
        class_confidence=1.0,
    )

    expected = np.zeros((12, 14), dtype=bool)
    expected[3:10, 4:12] = True
    np.testing.assert_array_equal(result.object_mask, expected)
    assert result.bbox_xywh == [4, 3, 8, 7]
    saved = np.asarray(Image.open(result.object_mask_path).convert("L")) > 0
    np.testing.assert_array_equal(saved, expected)
    payload = json.loads(result.roi_metadata_path.read_text(encoding="utf-8"))
    assert payload["component_filter"] == {
        "enabled": True,
        "connectivity": 8,
        "num_components_before": 4,
        "kept_area": 56,
        "removed_area": 3,
        "removed_area_ratio": pytest.approx(3 / 59),
    }


def test_build_object_roi_from_mask_rejects_empty_mask(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    mask_path = tmp_path / "empty.png"
    Image.fromarray(np.zeros((12, 14, 3), dtype=np.uint8)).save(rgb_path)
    Image.fromarray(np.zeros((12, 14), dtype=np.uint8), mode="L").save(mask_path)

    with pytest.raises(ObjectRoiError, match="empty object mask"):
        build_object_roi_from_mask(
            rgb_path=rgb_path,
            mask_path=mask_path,
            output_dir=tmp_path / "out",
            sample_id="0034",
            workpiece_type="square_tube",
            class_id=-1,
            class_confidence=1.0,
        )


def test_object_roi_estimator_uses_top_eomt_prediction(monkeypatch, tmp_path):
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((12, 14, 3), dtype=np.uint8)).save(rgb_path)

    class FakeCv2:
        INTER_LINEAR = 1
        COLOR_BGR2RGB = 4
        IMREAD_COLOR = 1

        @staticmethod
        def imread(path, flags=None):
            return np.zeros((12, 14, 3), dtype=np.uint8)

        @staticmethod
        def cvtColor(image, code):
            return image

        @staticmethod
        def resize(image, size, interpolation=None):
            width, height = size
            if image.ndim == 2:
                return np.ones((height, width), dtype=image.dtype)
            return np.zeros((height, width, image.shape[2]), dtype=image.dtype)

    class FakeNet:
        def __call__(self, batch, mode):
            assert mode == "segmentation"
            class_logits = torch.tensor([[[0.1, 4.0, 0.2, 0.0]]], dtype=torch.float32)
            mask_logits = torch.ones((1, 1, 2, 2), dtype=torch.float32) * 10.0
            return class_logits, mask_logits

    runtime = SimpleNamespace(
        cv2=FakeCv2,
        torch=torch,
        CLASS_NAMES=["盖板", "方管", "喇叭口"],
        main_cfg=SimpleNamespace(device="cpu"),
        main_agent=SimpleNamespace(net=FakeNet()),
    )

    estimator = ObjectRoiEstimator(seg_ckpt="seg.pth", device="cpu", img_size=4)
    monkeypatch.setattr(estimator, "_load_runtime_bundle", lambda: runtime)

    result = estimator.estimate(
        rgb_path=rgb_path,
        output_dir=tmp_path / "out",
        sample_id="0034",
        score_threshold=0.5,
    )

    assert result.workpiece_type == "square_tube"
    assert result.class_id == 1
    assert result.class_confidence > 0.9
    assert result.object_mask.shape == (12, 14)
    assert result.object_mask_path.exists()
    assert result.masked_full_rgb_path.exists()
