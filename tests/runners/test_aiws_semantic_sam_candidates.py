from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REGION_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REGION_ROOT.parent
if str(REGION_ROOT) not in sys.path:
    sys.path.insert(0, str(REGION_ROOT))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from runners.aiws_semantic_sam_candidates import (
    build_mask_candidates,
    write_selected_parts_template,
)


def _write_mask(path: Path, data: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((data.astype(np.uint8) * 255), mode="L").save(path)
    return path


def test_build_mask_candidates_computes_depth_metrics_and_sorts(tmp_path):
    masks_dir = tmp_path / "semantic_sam/0034_masks"
    mask_a = _write_mask(
        masks_dir / "0034_mask_000.png",
        np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=bool),
    )
    mask_b = _write_mask(
        masks_dir / "0034_mask_001.png",
        np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=bool),
    )
    depth = np.array(
        [[0.4, 0.5, 0.0], [0.6, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    object_mask = np.array(
        [[1, 1, 1], [1, 1, 0], [0, 0, 0]],
        dtype=bool,
    )
    metadata_path = tmp_path / "semantic_sam/0034_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "model_type": "T",
                "levels": [4, 5, 6],
                "overlay_path": str(tmp_path / "semantic_sam/0034_overlay.png"),
                "masks": [
                    {
                        "mask_index": 0,
                        "mask_path": str(mask_a),
                        "area": 2,
                        "bbox": [0, 0, 2, 1],
                        "predicted_iou": 0.99,
                        "stability_score": 0.98,
                    },
                    {
                        "mask_index": 1,
                        "mask_path": str(mask_b),
                        "area": 4,
                        "bbox": [0, 0, 3, 2],
                        "predicted_iou": 0.90,
                        "stability_score": 0.95,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    output_path = build_mask_candidates(
        metadata_path=metadata_path,
        output_path=tmp_path / "mask_candidates.json",
        sample_id="0034",
        workpiece_type="square_tube",
        depth=depth,
        object_mask=object_mask,
        output_root=tmp_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["semantic_sam"]["levels"] == [4, 5, 6]
    assert [item["mask_id"] for item in payload["candidates"]] == [
        "0034_mask_001",
        "0034_mask_000",
    ]
    first = payload["candidates"][0]
    assert first["depth_valid_pixels"] == 3
    assert first["depth_valid_ratio"] == 0.75
    assert first["object_overlap_ratio"] == 1.0
    assert first["mask_path"] == "semantic_sam/0034_masks/0034_mask_001.png"


def test_build_mask_candidates_resizes_masks_to_depth_shape_for_metrics(tmp_path):
    masks_dir = tmp_path / "semantic_sam/0034_masks"
    mask_path = _write_mask(
        masks_dir / "0034_mask_000.png",
        np.array([[1, 0, 0], [0, 0, 1]], dtype=bool),
    )
    metadata_path = tmp_path / "semantic_sam/0034_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "model_type": "T",
                "levels": [4, 5, 6],
                "overlay_path": str(tmp_path / "semantic_sam/0034_overlay.png"),
                "masks": [
                    {
                        "mask_index": 0,
                        "mask_path": str(mask_path),
                        "area": 2,
                        "bbox": [0, 0, 3, 2],
                        "predicted_iou": 0.99,
                        "stability_score": 0.98,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output_path = build_mask_candidates(
        metadata_path=metadata_path,
        output_path=tmp_path / "mask_candidates.json",
        sample_id="0034",
        workpiece_type="square_tube",
        depth=np.ones((4, 6), dtype=np.float32),
        object_mask=None,
        output_root=tmp_path,
    )

    candidate = json.loads(output_path.read_text(encoding="utf-8"))["candidates"][0]
    assert candidate["source_shape_hw"] == [2, 3]
    assert candidate["metrics_shape_hw"] == [4, 6]
    assert candidate["area_px"] == 8
    assert candidate["bbox_xywh"] == [0, 0, 6, 4]
    assert candidate["depth_valid_pixels"] == 8
    assert candidate["depth_valid_ratio"] == 1.0


def test_build_mask_candidates_maps_crop_masks_to_original_shape(tmp_path):
    masks_dir = tmp_path / "semantic_sam/0034_masks"
    mask_path = _write_mask(
        masks_dir / "0034_mask_000.png",
        np.array([[1, 0], [0, 1]], dtype=bool),
    )
    metadata_path = tmp_path / "semantic_sam/0034_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "model_type": "T",
                "levels": [4, 5, 6],
                "overlay_path": str(tmp_path / "semantic_sam/0034_overlay.png"),
                "masks": [
                    {
                        "mask_index": 0,
                        "mask_path": str(mask_path),
                        "area": 2,
                        "bbox": [0, 0, 2, 2],
                        "predicted_iou": 0.9,
                        "stability_score": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    object_mask = np.zeros((6, 8), dtype=bool)
    object_mask[1:5, 2:6] = True

    output_path = build_mask_candidates(
        metadata_path=metadata_path,
        output_path=tmp_path / "mask_candidates.json",
        sample_id="0034",
        workpiece_type="square_tube",
        depth=np.ones((6, 8), dtype=np.float32),
        object_mask=object_mask,
        output_root=tmp_path,
        crop_bbox_xywh=[2, 1, 4, 4],
    )

    candidate = json.loads(output_path.read_text(encoding="utf-8"))["candidates"][0]
    assert candidate["coordinate_space"] == "original_image"
    assert candidate["mask_coordinate_space"] == "semantic_sam_crop"
    assert candidate["crop_bbox_xywh"] == [2, 1, 4, 4]
    assert candidate["source_shape_hw"] == [2, 2]
    assert candidate["metrics_shape_hw"] == [6, 8]
    assert candidate["area_px"] == 8
    assert candidate["bbox_xywh"] == [2, 1, 4, 4]
    assert candidate["roi_bbox_xywh"] == [0, 0, 4, 4]
    assert candidate["object_overlap_ratio"] == 1.0


def test_write_selected_parts_template_uses_weld_focus(tmp_path):
    output_path = write_selected_parts_template(
        output_path=tmp_path / "selected_parts.template.json",
        sample_id="0034",
        workpiece_type="square_tube",
        weld_focus=["tube"],
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "sample_id": "0034",
        "workpiece_type": "square_tube",
        "focused_parts": {"tube": ""},
    }
