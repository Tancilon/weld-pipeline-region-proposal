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

from runners.aiws_auto_part_selection import select_weld_focus_masks


def _write_mask(
    path: Path,
    rows: slice,
    cols: slice,
    shape: tuple[int, int] = (20, 30),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rows, cols] = 255
    Image.fromarray(mask, mode="L").save(path)
    return path


def _candidate(
    mask_id: str,
    mask_path: Path,
    area_px: int,
    bbox_xywh: list[int],
    depth_valid_ratio: float,
    overlap: float,
) -> dict:
    return {
        "mask_id": mask_id,
        "mask_path": str(mask_path),
        "area_px": area_px,
        "bbox_xywh": bbox_xywh,
        "predicted_iou": 0.96,
        "stability_score": 0.95,
        "depth_valid_pixels": int(area_px * depth_valid_ratio),
        "depth_valid_ratio": depth_valid_ratio,
        "object_overlap_ratio": overlap,
    }


def _write_candidates(
    output_root: Path,
    candidates: list[dict],
    workpiece_type: str = "square_tube",
) -> Path:
    path = output_root / "mask_candidates.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_id": "0034",
                "workpiece_type": workpiece_type,
                "semantic_sam": {"levels": [4], "model_type": "T"},
                "candidates": candidates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _object_mask() -> np.ndarray:
    mask = np.zeros((20, 30), dtype=np.bool_)
    mask[2:18, 4:26] = True
    return mask


def test_auto_selector_accepts_clear_tube_candidate(tmp_path):
    output_root = tmp_path
    object_mask = _object_mask()
    depth = np.ones((20, 30), dtype=np.float32) * 0.5
    good_mask = _write_mask(
        output_root / "semantic_sam/0034_masks/0034_mask_001.png",
        slice(4, 16),
        slice(7, 24),
    )
    tiny_mask = _write_mask(
        output_root / "semantic_sam/0034_masks/0034_mask_002.png",
        slice(5, 7),
        slice(8, 10),
    )
    candidates_path = _write_candidates(
        output_root,
        [
            _candidate("0034_mask_001", good_mask, 204, [7, 4, 17, 12], 0.75, 1.0),
            _candidate("0034_mask_002", tiny_mask, 4, [8, 5, 2, 2], 1.0, 1.0),
        ],
    )

    result = select_weld_focus_masks(
        candidates_path=candidates_path,
        output_root=output_root,
        weld_focus=["tube"],
        object_mask=object_mask,
        depth=depth,
    )

    assert result.accepted is True
    assert result.selected_parts_path == output_root / "selected_parts.auto.json"
    selected = json.loads(result.selected_parts_path.read_text(encoding="utf-8"))
    assert selected["focused_parts"] == {"tube": "0034_mask_001"}
    diagnostics = json.loads(
        (output_root / "auto_part_selection.json").read_text(encoding="utf-8")
    )
    assert diagnostics["focused_parts"]["tube"]["decision"] == "accepted"
    assert diagnostics["focused_parts"]["tube"]["selected_mask_id"] == "0034_mask_001"


def test_auto_selector_rejects_full_object_candidate(tmp_path):
    output_root = tmp_path
    object_mask = _object_mask()
    depth = np.ones((20, 30), dtype=np.float32) * 0.5
    full_mask = _write_mask(
        output_root / "semantic_sam/0034_masks/0034_mask_000.png",
        slice(2, 18),
        slice(4, 26),
    )
    candidates_path = _write_candidates(
        output_root,
        [
            _candidate(
                "0034_mask_000",
                full_mask,
                int(object_mask.sum()),
                [4, 2, 22, 16],
                0.7,
                1.0,
            )
        ],
    )

    result = select_weld_focus_masks(
        candidates_path=candidates_path,
        output_root=output_root,
        weld_focus=["tube"],
        object_mask=object_mask,
        depth=depth,
    )

    assert result.accepted is False
    assert result.selected_parts_path is None
    assert result.reason == "low_confidence_auto_selection"
    diagnostics = json.loads(
        (output_root / "auto_part_selection.json").read_text(encoding="utf-8")
    )
    rejected = diagnostics["focused_parts"]["tube"]["candidates"][0]
    assert "area_ratio_above_max" in rejected["rejection_reasons"]


def test_auto_selector_stops_when_best_score_has_small_margin(tmp_path):
    output_root = tmp_path
    object_mask = _object_mask()
    depth = np.ones((20, 30), dtype=np.float32) * 0.5
    mask_a = _write_mask(
        output_root / "semantic_sam/0034_masks/0034_mask_001.png",
        slice(4, 16),
        slice(7, 24),
    )
    mask_b = _write_mask(
        output_root / "semantic_sam/0034_masks/0034_mask_002.png",
        slice(5, 17),
        slice(8, 25),
    )
    candidates_path = _write_candidates(
        output_root,
        [
            _candidate("0034_mask_001", mask_a, 204, [7, 4, 17, 12], 0.75, 1.0),
            _candidate("0034_mask_002", mask_b, 204, [8, 5, 17, 12], 0.74, 1.0),
        ],
    )

    result = select_weld_focus_masks(
        candidates_path=candidates_path,
        output_root=output_root,
        weld_focus=["tube"],
        object_mask=object_mask,
        depth=depth,
        min_score_margin=0.10,
    )

    assert result.accepted is False
    diagnostics = json.loads(
        (output_root / "auto_part_selection.json").read_text(encoding="utf-8")
    )
    assert diagnostics["focused_parts"]["tube"]["decision"] == "rejected"
    assert diagnostics["focused_parts"]["tube"]["reason"] == "score_margin_below_threshold"
    assert not (output_root / "selected_parts.auto.json").exists()
