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


def _write_bbox_area_mask(
    path: Path,
    bbox_xywh: list[int],
    area_px: int,
    shape: tuple[int, int] = (1080, 1920),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(shape, dtype=np.uint8)
    x, y, width, height = bbox_xywh
    capacity = int(width) * int(height)
    if area_px > capacity:
        raise ValueError(f"area {area_px} exceeds bbox capacity {capacity}")
    yy, xx = np.unravel_index(np.arange(area_px), (int(height), int(width)))
    mask[int(y) + yy, int(x) + xx] = 255
    Image.fromarray(mask, mode="L").save(path)
    return path


def _bbox_area_object_mask(
    bbox_xywh: list[int],
    area_px: int,
    shape: tuple[int, int] = (1080, 1920),
) -> np.ndarray:
    x, y, width, height = bbox_xywh
    capacity = int(width) * int(height)
    if area_px > capacity:
        raise ValueError(f"area {area_px} exceeds bbox capacity {capacity}")
    mask = np.zeros(shape, dtype=np.bool_)
    yy, xx = np.unravel_index(np.arange(area_px), (int(height), int(width)))
    mask[int(y) + yy, int(x) + xx] = True
    return mask


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


def _f101_object_mask() -> np.ndarray:
    mask = np.zeros((1080, 1920), dtype=np.bool_)
    mask[44 : 44 + 685, 909 : 909 + 566] = True
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


def test_auto_selector_keeps_low_depth_ratio_candidate_eligible(tmp_path):
    output_root = tmp_path
    object_mask = _f101_object_mask()
    depth = np.ones(object_mask.shape, dtype=np.float32) * 0.5
    tube_mask = _write_mask(
        output_root / "semantic_sam/f101_masks/f101_mask_001.png",
        slice(46, 46 + 528),
        slice(995, 995 + 326),
        shape=object_mask.shape,
    )
    candidates_path = _write_candidates(
        output_root,
        [
            {
                "mask_id": "f101_mask_001",
                "mask_path": str(tube_mask),
                "area_px": 142287,
                "bbox_xywh": [995, 46, 326, 528],
                "predicted_iou": 0.993780791759491,
                "stability_score": 0.9915425181388855,
                "depth_valid_pixels": 16584,
                "depth_valid_ratio": 0.11655316367623184,
                "object_overlap_ratio": 0.99972590609121,
            }
        ],
        workpiece_type="square_tube",
    )

    result = select_weld_focus_masks(
        candidates_path=candidates_path,
        output_root=output_root,
        weld_focus=["tube"],
        object_mask=object_mask,
        depth=depth,
    )

    assert result.accepted is True
    selected = json.loads(result.selected_parts_path.read_text(encoding="utf-8"))
    assert selected["focused_parts"] == {"tube": "f101_mask_001"}
    diagnostics = json.loads(
        (output_root / "auto_part_selection.json").read_text(encoding="utf-8")
    )
    candidate = diagnostics["focused_parts"]["tube"]["candidates"][0]
    assert candidate["hard_rejection_reasons"] == []
    assert "depth_valid_ratio_soft_penalty" in candidate["soft_penalty_reasons"]
    assert "depth_valid_ratio_below_min" not in candidate["rejection_reasons"]


def test_square_tube_tube_prefers_vertical_upper_mask_over_plate_like_mask(tmp_path):
    output_root = tmp_path
    object_mask = _f101_object_mask()
    depth = np.ones(object_mask.shape, dtype=np.float32) * 0.5
    plate_like = _write_mask(
        output_root / "semantic_sam/f101_masks/f101_mask_000.png",
        slice(361, 361 + 366),
        slice(912, 912 + 561),
        shape=object_mask.shape,
    )
    tube_like = _write_mask(
        output_root / "semantic_sam/f101_masks/f101_mask_001.png",
        slice(46, 46 + 528),
        slice(995, 995 + 326),
        shape=object_mask.shape,
    )
    whole_like = _write_mask(
        output_root / "semantic_sam/f101_masks/f101_mask_002.png",
        slice(46, 46 + 680),
        slice(912, 912 + 561),
        shape=object_mask.shape,
    )
    candidates_path = _write_candidates(
        output_root,
        [
            {
                "mask_id": "f101_mask_000",
                "mask_path": str(plate_like),
                "area_px": 93914,
                "bbox_xywh": [912, 361, 561, 366],
                "predicted_iou": 0.9944691061973572,
                "stability_score": 0.9866863489151001,
                "depth_valid_pixels": 27643,
                "depth_valid_ratio": 0.2943437613135422,
                "object_overlap_ratio": 0.9993824136976382,
            },
            {
                "mask_id": "f101_mask_001",
                "mask_path": str(tube_like),
                "area_px": 142287,
                "bbox_xywh": [995, 46, 326, 528],
                "predicted_iou": 0.993780791759491,
                "stability_score": 0.9915425181388855,
                "depth_valid_pixels": 16584,
                "depth_valid_ratio": 0.11655316367623184,
                "object_overlap_ratio": 0.99972590609121,
            },
            {
                "mask_id": "f101_mask_002",
                "mask_path": str(whole_like),
                "area_px": 235896,
                "bbox_xywh": [912, 46, 561, 680],
                "predicted_iou": 0.9778555035591125,
                "stability_score": 0.9933527708053589,
                "depth_valid_pixels": 44011,
                "depth_valid_ratio": 0.18656950520568386,
                "object_overlap_ratio": 0.9995506494387357,
            },
        ],
        workpiece_type="square_tube",
    )

    result = select_weld_focus_masks(
        candidates_path=candidates_path,
        output_root=output_root,
        weld_focus=["tube"],
        object_mask=object_mask,
        depth=depth,
    )

    assert result.accepted is True
    selected = json.loads(result.selected_parts_path.read_text(encoding="utf-8"))
    assert selected["focused_parts"] == {"tube": "f101_mask_001"}
    diagnostics = json.loads(
        (output_root / "auto_part_selection.json").read_text(encoding="utf-8")
    )
    tube_payload = diagnostics["focused_parts"]["tube"]
    assert tube_payload["selected_mask_id"] == "f101_mask_001"
    assert tube_payload["eligible_best_mask_id"] == "f101_mask_001"
    ranked = {item["mask_id"]: item for item in tube_payload["candidates"]}
    assert ranked["f101_mask_001"]["score"] > ranked["f101_mask_000"]["score"]
    assert (
        ranked["f101_mask_001"]["category_score_components"]["verticality"]
        > ranked["f101_mask_000"]["category_score_components"]["verticality"]
    )
    assert ranked["f101_mask_000"]["category_score_components"]["plate_like_penalty"] > 0.30
    assert "area_ratio_above_max" in ranked["f101_mask_002"]["hard_rejection_reasons"]


def test_bellmouth_tube_prefers_l75_main_tube_over_bottom_plate(tmp_path):
    output_root = tmp_path
    object_area = 124615
    object_bbox = [908, 223, 360, 513]
    object_mask = _bbox_area_object_mask(object_bbox, object_area)
    depth = np.ones(object_mask.shape, dtype=np.float32) * 0.5
    bottom_plate = _write_bbox_area_mask(
        output_root / "semantic_sam/l75_masks/l75_mask_004.png",
        [961, 223, 228, 511],
        48634,
    )
    main_tube = _write_bbox_area_mask(
        output_root / "semantic_sam/l75_masks/l75_mask_000.png",
        [908, 267, 360, 229],
        73896,
    )
    lower_plate = _write_bbox_area_mask(
        output_root / "semantic_sam/l75_masks/l75_mask_002.png",
        [910, 400, 351, 336],
        70303,
    )
    full_object = _write_bbox_area_mask(
        output_root / "semantic_sam/l75_masks/l75_mask_001.png",
        [908, 223, 360, 513],
        122393,
    )
    candidates_path = _write_candidates(
        output_root,
        [
            _candidate(
                "l75_mask_004",
                bottom_plate,
                48634,
                [961, 223, 228, 511],
                0.3954435168811942,
                0.9993625858452934,
            ),
            _candidate(
                "l75_mask_000",
                main_tube,
                73896,
                [908, 267, 360, 229],
                0.3345377286997943,
                0.9992421781963841,
            ),
            _candidate(
                "l75_mask_002",
                lower_plate,
                70303,
                [910, 400, 351, 336],
                0.34934497816593885,
                0.9998435344153166,
            ),
            _candidate(
                "l75_mask_001",
                full_object,
                122393,
                [908, 223, 360, 513],
                0.35993071499187046,
                0.9997957399524483,
            ),
        ],
        workpiece_type="bellmouth",
    )

    result = select_weld_focus_masks(
        candidates_path=candidates_path,
        output_root=output_root,
        weld_focus=["tube"],
        object_mask=object_mask,
        depth=depth,
    )

    assert result.accepted is True
    selected = json.loads(result.selected_parts_path.read_text(encoding="utf-8"))
    assert selected["focused_parts"] == {"tube": "l75_mask_000"}
    diagnostics = json.loads(
        (output_root / "auto_part_selection.json").read_text(encoding="utf-8")
    )
    tube_payload = diagnostics["focused_parts"]["tube"]
    ranked = {item["mask_id"]: item for item in tube_payload["candidates"]}
    assert ranked["l75_mask_000"]["score"] > ranked["l75_mask_004"]["score"]
    assert "area_ratio_above_max" in ranked["l75_mask_001"]["hard_rejection_reasons"]


def test_bellmouth_tube_prefers_l148_body_over_side_plate(tmp_path):
    output_root = tmp_path
    object_area = 100000
    object_bbox = [601, 155, 649, 593]
    object_mask = _bbox_area_object_mask(object_bbox, object_area)
    depth = np.ones(object_mask.shape, dtype=np.float32) * 0.5
    side_plate = _write_bbox_area_mask(
        output_root / "semantic_sam/l148_masks/l148_mask_000.png",
        [601, 430, 649, 282],
        20671,
    )
    tube_body = _write_bbox_area_mask(
        output_root / "semantic_sam/l148_masks/l148_mask_001.png",
        [736, 155, 421, 582],
        75340,
    )
    full_object = _write_bbox_area_mask(
        output_root / "semantic_sam/l148_masks/l148_mask_003.png",
        [601, 155, 649, 593],
        97932,
    )
    candidates_path = _write_candidates(
        output_root,
        [
            _candidate(
                "l148_mask_000",
                side_plate,
                20671,
                [601, 430, 649, 282],
                0.3178502281842069,
                0.9996715530355414,
            ),
            _candidate(
                "l148_mask_001",
                tube_body,
                75340,
                [736, 155, 421, 582],
                0.3655111246021846,
                1.0,
            ),
            _candidate(
                "l148_mask_003",
                full_object,
                97932,
                [601, 155, 649, 593],
                0.35385058855295515,
                0.9999562142872781,
            ),
        ],
        workpiece_type="bellmouth",
    )

    result = select_weld_focus_masks(
        candidates_path=candidates_path,
        output_root=output_root,
        weld_focus=["tube"],
        object_mask=object_mask,
        depth=depth,
    )

    assert result.accepted is True
    selected = json.loads(result.selected_parts_path.read_text(encoding="utf-8"))
    assert selected["focused_parts"] == {"tube": "l148_mask_001"}
    diagnostics = json.loads(
        (output_root / "auto_part_selection.json").read_text(encoding="utf-8")
    )
    tube_payload = diagnostics["focused_parts"]["tube"]
    ranked = {item["mask_id"]: item for item in tube_payload["candidates"]}
    assert ranked["l148_mask_001"]["score"] > ranked["l148_mask_000"]["score"]
    assert "area_ratio_above_max" in ranked["l148_mask_003"]["hard_rejection_reasons"]


def test_cover_plate_tube_accepts_near_full_roi_tube_mask(tmp_path):
    output_root = tmp_path
    object_area = 100000
    object_bbox = [893, 305, 222, 545]
    object_mask = _bbox_area_object_mask(object_bbox, object_area)
    depth = np.ones(object_mask.shape, dtype=np.float32) * 0.5
    tube_mask = _write_bbox_area_mask(
        output_root / "semantic_sam/first_g140_masks/first_g140_mask_000.png",
        object_bbox,
        97916,
    )
    candidates_path = _write_candidates(
        output_root,
        [
            _candidate(
                "first_g140_mask_000",
                tube_mask,
                97916,
                object_bbox,
                0.26965787796360635,
                0.9999820455504386,
            ),
        ],
        workpiece_type="cover_plate",
    )

    result = select_weld_focus_masks(
        candidates_path=candidates_path,
        output_root=output_root,
        weld_focus=["tube"],
        object_mask=object_mask,
        depth=depth,
    )

    assert result.accepted is True
    selected = json.loads(result.selected_parts_path.read_text(encoding="utf-8"))
    assert selected["focused_parts"] == {"tube": "first_g140_mask_000"}
    diagnostics = json.loads(
        (output_root / "auto_part_selection.json").read_text(encoding="utf-8")
    )
    candidate = diagnostics["focused_parts"]["tube"]["candidates"][0]
    assert "area_ratio_above_max" not in candidate["hard_rejection_reasons"]
