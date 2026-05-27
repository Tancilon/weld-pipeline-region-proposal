from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REGION_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REGION_ROOT.parent
if str(REGION_ROOT) not in sys.path:
    sys.path.insert(0, str(REGION_ROOT))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from runners.aiws_part_selection import (
    PartSelectionError,
    parse_part_mask_overrides,
    resolve_selected_part_masks,
)


def _write_candidates(path: Path, mask_path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_id": "0034",
                "workpiece_type": "square_tube",
                "semantic_sam": {},
                "candidates": [
                    {
                        "mask_id": "0034_mask_012",
                        "mask_path": str(mask_path),
                        "area_px": 100,
                        "bbox_xywh": [0, 0, 10, 10],
                        "predicted_iou": 0.9,
                        "stability_score": 0.8,
                        "depth_valid_pixels": 90,
                        "depth_valid_ratio": 0.9,
                        "object_overlap_ratio": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_parse_part_mask_overrides_rejects_missing_equals():
    with pytest.raises(PartSelectionError, match="part=mask"):
        parse_part_mask_overrides(["tube"])


def test_resolve_selected_part_masks_prefers_cli_override(tmp_path):
    json_mask = tmp_path / "semantic_sam/0034_masks/0034_mask_012.png"
    cli_mask = tmp_path / "manual/tube.png"
    json_mask.parent.mkdir(parents=True)
    cli_mask.parent.mkdir(parents=True)
    json_mask.write_bytes(b"mask")
    cli_mask.write_bytes(b"mask")
    candidates_path = _write_candidates(tmp_path / "mask_candidates.json", json_mask)
    selected_path = tmp_path / "selected_parts.json"
    selected_path.write_text(
        json.dumps({"focused_parts": {"tube": "0034_mask_012"}}),
        encoding="utf-8",
    )

    result = resolve_selected_part_masks(
        candidates_path=candidates_path,
        selected_parts_path=selected_path,
        cli_overrides={"tube": str(cli_mask)},
        weld_focus=["tube"],
        output_root=tmp_path,
    )

    assert result == {"tube": cli_mask.resolve()}


def test_resolve_selected_part_masks_accepts_candidate_id(tmp_path):
    mask_path = tmp_path / "semantic_sam/0034_masks/0034_mask_012.png"
    mask_path.parent.mkdir(parents=True)
    mask_path.write_bytes(b"mask")
    candidates_path = _write_candidates(tmp_path / "mask_candidates.json", mask_path)
    selected_path = tmp_path / "selected_parts.json"
    selected_path.write_text(
        json.dumps({"focused_parts": {"tube": "0034_mask_012"}}),
        encoding="utf-8",
    )

    result = resolve_selected_part_masks(
        candidates_path=candidates_path,
        selected_parts_path=selected_path,
        cli_overrides={},
        weld_focus=["tube"],
        output_root=tmp_path,
    )

    assert result == {"tube": mask_path.resolve()}


def test_resolve_selected_part_masks_raises_when_focus_missing(tmp_path):
    candidates_path = tmp_path / "mask_candidates.json"
    candidates_path.write_text(
        json.dumps({"schema_version": 1, "candidates": []}),
        encoding="utf-8",
    )

    with pytest.raises(PartSelectionError, match="tube"):
        resolve_selected_part_masks(
            candidates_path=candidates_path,
            selected_parts_path=tmp_path / "selected_parts.json",
            cli_overrides={},
            weld_focus=["tube"],
            output_root=tmp_path,
        )
