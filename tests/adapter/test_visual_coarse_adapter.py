from __future__ import annotations

import json
import inspect
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REGION_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REGION_ROOT.parent
for path in (WORKSPACE, REGION_ROOT):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))
for name in [
    module_name
    for module_name in sys.modules
    if module_name == "adapter"
    or module_name.startswith("adapter.")
    or module_name == "components"
    or module_name.startswith("components.")
]:
    sys.modules.pop(name, None)

import adapter.visual_coarse as visual_coarse
from adapter.visual_coarse import (
    VisualCoarseRuntimeConfig,
    visual_coarse_localize_with_config,
)


def _write_rgb_depth_mask(tmp_path: Path) -> tuple[Path, Path, Path]:
    rgb_path = tmp_path / "0055_color.png"
    depth_path = tmp_path / "0055_depth.png"
    mask_path = tmp_path / "object_mask.png"
    Image.fromarray(np.zeros((16, 20, 3), dtype=np.uint8)).save(rgb_path)
    Image.fromarray(np.ones((16, 20), dtype=np.uint16) * 500).save(depth_path)
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[4:14, 5:18] = 255
    Image.fromarray(mask, mode="L").save(mask_path)
    return rgb_path, depth_path, mask_path


def test_visual_coarse_uses_region_local_depth_compat():
    expected_path = REGION_ROOT / "utils/depth_compat.py"

    assert Path(inspect.getfile(visual_coarse.load_depth)).resolve() == expected_path


def test_visual_coarse_cover_plate_returns_minimal_result(tmp_path, monkeypatch):
    rgb_path, depth_path, object_mask_path = _write_rgb_depth_mask(tmp_path)
    output_dir = tmp_path / "out"

    class FakeEstimator:
        def __init__(self, **_kwargs):
            pass

        def estimate_part(self, **_kwargs):
            return {
                "R": np.eye(3, dtype=np.float32),
                "t": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "size": np.array([0.1, 0.05759, 0.1], dtype=np.float32),
                "size_source": "scale_net",
            }

    monkeypatch.setattr(visual_coarse, "GenPosePartEstimator", FakeEstimator)

    result = visual_coarse_localize_with_config(
        rgb_path=str(rgb_path),
        depth_path=str(depth_path),
        output_dir=str(output_dir),
        workpiece_type="cover_plate",
        object_mask_path=str(object_mask_path),
        verbose=False,
        config=VisualCoarseRuntimeConfig(
            workpiece_info_path=str(WORKSPACE / "workpiece_priors/workpiece_info.yaml"),
            camera_path=str(WORKSPACE / "workpiece_priors/camera.json"),
            skip_semantic_sam=False,
            force=True,
        ),
    )

    payload = result.to_dict()
    assert payload["status"] == "ok"
    assert payload["workpiece_type"] == "cover_plate"
    assert payload["class_confidence"] == 1.0
    assert payload["matched_size_xyz_mm"] == [100.0, 57.59, 100.0]
    assert payload["size_match_confidence"] is not None
    assert payload["part_masks"]["tube"].endswith("part_masks/tube.png")
    assert payload["region_proposal_path"].endswith("region_proposal.json")
    assert Path(payload["region_proposal_path"]).exists()
    assert "object_mask_path" not in payload


def test_visual_coarse_returns_segmentation_required_when_auto_selection_rejects(
    tmp_path, monkeypatch
):
    rgb_path, depth_path, object_mask_path = _write_rgb_depth_mask(tmp_path)
    output_dir = tmp_path / "out"

    def fake_ensure_semantic_sam(*_args, **_kwargs):
        metadata_dir = output_dir / "semantic_sam"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / "0055_metadata.json"
        metadata_path.write_text(json.dumps({"masks": []}), encoding="utf-8")
        return metadata_path

    class FakeSelection:
        accepted = False
        reason = "low_confidence_auto_selection"
        diagnostics_path = output_dir / "auto_part_selection.json"
        selected_parts_path = output_dir / "selected_parts.auto.json"

    monkeypatch.setattr(visual_coarse, "_ensure_semantic_sam", fake_ensure_semantic_sam)
    def fake_build_mask_candidates(**_kwargs):
        candidates_path = output_dir / "mask_candidates.json"
        candidates_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sample_id": "0055",
                    "workpiece_type": "square_tube",
                    "semantic_sam": {},
                    "candidates": [],
                }
            ),
            encoding="utf-8",
        )
        return candidates_path

    monkeypatch.setattr(visual_coarse, "build_mask_candidates", fake_build_mask_candidates)
    monkeypatch.setattr(
        visual_coarse,
        "select_weld_focus_masks",
        lambda **_kwargs: FakeSelection(),
    )

    result = visual_coarse_localize_with_config(
        rgb_path=str(rgb_path),
        depth_path=str(depth_path),
        output_dir=str(output_dir),
        workpiece_type="square_tube",
        object_mask_path=str(object_mask_path),
        verbose=True,
        config=VisualCoarseRuntimeConfig(
            workpiece_info_path=str(WORKSPACE / "workpiece_priors/workpiece_info.yaml"),
            camera_path=str(WORKSPACE / "workpiece_priors/camera.json"),
            semantic_sam_python=sys.executable,
            skip_semantic_sam=False,
            force=True,
        ),
    )

    assert result.status == "segmentation_required"
    assert result.region_proposal_path is None
    assert result.diagnostics["reason"] == "low_confidence_auto_selection"
