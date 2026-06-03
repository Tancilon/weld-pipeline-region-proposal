from __future__ import annotations

from pathlib import Path
import sys

REGION_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REGION_ROOT.parent
if str(REGION_ROOT) not in sys.path:
    sys.path.insert(0, str(REGION_ROOT))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
for name in [
    module_name
    for module_name in sys.modules
    if module_name == "adapter" or module_name.startswith("adapter.")
]:
    sys.modules.pop(name, None)

from adapter.types import CoarseLocalizationResult


def test_coarse_result_to_dict_hides_verbose_fields_by_default():
    result = CoarseLocalizationResult(
        status="ok",
        workpiece_type="cover_plate",
        class_confidence=0.97,
        matched_size_xyz_mm=[100.0, 57.59, 100.0],
        size_match_confidence=0.86,
        part_masks={"tube": "out/part_masks/tube.png"},
        region_proposal_path="out/region_proposal.json",
        object_mask_path="out/object_mask.png",
        raw_size_xyz_mm=[99.0, 58.0, 101.0],
        coarse_pose_cam_4x4=[
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ],
        selected_parts_path="out/selected_parts.auto.json",
        diagnostics={"selector": "accepted"},
    )

    payload = result.to_dict()

    assert payload == {
        "status": "ok",
        "workpiece_type": "cover_plate",
        "class_confidence": 0.97,
        "matched_size_xyz_mm": [100.0, 57.59, 100.0],
        "size_match_confidence": 0.86,
        "part_masks": {"tube": "out/part_masks/tube.png"},
        "region_proposal_path": "out/region_proposal.json",
    }


def test_coarse_result_to_dict_can_include_verbose_fields():
    result = CoarseLocalizationResult(
        status="ok",
        workpiece_type="square_tube",
        class_confidence=0.92,
        matched_size_xyz_mm=[101.0, 201.0, 101.0],
        size_match_confidence=0.74,
        part_masks={"tube": "out/part_masks/tube.png"},
        region_proposal_path="out/region_proposal.json",
        object_mask_path="out/object_mask.png",
        raw_size_xyz_mm=[103.0, 199.0, 100.0],
        coarse_pose_cam_4x4=[
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ],
        selected_parts_path="out/selected_parts.auto.json",
        diagnostics={"auto_selection_reason": "accepted"},
    )

    payload = result.to_dict(verbose=True)

    assert payload["object_mask_path"] == "out/object_mask.png"
    assert payload["raw_size_xyz_mm"] == [103.0, 199.0, 100.0]
    assert payload["coarse_pose_cam_4x4"][0] == [1.0, 0.0, 0.0, 0.1]
    assert payload["selected_parts_path"] == "out/selected_parts.auto.json"
    assert payload["diagnostics"] == {"auto_selection_reason": "accepted"}
