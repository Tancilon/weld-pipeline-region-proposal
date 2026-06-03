from __future__ import annotations

from adapter.types import CoarseLocalizationResult


def visual_coarse_localize(*_args, **_kwargs) -> CoarseLocalizationResult:
    return CoarseLocalizationResult(
        status="failed",
        workpiece_type=None,
        class_confidence=None,
        matched_size_xyz_mm=None,
        size_match_confidence=None,
        part_masks={},
        region_proposal_path=None,
        diagnostics={"error": "visual_coarse_localize is not implemented"},
    )
