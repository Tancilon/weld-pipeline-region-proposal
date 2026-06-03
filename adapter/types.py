from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CoarseLocalizationResult:
    status: str
    workpiece_type: str | None
    class_confidence: float | None
    matched_size_xyz_mm: list[float] | None
    size_match_confidence: float | None
    part_masks: dict[str, str]
    region_proposal_path: str | None
    object_mask_path: str | None = None
    raw_size_xyz_mm: list[float] | None = None
    coarse_pose_cam_4x4: list[list[float]] | None = None
    selected_parts_path: str | None = None
    diagnostics: dict[str, Any] | None = None

    def to_dict(self, *, verbose: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "workpiece_type": self.workpiece_type,
            "class_confidence": self.class_confidence,
            "matched_size_xyz_mm": self.matched_size_xyz_mm,
            "size_match_confidence": self.size_match_confidence,
            "part_masks": dict(self.part_masks),
            "region_proposal_path": self.region_proposal_path,
        }
        if not verbose:
            return payload
        payload.update(
            {
                "object_mask_path": self.object_mask_path,
                "raw_size_xyz_mm": self.raw_size_xyz_mm,
                "coarse_pose_cam_4x4": self.coarse_pose_cam_4x4,
                "selected_parts_path": self.selected_parts_path,
                "diagnostics": self.diagnostics,
            }
        )
        return payload
