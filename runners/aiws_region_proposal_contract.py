from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from components.aiws_pipeline_contracts import validate_region_proposal
from components.workpiece_priors import WorkpiecePriorRegistry


def _to_float_list(values: Any) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def _pose_from_rt(rotation: Any, translation: Any) -> list[list[float]]:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return pose.tolist()


def genpose_result_to_part_payload(
    registry: WorkpiecePriorRegistry,
    category: str,
    part_name: str,
    mask_path: str | Path,
    genpose_result: dict[str, Any],
    sigma: float = 0.05,
) -> dict[str, Any]:
    raw_size_m = _to_float_list(genpose_result["size"])
    if len(raw_size_m) != 3:
        raise ValueError("GenPose2 size must contain 3 values")
    raw_size_mm = [value * 1000.0 for value in raw_size_m]
    match = registry.match_part_size(category, part_name, raw_size_mm, sigma=sigma)
    payload = {
        "mask_path": str(mask_path),
        "raw_size_xyz_m": raw_size_m,
        "raw_size_xyz_mm": raw_size_mm,
        "size_source": str(genpose_result.get("size_source", "scale_net")),
        **match,
    }
    if "R" in genpose_result and "t" in genpose_result:
        payload["coarse_pose_cam_4x4"] = _pose_from_rt(
            genpose_result["R"], genpose_result["t"]
        )
        payload["coarse_pose_source"] = "genpose2"
    return payload


def write_region_proposal(
    output_dir: str | Path,
    sample_id: str,
    workpiece_type: str,
    camera_path: str | Path,
    rgb_path: str | Path,
    depth_path: str | Path,
    object_mask_path: str | Path,
    focused_parts: dict[str, dict[str, Any]],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "sample_id": sample_id,
        "workpiece_type": workpiece_type,
        "camera_path": str(camera_path),
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "object_mask_path": str(object_mask_path),
        "focused_parts": focused_parts,
    }
    validate_region_proposal(payload)
    output_path = output_dir / "region_proposal.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path
