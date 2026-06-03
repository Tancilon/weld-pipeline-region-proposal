from __future__ import annotations

from typing import Any


class ContractError(RuntimeError):
    pass


def _require_mapping(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be a mapping")
    return payload


def _require_key(payload: dict[str, Any], key: str, label: str) -> Any:
    if key not in payload:
        raise ContractError(f"{label} missing required key: {key}")
    return payload[key]


def _require_xyz(values: Any, label: str) -> list[float]:
    if not isinstance(values, list) or len(values) != 3:
        raise ContractError(f"{label} must be a 3-value list")
    result = [float(value) for value in values]
    if any(value <= 0 for value in result):
        raise ContractError(f"{label} values must be positive")
    return result


def ensure_pose_matrix(matrix: Any, label: str) -> list[list[float]]:
    if not isinstance(matrix, list) or len(matrix) != 4:
        raise ContractError(f"{label} must be a 4x4 matrix")
    result: list[list[float]] = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != 4:
            raise ContractError(f"{label} must be a 4x4 matrix")
        result.append([float(value) for value in row])
    return result


def validate_region_proposal(payload: Any) -> None:
    payload = _require_mapping(payload, "region_proposal")
    for key in (
        "schema_version",
        "sample_id",
        "workpiece_type",
        "focused_parts",
        "camera_path",
        "rgb_path",
        "depth_path",
        "object_mask_path",
    ):
        _require_key(payload, key, "region_proposal")
    focused_parts = _require_mapping(payload["focused_parts"], "focused_parts")
    if not focused_parts:
        raise ContractError("focused_parts must not be empty")
    for part_name, part_payload in focused_parts.items():
        part_payload = _require_mapping(part_payload, f"focused_parts.{part_name}")
        for key in (
            "mask_path",
            "raw_size_xyz_m",
            "raw_size_xyz_mm",
            "size_source",
            "matched_size_xyz_mm",
            "size_match_error",
            "match_confidence",
        ):
            _require_key(part_payload, key, f"focused_parts.{part_name}")
        _require_xyz(part_payload["raw_size_xyz_m"], f"{part_name}.raw_size_xyz_m")
        _require_xyz(part_payload["raw_size_xyz_mm"], f"{part_name}.raw_size_xyz_mm")
        _require_xyz(
            part_payload["matched_size_xyz_mm"], f"{part_name}.matched_size_xyz_mm"
        )
        if "coarse_pose_cam_4x4" in part_payload:
            ensure_pose_matrix(
                part_payload["coarse_pose_cam_4x4"],
                f"{part_name}.coarse_pose_cam_4x4",
            )


def validate_alignment_result(payload: Any) -> None:
    payload = _require_mapping(payload, "alignment_result")
    for key in ("schema_version", "sample_id", "workpiece_type", "focused_parts", "weld_result"):
        _require_key(payload, key, "alignment_result")
    focused_parts = _require_mapping(payload["focused_parts"], "focused_parts")
    if not focused_parts:
        raise ContractError("focused_parts must not be empty")
    for part_name, part_payload in focused_parts.items():
        part_payload = _require_mapping(part_payload, f"focused_parts.{part_name}")
        status = _require_key(part_payload, "status", f"focused_parts.{part_name}")
        if status == "aligned":
            for key in (
                "pose_cam_4x4",
                "cad_template",
                "scaled_cad_path",
                "matched_size_xyz_mm",
                "pose_source",
            ):
                _require_key(part_payload, key, f"focused_parts.{part_name}")
            _require_xyz(
                part_payload["matched_size_xyz_mm"],
                f"{part_name}.matched_size_xyz_mm",
            )
            ensure_pose_matrix(part_payload["pose_cam_4x4"], f"{part_name}.pose_cam_4x4")
            if part_payload["pose_source"] != "foundationpose":
                raise ContractError(f"{part_name}.pose_source must be foundationpose")
        elif status == "failed":
            _require_key(part_payload, "error", f"focused_parts.{part_name}")
        else:
            raise ContractError(f"{part_name}.status must be aligned or failed")
    weld_result = _require_mapping(payload["weld_result"], "weld_result")
    weld_status = weld_result.get("status")
    if weld_status == "not_implemented":
        return
    if weld_status == "extracted":
        _require_key(weld_result, "weld_json_path", "weld_result")
        return
    if weld_status == "failed":
        _require_key(weld_result, "error", "weld_result")
        return
    raise ContractError("weld_result.status must be not_implemented, extracted, or failed")
