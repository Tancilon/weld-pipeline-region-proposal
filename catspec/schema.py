"""CatSpec YAML loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class CatSpecError(ValueError):
    """Raised when a CatSpec file is malformed."""


REQUIRED_PATHS = (
    ("schema_version",),
    ("category",),
    ("units",),
    ("provenance", "source_mesh"),
    ("provenance", "source_weld_mesh"),
    ("parts",),
    ("welds",),
)

SUPPORTED_CATEGORIES = {"square_tube", "channel_steel", "H_beam", "bellmouth"}
OPEN_PROFILE_CATEGORIES = {"channel_steel", "H_beam"}
PARALLEL_LINE_CATEGORIES = {"bellmouth"}

LOCUS_REQUIRED_PARAMS = {
    "closed_rounded_rect": (
        "plane_axis",
        "plane_side",
        "profile_axes",
        "profile_quantile",
        "corner_radius_source",
        "sample_points_per_segment",
    ),
    "open_line_arc_line_arc_line": (
        "plane_axis",
        "plane_values",
        "profile_axes",
        "path_count",
        "sample_points_per_segment",
    ),
    "parallel_open_lines": (
        "plane_axis",
        "plane_side",
        "profile_axes",
        "line_axis",
        "offset_axis",
        "offset_values",
        "path_count",
        "sample_points_per_segment",
    ),
}


def _get_nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise CatSpecError(f"missing required field: {'.'.join(path)}")
        cur = cur[key]
    return cur


def _require_non_empty_list(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise CatSpecError(f"{key} must be a non-empty list")
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise CatSpecError(f"{key}[{idx}] must be an object")
    return value


def _require_object(value: Any, field_path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatSpecError(f"{field_path} must be an object")
    return value


def _validate_profile_axes(value: Any, field_path: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise CatSpecError(f"{field_path} must be a two-item list")
    if len(set(value)) != 2 or any(axis not in {"x", "y", "z"} for axis in value):
        raise CatSpecError(f"{field_path} must contain two distinct axes from x, y, z")


def _validate_catspec_v0(data: dict[str, Any]) -> None:
    if data["schema_version"] != "catspec.v0":
        raise CatSpecError(f"unsupported schema_version: {data['schema_version']}")
    if data["category"] not in SUPPORTED_CATEGORIES:
        raise CatSpecError(f"unsupported category: {data['category']}")

    parts = _require_non_empty_list(data, "parts")
    welds = _require_non_empty_list(data, "welds")

    for idx, part in enumerate(parts):
        for key in ("id", "primitive"):
            if key not in part:
                raise CatSpecError(f"missing required field: parts[{idx}].{key}")

    for idx, weld in enumerate(welds):
        locus = _require_object(weld.get("locus"), f"welds[{idx}].locus")
        locus_type = locus.get("type")
        if locus_type not in LOCUS_REQUIRED_PARAMS:
            raise CatSpecError(f"unsupported locus type: {locus_type}")
        if data["category"] == "square_tube" and locus_type != "closed_rounded_rect":
            raise CatSpecError("square_tube v0 requires welds[0].locus.type == closed_rounded_rect")
        if data["category"] in OPEN_PROFILE_CATEGORIES and locus_type != "open_line_arc_line_arc_line":
            raise CatSpecError(f"{data['category']} v0.1 requires open_line_arc_line_arc_line locus")
        if data["category"] in PARALLEL_LINE_CATEGORIES and locus_type != "parallel_open_lines":
            raise CatSpecError(f"{data['category']} v0.2 requires parallel_open_lines locus")

        params = _require_object(locus.get("params"), f"welds[{idx}].locus.params")
        for key in LOCUS_REQUIRED_PARAMS[locus_type]:
            if key not in params:
                raise CatSpecError(f"missing required field: welds[{idx}].locus.params.{key}")
        _validate_profile_axes(params["profile_axes"], f"welds[{idx}].locus.params.profile_axes")
        if params["plane_axis"] not in {"x", "y", "z"}:
            raise CatSpecError(f"welds[{idx}].locus.params.plane_axis must be x, y, or z")
        if params["plane_axis"] in params["profile_axes"]:
            raise CatSpecError(f"welds[{idx}].locus.params plane_axis must differ from profile_axes")
        if int(params["sample_points_per_segment"]) < 2:
            raise CatSpecError(f"welds[{idx}].locus.params.sample_points_per_segment must be at least 2")
        if locus_type == "open_line_arc_line_arc_line":
            if params["plane_values"] != "dense_internal":
                raise CatSpecError(f"welds[{idx}].locus.params.plane_values must be dense_internal")
            if int(params["path_count"]) < 1:
                raise CatSpecError(f"welds[{idx}].locus.params.path_count must be at least 1")
        if locus_type == "parallel_open_lines":
            if params["line_axis"] not in params["profile_axes"]:
                raise CatSpecError(f"welds[{idx}].locus.params.line_axis must be in profile_axes")
            if params["offset_axis"] not in params["profile_axes"]:
                raise CatSpecError(f"welds[{idx}].locus.params.offset_axis must be in profile_axes")
            if params["line_axis"] == params["offset_axis"]:
                raise CatSpecError(f"welds[{idx}].locus.params line_axis must differ from offset_axis")
            if params["offset_values"] != "dense_internal":
                raise CatSpecError(f"welds[{idx}].locus.params.offset_values must be dense_internal")
            if int(params["path_count"]) < 1:
                raise CatSpecError(f"welds[{idx}].locus.params.path_count must be at least 1")

        weld_meta = _require_object(weld.get("weld_meta"), f"welds[{idx}].weld_meta")
        for key in ("weld_type_prior", "torch_constraints", "is_load_bearing", "confidence"):
            if key not in weld_meta:
                raise CatSpecError(f"missing required field: welds[{idx}].weld_meta.{key}")


def load_catspec(path: str | Path) -> dict[str, Any]:
    """Load and validate a CatSpec YAML file."""
    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise CatSpecError("CatSpec root must be a YAML object")
    for required_path in REQUIRED_PATHS:
        _get_nested(data, required_path)
    _validate_catspec_v0(data)
    return data


def resolve_asset_path(raw_path: str | Path, spec_path: str | Path) -> Path:
    """Resolve absolute, repo-relative, spec-relative, and repo-parent-relative asset paths."""
    raw = Path(raw_path)
    if raw.is_absolute():
        return raw

    spec_path = Path(spec_path).resolve()
    spec_dir = spec_path.parent
    search_roots = [spec_dir]
    for ancestor in spec_dir.parents:
        search_roots.append(ancestor)
        search_roots.append(ancestor.parent)

    seen: set[Path] = set()
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = (root / raw).resolve()
        if candidate.exists():
            return candidate
    return (spec_dir / raw).resolve()
