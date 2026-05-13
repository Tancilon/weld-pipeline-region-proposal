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


def _validate_square_tube_v0(data: dict[str, Any]) -> None:
    if data["schema_version"] != "catspec.v0":
        raise CatSpecError(f"unsupported schema_version: {data['schema_version']}")
    if data["category"] != "square_tube":
        raise CatSpecError("CatSpec v0 loader currently accepts category square_tube only")

    parts = _require_non_empty_list(data, "parts")
    welds = _require_non_empty_list(data, "welds")

    part = parts[0]
    if part.get("id") != "tube_body":
        raise CatSpecError("square_tube v0 requires parts[0].id == tube_body")
    if part.get("primitive") != "square_tube":
        raise CatSpecError("square_tube v0 requires parts[0].primitive == square_tube")

    weld = welds[0]
    locus = weld.get("locus")
    if not isinstance(locus, dict):
        raise CatSpecError("welds[0].locus must be an object")
    if locus.get("type") != "closed_rounded_rect":
        raise CatSpecError("square_tube v0 requires welds[0].locus.type == closed_rounded_rect")
    params = locus.get("params")
    if not isinstance(params, dict):
        raise CatSpecError("welds[0].locus.params must be an object")
    for key in (
        "plane_axis",
        "plane_side",
        "profile_axes",
        "profile_quantile",
        "corner_radius_source",
        "sample_points_per_segment",
    ):
        if key not in params:
            raise CatSpecError(f"missing required field: welds[0].locus.params.{key}")

    weld_meta = weld.get("weld_meta")
    if not isinstance(weld_meta, dict):
        raise CatSpecError("welds[0].weld_meta must be an object")
    for key in ("weld_type_prior", "torch_constraints", "is_load_bearing", "confidence"):
        if key not in weld_meta:
            raise CatSpecError(f"missing required field: welds[0].weld_meta.{key}")


def load_catspec(path: str | Path) -> dict[str, Any]:
    """Load and validate a CatSpec YAML file."""
    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise CatSpecError("CatSpec root must be a YAML object")
    for required_path in REQUIRED_PATHS:
        _get_nested(data, required_path)
    _validate_square_tube_v0(data)
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
