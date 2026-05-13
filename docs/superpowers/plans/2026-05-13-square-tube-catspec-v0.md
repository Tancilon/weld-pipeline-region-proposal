# Square Tube CatSpec v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a prompt-first `square_tube` CatSpec v0, generate a closed rounded-rectangle weld locus from the spec and workpiece mesh, and validate it against the existing weld OBJ.

**Architecture:** Add a small `catspec/` package with three boundaries: schema loading/validation, square-tube locus generation, and static validation/reporting. Keep the existing `weld/strategies/square_tube.py` as the reference extractor for `_weld.obj`; the new CatSpec path does not feed the reference weld centerline back into generation.

**Tech Stack:** Python 3.13, PyYAML, numpy, trimesh, matplotlib for overlay output, pytest.

**Spec:** `docs/superpowers/specs/2026-05-13-square-tube-catspec-v0-design.md`

**Branch:** `aiws5.3`

---

## Final File Structure

```
catspec/
├── __init__.py                # Public package exports
├── locus.py                   # Square-tube profile estimation, locus sampling, metrics
├── schema.py                  # YAML loading, required-field checks, asset path resolution
└── validation.py              # square_tube static validation and overlay/report writers

specs/
└── categories/
    └── square_tube.yaml       # Prompt-first CatSpec v0

scripts/
└── validate_catspec_square_tube.py

tests/
├── catspec/
│   ├── test_locus.py
│   ├── test_schema.py
│   └── test_validation.py
└── scripts/
    └── test_validate_catspec_square_tube.py
```

---

## Task 1: Add CatSpec YAML and Schema Loader

**Goal:** Add the first `square_tube` YAML and a strict but small loader that catches missing required fields before validation runs.

**Files:**
- Create: `specs/categories/square_tube.yaml`
- Create: `catspec/__init__.py`
- Create: `catspec/schema.py`
- Create: `tests/catspec/test_schema.py`

- [ ] **Step 1: Write failing schema tests**

Create `tests/catspec/test_schema.py` with:

```python
from pathlib import Path

import pytest

from catspec.schema import CatSpecError, load_catspec, resolve_asset_path


VALID_YAML = """\
schema_version: catspec.v0
category: square_tube
units: meter
provenance:
  source_mesh: ../datasets/obj_share_models/square_tube/square_tube.obj
  source_weld_mesh: ../datasets/obj_share_models/square_tube/square_tube_weld.obj
  size_source: ../datasets/workpiece_info.json
parts:
  - id: tube_body
    primitive: square_tube
    role: primary_structure
    frame: canonical_bbox
    size_priors:
      bbox_xyz:
        - [0.220000, 0.209000, 0.220000]
        - [0.220000, 0.408000, 0.220000]
    symmetry: z2_or_c4
    prompt_tags:
      - hollow_profile
      - rectilinear_tube
      - rounded_corners
welds:
  - id: outer_perimeter
    parts:
      - tube_body
    locus:
      type: closed_rounded_rect
      source: analytic_from_profile
      frame: weld_local_pca
      params:
        plane_axis: y
        plane_side: min_dense
        profile_axes: [x, z]
        profile_quantile: 5.0
        corner_radius_source: estimate_from_workpiece_mesh
        sample_points_per_segment: 32
    weld_meta:
      weld_type_prior: fillet
      torch_constraints: default_single_pass
      is_load_bearing: true
      confidence: medium
"""


def test_load_catspec_accepts_square_tube_v0(tmp_path):
    spec_path = tmp_path / "square_tube.yaml"
    spec_path.write_text(VALID_YAML, encoding="utf-8")

    spec = load_catspec(spec_path)

    assert spec["schema_version"] == "catspec.v0"
    assert spec["category"] == "square_tube"
    assert spec["parts"][0]["id"] == "tube_body"
    assert spec["welds"][0]["locus"]["type"] == "closed_rounded_rect"


def test_load_catspec_rejects_missing_required_field(tmp_path):
    spec_path = tmp_path / "bad.yaml"
    spec_path.write_text("schema_version: catspec.v0\ncategory: square_tube\n", encoding="utf-8")

    with pytest.raises(CatSpecError, match="missing required field: units"):
        load_catspec(spec_path)


def test_load_catspec_rejects_wrong_schema_version(tmp_path):
    spec_path = tmp_path / "bad.yaml"
    spec_path.write_text(VALID_YAML.replace("catspec.v0", "catspec.v9"), encoding="utf-8")

    with pytest.raises(CatSpecError, match="unsupported schema_version"):
        load_catspec(spec_path)


def test_resolve_asset_path_prefers_existing_absolute_path(tmp_path):
    asset = tmp_path / "mesh.obj"
    asset.write_text("o mesh\n", encoding="utf-8")

    resolved = resolve_asset_path(str(asset), spec_path=tmp_path / "spec.yaml")

    assert resolved == asset


def test_resolve_asset_path_handles_repo_parent_paths(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    sibling = tmp_path / "datasets" / "obj_share_models"
    sibling.mkdir(parents=True)
    asset = sibling / "mesh.obj"
    asset.write_text("o mesh\n", encoding="utf-8")
    spec_dir = repo_root / "specs" / "categories"
    spec_dir.mkdir(parents=True)
    monkeypatch.chdir(repo_root)

    resolved = resolve_asset_path("../datasets/obj_share_models/mesh.obj", spec_path=spec_dir / "spec.yaml")

    assert resolved == asset.resolve()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/catspec/test_schema.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'catspec'`.

- [ ] **Step 3: Create `specs/categories/square_tube.yaml`**

Create `specs/categories/square_tube.yaml` with:

```yaml
schema_version: catspec.v0
category: square_tube
units: meter

provenance:
  source_mesh: ../datasets/obj_share_models/square_tube/square_tube.obj
  source_weld_mesh: ../datasets/obj_share_models/square_tube/square_tube_weld.obj
  size_source: ../datasets/workpiece_info.json

parts:
  - id: tube_body
    primitive: square_tube
    role: primary_structure
    frame: canonical_bbox
    size_priors:
      bbox_xyz:
        - [0.220000, 0.209000, 0.220000]
        - [0.220000, 0.408000, 0.220000]
    symmetry: z2_or_c4
    prompt_tags:
      - hollow_profile
      - rectilinear_tube
      - rounded_corners

welds:
  - id: outer_perimeter
    parts:
      - tube_body
    locus:
      type: closed_rounded_rect
      source: analytic_from_profile
      frame: weld_local_pca
      params:
        plane_axis: y
        plane_side: min_dense
        profile_axes: [x, z]
        profile_quantile: 5.0
        corner_radius_source: estimate_from_workpiece_mesh
        sample_points_per_segment: 32
    weld_meta:
      weld_type_prior: fillet
      torch_constraints: default_single_pass
      is_load_bearing: true
      confidence: medium
```

- [ ] **Step 4: Create `catspec/__init__.py`**

Create `catspec/__init__.py` with:

```python
"""Structured category specifications for part-aware pose and weld proposals."""

from catspec.schema import CatSpecError, load_catspec, resolve_asset_path

__all__ = ["CatSpecError", "load_catspec", "resolve_asset_path"]
```

- [ ] **Step 5: Create `catspec/schema.py`**

Create `catspec/schema.py` with:

```python
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
    for key in ("plane_axis", "plane_side", "profile_axes", "profile_quantile", "sample_points_per_segment"):
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

    spec_path = Path(spec_path)
    search_roots = (
        Path.cwd(),
        spec_path.parent,
        Path.cwd().parent,
    )
    for root in search_roots:
        candidate = (root / raw).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / raw).resolve()
```

- [ ] **Step 6: Run schema tests to verify they pass**

Run: `python -m pytest tests/catspec/test_schema.py -v`

Expected: PASS, 5 tests.

- [ ] **Step 7: Commit Task 1**

```bash
git add catspec/__init__.py catspec/schema.py specs/categories/square_tube.yaml tests/catspec/test_schema.py
git commit -m "feat: add square tube catspec schema"
```

---

## Task 2: Add Square-Tube Locus Generation and Metrics

**Goal:** Estimate the square-tube weld profile from the workpiece mesh without reading `_weld.obj`, generate four line segments plus four arcs, and expose sampling/metric helpers for validation.

**Files:**
- Create: `catspec/locus.py`
- Create: `tests/catspec/test_locus.py`

- [ ] **Step 1: Write failing locus tests**

Create `tests/catspec/test_locus.py` with:

```python
import numpy as np
import trimesh

from catspec.locus import (
    build_closed_rounded_rect_locus,
    closed_path_gap,
    estimate_square_tube_profile,
    sample_locus_3d,
    symmetric_hausdorff,
    symmetric_rmse,
)


def _make_square_tube_like_mesh():
    y0 = -0.461722
    y1 = 0.5
    outer = 0.228986
    radius = 0.043545
    n = 36
    pts = []

    def add_round_rect(y):
        centers = [
            (outer - radius, outer - radius, 0.0, np.pi / 2),
            (outer - radius, -outer + radius, -np.pi / 2, 0.0),
            (-outer + radius, -outer + radius, np.pi, 3 * np.pi / 2),
            (-outer + radius, outer - radius, np.pi / 2, np.pi),
        ]
        for cx, cz, a0, a1 in centers:
            for angle in np.linspace(a0, a1, n):
                pts.append([cx + radius * np.cos(angle), y, cz + radius * np.sin(angle)])

    add_round_rect(y0)
    add_round_rect(y1)
    # Sparse bbox outliers mimic the existing square_tube.obj normalization vertices.
    pts.extend([
        [-0.5, y0, -0.5],
        [-0.5, y0, 0.5],
        [0.5, y0, -0.5],
        [0.5, y0, 0.5],
    ])
    return trimesh.Trimesh(vertices=np.array(pts), faces=np.empty((0, 3), dtype=int), process=False)


def test_estimate_square_tube_profile_uses_dense_end_face_not_sparse_bbox():
    mesh = _make_square_tube_like_mesh()

    profile = estimate_square_tube_profile(
        mesh,
        plane_axis="y",
        plane_side="min_dense",
        profile_axes=("x", "z"),
        profile_quantile=5.0,
    )

    assert abs(profile.plane_value - (-0.461722)) < 1e-6
    assert profile.min_uv[0] > -0.24
    assert profile.max_uv[0] < 0.24
    assert profile.min_uv[1] > -0.24
    assert profile.max_uv[1] < 0.24
    assert 0.025 < profile.corner_radius < 0.070


def test_build_closed_rounded_rect_locus_has_expected_topology():
    mesh = _make_square_tube_like_mesh()
    profile = estimate_square_tube_profile(mesh, "y", "min_dense", ("x", "z"), 5.0)

    locus = build_closed_rounded_rect_locus(profile)

    types = [seg["type"] for seg in locus["segments"]]
    assert types == ["line", "arc", "line", "arc", "line", "arc", "line", "arc"]
    assert locus["closed"] is True
    assert closed_path_gap(locus) < 1e-9


def test_sample_locus_3d_returns_closed_samples_in_original_axes():
    mesh = _make_square_tube_like_mesh()
    profile = estimate_square_tube_profile(mesh, "y", "min_dense", ("x", "z"), 5.0)
    locus = build_closed_rounded_rect_locus(profile)

    samples = sample_locus_3d(locus, points_per_segment=8)

    assert samples.shape[1] == 3
    assert np.allclose(samples[:, 1], profile.plane_value)
    assert np.linalg.norm(samples[0] - samples[-1]) < 1e-9


def test_symmetric_metrics_are_zero_for_identical_paths():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])

    assert symmetric_rmse(pts, pts) == 0.0
    assert symmetric_hausdorff(pts, pts) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/catspec/test_locus.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'catspec.locus'`.

- [ ] **Step 3: Create `catspec/locus.py`**

Create `catspec/locus.py` with:

```python
"""CatSpec weld-locus geometry helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import trimesh


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class SquareTubeProfile:
    plane_axis: str
    plane_value: float
    profile_axes: tuple[str, str]
    min_uv: tuple[float, float]
    max_uv: tuple[float, float]
    corner_radius: float


def _dense_plane_value(coords: np.ndarray, side: str) -> float:
    rounded = np.round(coords, 6)
    values, counts = np.unique(rounded, return_counts=True)
    if len(values) == 0:
        raise ValueError("mesh has no vertices")
    dense_threshold = max(3, int(np.ceil(counts.max() * 0.2)))
    dense_values = values[counts >= dense_threshold]
    if len(dense_values) == 0:
        dense_values = values
    if side == "min_dense":
        return float(dense_values.min())
    if side == "max_dense":
        return float(dense_values.max())
    raise ValueError(f"unsupported plane_side: {side}")


def _estimate_corner_radius(points_uv: np.ndarray, min_uv: np.ndarray, max_uv: np.ndarray) -> float:
    corners = np.array([
        [max_uv[0], max_uv[1]],
        [min_uv[0], max_uv[1]],
        [min_uv[0], min_uv[1]],
        [max_uv[0], min_uv[1]],
    ])
    radii = []
    for corner in corners:
        nearest = float(np.min(np.linalg.norm(points_uv - corner, axis=1)))
        radii.append(nearest / (np.sqrt(2.0) - 1.0))
    radius = float(np.median(radii))
    max_radius = 0.5 * float(np.min(max_uv - min_uv)) - 1e-9
    return min(max(radius, 1e-9), max_radius)


def estimate_square_tube_profile(
    mesh: trimesh.Trimesh,
    plane_axis: str,
    plane_side: str,
    profile_axes: tuple[str, str],
    profile_quantile: float,
) -> SquareTubeProfile:
    """Estimate the square-tube weld profile from the dense end face of the workpiece mesh."""
    if plane_axis not in AXIS_INDEX:
        raise ValueError(f"unsupported plane_axis: {plane_axis}")
    if len(profile_axes) != 2 or any(axis not in AXIS_INDEX for axis in profile_axes):
        raise ValueError(f"unsupported profile_axes: {profile_axes}")
    if not (0.0 <= profile_quantile < 25.0):
        raise ValueError("profile_quantile must be in [0, 25)")

    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.size == 0:
        raise ValueError("mesh has no vertices")

    plane_idx = AXIS_INDEX[plane_axis]
    plane_value = _dense_plane_value(vertices[:, plane_idx], plane_side)
    plane_mask = np.isclose(vertices[:, plane_idx], plane_value, atol=1e-5)
    plane_vertices = vertices[plane_mask]
    if len(plane_vertices) < 8:
        raise ValueError(f"not enough vertices on dense plane {plane_axis}={plane_value}")

    u_idx = AXIS_INDEX[profile_axes[0]]
    v_idx = AXIS_INDEX[profile_axes[1]]
    points_uv = plane_vertices[:, [u_idx, v_idx]]
    lo = np.percentile(points_uv, profile_quantile, axis=0)
    hi = np.percentile(points_uv, 100.0 - profile_quantile, axis=0)
    keep = np.all((points_uv >= lo) & (points_uv <= hi), axis=1)
    robust_points = points_uv[keep]
    if len(robust_points) < 8:
        raise ValueError("not enough vertices after profile quantile clipping")

    min_uv = robust_points.min(axis=0)
    max_uv = robust_points.max(axis=0)
    radius = _estimate_corner_radius(robust_points, min_uv, max_uv)

    return SquareTubeProfile(
        plane_axis=plane_axis,
        plane_value=plane_value,
        profile_axes=profile_axes,
        min_uv=(float(min_uv[0]), float(min_uv[1])),
        max_uv=(float(max_uv[0]), float(max_uv[1])),
        corner_radius=radius,
    )


def _arc_mid(center: np.ndarray, radius: float, angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return center + radius * np.array([np.cos(angle), np.sin(angle)])


def _point_to_3d(point_uv: Iterable[float], profile: SquareTubeProfile) -> list[float]:
    coords = [0.0, 0.0, 0.0]
    coords[AXIS_INDEX[profile.plane_axis]] = profile.plane_value
    coords[AXIS_INDEX[profile.profile_axes[0]]] = float(point_uv[0])
    coords[AXIS_INDEX[profile.profile_axes[1]]] = float(point_uv[1])
    return coords


def build_closed_rounded_rect_locus(profile: SquareTubeProfile) -> dict:
    """Build an analytic closed rounded-rectangle locus from an estimated square-tube profile."""
    u_min, v_min = np.array(profile.min_uv, dtype=float)
    u_max, v_max = np.array(profile.max_uv, dtype=float)
    r = float(profile.corner_radius)

    top_l = np.array([u_min + r, v_max])
    top_r = np.array([u_max - r, v_max])
    right_t = np.array([u_max, v_max - r])
    right_b = np.array([u_max, v_min + r])
    bot_r = np.array([u_max - r, v_min])
    bot_l = np.array([u_min + r, v_min])
    left_b = np.array([u_min, v_min + r])
    left_t = np.array([u_min, v_max - r])

    tr_c = np.array([u_max - r, v_max - r])
    br_c = np.array([u_max - r, v_min + r])
    bl_c = np.array([u_min + r, v_min + r])
    tl_c = np.array([u_min + r, v_max - r])

    segments_2d = [
        ("line", [top_l, top_r]),
        ("arc", [top_r, _arc_mid(tr_c, r, 45.0), right_t]),
        ("line", [right_t, right_b]),
        ("arc", [right_b, _arc_mid(br_c, r, -45.0), bot_r]),
        ("line", [bot_r, bot_l]),
        ("arc", [bot_l, _arc_mid(bl_c, r, -135.0), left_b]),
        ("line", [left_b, left_t]),
        ("arc", [left_t, _arc_mid(tl_c, r, 135.0), top_l]),
    ]

    segments = []
    for seg_type, points in segments_2d:
        segments.append({
            "type": seg_type,
            "points_2d": [[float(x), float(y)] for x, y in points],
            "points_3d": [_point_to_3d(point, profile) for point in points],
        })

    return {
        "closed": True,
        "profile": asdict(profile),
        "segments": segments,
    }


def _interpolate_arc_2d(p0: np.ndarray, pm: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    ax, ay = p0
    bx, by = pm
    cx, cy = p1
    det = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(det) < 1e-12:
        return np.column_stack([np.linspace(p0[0], p1[0], n), np.linspace(p0[1], p1[1], n)])
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / det
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / det

    a0 = np.arctan2(p0[1] - uy, p0[0] - ux)
    am = np.arctan2(pm[1] - uy, pm[0] - ux)
    a1 = np.arctan2(p1[1] - uy, p1[0] - ux)
    mid_ccw = (am - a0) % (2 * np.pi)
    end_ccw = (a1 - a0) % (2 * np.pi)
    if mid_ccw < end_ccw:
        angles = np.linspace(a0, a0 + end_ccw, n)
    else:
        angles = np.linspace(a0, a0 + end_ccw - 2 * np.pi, n)
    radius = float(np.linalg.norm(p0 - np.array([ux, uy])))
    return np.column_stack([ux + radius * np.cos(angles), uy + radius * np.sin(angles)])


def sample_segments_2d(segments: list[dict], points_per_segment: int) -> np.ndarray:
    samples = []
    for segment in segments:
        points = np.asarray(segment["points_2d"], dtype=float)
        if segment["type"] == "line":
            interp = np.column_stack([
                np.linspace(points[0, 0], points[-1, 0], points_per_segment, endpoint=False),
                np.linspace(points[0, 1], points[-1, 1], points_per_segment, endpoint=False),
            ])
        elif segment["type"] == "arc":
            interp = _interpolate_arc_2d(points[0], points[1], points[2], points_per_segment)
            interp = interp[:-1]
        else:
            raise ValueError(f"unsupported segment type: {segment['type']}")
        samples.append(interp)
    result = np.vstack(samples)
    return np.vstack([result, result[0]])


def sample_locus_3d(locus: dict, points_per_segment: int) -> np.ndarray:
    profile_data = locus["profile"]
    profile = SquareTubeProfile(
        plane_axis=profile_data["plane_axis"],
        plane_value=profile_data["plane_value"],
        profile_axes=tuple(profile_data["profile_axes"]),
        min_uv=tuple(profile_data["min_uv"]),
        max_uv=tuple(profile_data["max_uv"]),
        corner_radius=profile_data["corner_radius"],
    )
    points_2d = sample_segments_2d(locus["segments"], points_per_segment)
    return np.asarray([_point_to_3d(point, profile) for point in points_2d], dtype=float)


def closed_path_gap(locus: dict) -> float:
    first = np.asarray(locus["segments"][0]["points_3d"][0], dtype=float)
    last = np.asarray(locus["segments"][-1]["points_3d"][-1], dtype=float)
    return float(np.linalg.norm(first - last))


def _nearest_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))


def symmetric_rmse(a: np.ndarray, b: np.ndarray) -> float:
    d_ab = _nearest_distances(a, b)
    d_ba = _nearest_distances(b, a)
    return float(np.sqrt(np.mean(np.concatenate([d_ab, d_ba]) ** 2)))


def symmetric_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    d_ab = _nearest_distances(a, b)
    d_ba = _nearest_distances(b, a)
    return float(max(np.max(d_ab), np.max(d_ba)))
```

- [ ] **Step 4: Run locus tests to verify they pass**

Run: `python -m pytest tests/catspec/test_locus.py -v`

Expected: PASS, 4 tests.

- [ ] **Step 5: Commit Task 2**

```bash
git add catspec/locus.py tests/catspec/test_locus.py
git commit -m "feat: generate square tube catspec locus"
```

---

## Task 3: Add Static Validation Module

**Goal:** Compare the CatSpec-derived locus with the existing `square_tube_weld.obj` reference path, write a JSON report, and create an overlay PNG.

**Files:**
- Create: `catspec/validation.py`
- Create: `tests/catspec/test_validation.py`

- [ ] **Step 1: Write failing validation tests**

Create `tests/catspec/test_validation.py` with:

```python
import json
from pathlib import Path

import numpy as np
import trimesh

from catspec.validation import validate_square_tube


def _make_workpiece_mesh(path: Path):
    y0 = -0.461722
    y1 = 0.5
    outer = 0.228986
    radius = 0.043545
    pts = []
    for y in (y0, y1):
        for angle in np.linspace(0.0, 2 * np.pi, 96, endpoint=False):
            c = np.cos(angle)
            s = np.sin(angle)
            u = np.sign(c) * (outer - radius) + radius * c
            v = np.sign(s) * (outer - radius) + radius * s
            pts.append([u, y, v])
    pts.extend([
        [-0.5, y0, -0.5],
        [-0.5, y0, 0.5],
        [0.5, y0, -0.5],
        [0.5, y0, 0.5],
    ])
    mesh = trimesh.Trimesh(vertices=np.asarray(pts), faces=np.empty((0, 3), dtype=int), process=False)
    mesh.export(path)


def _make_weld_mesh(path: Path):
    outer = 0.228986
    radius = 0.043545
    centerline = []
    for angle in np.linspace(0.0, 2 * np.pi, 128, endpoint=False):
        c = np.cos(angle)
        s = np.sin(angle)
        x = np.sign(c) * (outer - radius) + radius * c
        z = np.sign(s) * (outer - radius) + radius * s
        centerline.append([x, -0.461722, z])
    verts = []
    for x, y, z in centerline:
        verts.append([x, y, z - 0.001])
        verts.append([x, y, z + 0.001])
    faces = []
    n = len(centerline)
    for i in range(n):
        j = (i + 1) % n
        faces.append([2 * i, 2 * j, 2 * i + 1])
        faces.append([2 * j, 2 * j + 1, 2 * i + 1])
    mesh = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=False)
    mesh.export(path)


def _write_spec(path: Path, workpiece: Path, weld: Path):
    path.write_text(f"""\
schema_version: catspec.v0
category: square_tube
units: meter
provenance:
  source_mesh: {workpiece}
  source_weld_mesh: {weld}
  size_source: ignored.json
parts:
  - id: tube_body
    primitive: square_tube
    role: primary_structure
    frame: canonical_bbox
    size_priors:
      bbox_xyz:
        - [0.220000, 0.209000, 0.220000]
        - [0.220000, 0.408000, 0.220000]
    symmetry: z2_or_c4
    prompt_tags: [hollow_profile, rectilinear_tube, rounded_corners]
welds:
  - id: outer_perimeter
    parts: [tube_body]
    locus:
      type: closed_rounded_rect
      source: analytic_from_profile
      frame: weld_local_pca
      params:
        plane_axis: y
        plane_side: min_dense
        profile_axes: [x, z]
        profile_quantile: 5.0
        corner_radius_source: estimate_from_workpiece_mesh
        sample_points_per_segment: 16
    weld_meta:
      weld_type_prior: fillet
      torch_constraints: default_single_pass
      is_load_bearing: true
      confidence: medium
""", encoding="utf-8")


def test_validate_square_tube_writes_report_and_overlay(tmp_path):
    workpiece = tmp_path / "square_tube.obj"
    weld = tmp_path / "square_tube_weld.obj"
    spec = tmp_path / "square_tube.yaml"
    output_dir = tmp_path / "out"
    _make_workpiece_mesh(workpiece)
    _make_weld_mesh(weld)
    _write_spec(spec, workpiece, weld)

    report = validate_square_tube(spec, output_dir)

    assert report["category"] == "square_tube"
    assert report["topology_match"] is True
    assert report["generated"]["segment_types"] == ["line", "arc", "line", "arc", "line", "arc", "line", "arc"]
    assert report["metrics"]["closed_path_gap"] < 1e-9
    assert Path(report["report_path"]).exists()
    assert Path(report["overlay_path"]).exists()
    saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert saved["category"] == "square_tube"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/catspec/test_validation.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'catspec.validation'`.

- [ ] **Step 3: Create `catspec/validation.py`**

Create `catspec/validation.py` with:

```python
"""Static validation for CatSpec files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from catspec.locus import (
    build_closed_rounded_rect_locus,
    closed_path_gap,
    estimate_square_tube_profile,
    sample_locus_3d,
    sample_segments_2d,
    symmetric_hausdorff,
    symmetric_rmse,
)
from catspec.schema import load_catspec, resolve_asset_path
from weld.core import back_project, load_weld_mesh
from weld.strategies.square_tube import SquareTubeStrategy


EXPECTED_ROUNDED_RECT_TYPES = ["line", "arc", "line", "arc", "line", "arc", "line", "arc"]


def _sample_reference_path_3d(path: dict[str, Any], points_per_segment: int) -> np.ndarray:
    segments = []
    for segment in path["fitted"]:
        segments.append({
            "type": segment["type"],
            "points_2d": [[float(x), float(y)] for x, y in np.asarray(segment["points_2d"], dtype=float)],
        })
    samples_2d = sample_segments_2d(segments, points_per_segment)
    return back_project(samples_2d, path["plane"])


def _project_to_plane(points_3d: np.ndarray, plane: dict[str, Any]) -> np.ndarray:
    diffs = points_3d - plane["origin"]
    return np.column_stack([diffs @ plane["u"], diffs @ plane["v"]])


def _write_overlay(reference_3d: np.ndarray, generated_3d: np.ndarray, plane: dict[str, Any], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_2d = _project_to_plane(reference_3d, plane)
    gen_2d = _project_to_plane(generated_3d, plane)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(ref_2d[:, 0], ref_2d[:, 1], color="tab:blue", linewidth=2, label="weld OBJ reference")
    ax.plot(gen_2d[:, 0], gen_2d[:, 1], color="tab:red", linewidth=1.5, linestyle="--", label="CatSpec locus")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    ax.set_title("square_tube CatSpec v0 validation")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _generate_square_tube_locus(spec: dict[str, Any], workpiece_path: Path) -> dict[str, Any]:
    params = spec["welds"][0]["locus"]["params"]
    mesh = trimesh.load(workpiece_path, force="mesh", process=False)
    profile = estimate_square_tube_profile(
        mesh,
        plane_axis=params["plane_axis"],
        plane_side=params["plane_side"],
        profile_axes=tuple(params["profile_axes"]),
        profile_quantile=float(params["profile_quantile"]),
    )
    return build_closed_rounded_rect_locus(profile)


def validate_square_tube(spec_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Validate square_tube CatSpec v0 against the existing weld OBJ reference."""
    spec_path = Path(spec_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = load_catspec(spec_path)
    workpiece_path = resolve_asset_path(spec["provenance"]["source_mesh"], spec_path)
    weld_path = resolve_asset_path(spec["provenance"]["source_weld_mesh"], spec_path)

    generated_locus = _generate_square_tube_locus(spec, workpiece_path)
    points_per_segment = int(spec["welds"][0]["locus"]["params"]["sample_points_per_segment"])
    generated_3d = sample_locus_3d(generated_locus, points_per_segment)

    weld_mesh = load_weld_mesh(str(weld_path))
    reference_paths = SquareTubeStrategy().process(weld_mesh)
    if len(reference_paths) != 1:
        raise ValueError(f"square_tube reference expected 1 weld path, got {len(reference_paths)}")
    reference_path = reference_paths[0]
    reference_types = [segment["type"] for segment in reference_path["fitted"]]
    generated_types = [segment["type"] for segment in generated_locus["segments"]]
    topology_match = reference_types == EXPECTED_ROUNDED_RECT_TYPES and generated_types == EXPECTED_ROUNDED_RECT_TYPES

    reference_3d = _sample_reference_path_3d(reference_path, points_per_segment)
    metrics = {
        "closed_path_gap": closed_path_gap(generated_locus),
        "centerline_rmse": symmetric_rmse(generated_3d, reference_3d),
        "hausdorff": symmetric_hausdorff(generated_3d, reference_3d),
    }

    report_path = output_dir / "square_tube_catspec_validation.json"
    overlay_path = output_dir / "square_tube_catspec_overlay.png"

    report = {
        "category": spec["category"],
        "spec_path": str(spec_path),
        "workpiece_path": str(workpiece_path),
        "weld_path": str(weld_path),
        "topology_match": bool(topology_match),
        "reference": {
            "segment_types": reference_types,
        },
        "generated": {
            "segment_types": generated_types,
            "profile": generated_locus["profile"],
        },
        "metrics": metrics,
        "report_path": str(report_path),
        "overlay_path": str(overlay_path),
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_overlay(reference_3d, generated_3d, reference_path["plane"], overlay_path)
    return report
```

- [ ] **Step 4: Run validation tests to verify they pass**

Run: `python -m pytest tests/catspec/test_validation.py -v`

Expected: PASS, 1 test.

- [ ] **Step 5: Commit Task 3**

```bash
git add catspec/validation.py tests/catspec/test_validation.py
git commit -m "feat: validate square tube catspec locus"
```

---

## Task 4: Add CLI Script

**Goal:** Provide a simple local command to run the v0 validation on the real `square_tube` assets and write artifacts under `results/catspec/square_tube/`.

**Files:**
- Create: `scripts/validate_catspec_square_tube.py`
- Create: `tests/scripts/test_validate_catspec_square_tube.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/scripts/test_validate_catspec_square_tube.py` with:

```python
import subprocess
import sys


def test_validate_catspec_square_tube_help():
    result = subprocess.run(
        [sys.executable, "scripts/validate_catspec_square_tube.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--spec" in result.stdout
    assert "--output-dir" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/test_validate_catspec_square_tube.py -v`

Expected: FAIL because `scripts/validate_catspec_square_tube.py` does not exist.

- [ ] **Step 3: Create `scripts/validate_catspec_square_tube.py`**

Create `scripts/validate_catspec_square_tube.py` with:

```python
#!/usr/bin/env python
"""Validate square_tube CatSpec v0 against the existing weld OBJ."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        default="specs/categories/square_tube.yaml",
        help="Path to square_tube CatSpec YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/catspec/square_tube",
        help="Directory for JSON report and overlay PNG.",
    )
    args = parser.parse_args()

    from catspec.validation import validate_square_tube

    report = validate_square_tube(args.spec, args.output_dir)
    print(json.dumps({
        "category": report["category"],
        "topology_match": report["topology_match"],
        "metrics": report["metrics"],
        "report_path": report["report_path"],
        "overlay_path": report["overlay_path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI tests to verify they pass**

Run: `python -m pytest tests/scripts/test_validate_catspec_square_tube.py -v`

Expected: PASS, 1 test.

- [ ] **Step 5: Commit Task 4**

```bash
git add scripts/validate_catspec_square_tube.py tests/scripts/test_validate_catspec_square_tube.py
git commit -m "feat: add square tube catspec validation cli"
```

---

## Task 5: Verify on Real Assets and Record Artifacts

**Goal:** Run the focused test suite and the real `square_tube` validation command. Confirm the report and overlay exist.

**Files:**
- Read: `specs/categories/square_tube.yaml`
- Generate: `results/catspec/square_tube/square_tube_catspec_validation.json`
- Generate: `results/catspec/square_tube/square_tube_catspec_overlay.png`

- [ ] **Step 1: Run all CatSpec tests**

Run:

```bash
python -m pytest tests/catspec tests/scripts/test_validate_catspec_square_tube.py -v
```

Expected: PASS, 11 tests.

- [ ] **Step 2: Run existing square-tube strategy regression**

Run:

```bash
python -m pytest tests/weld/test_strategies.py::test_square_tube_strategy_fits_rounded_rect -v
```

Expected: PASS, 1 test.

- [ ] **Step 3: Run real-asset validation**

Run:

```bash
python scripts/validate_catspec_square_tube.py --output-dir results/catspec/square_tube
```

Expected: command exits 0 and prints JSON containing `category`, `topology_match`, `metrics`, `report_path`, and `overlay_path`.

- [ ] **Step 4: Inspect generated report fields**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("results/catspec/square_tube/square_tube_catspec_validation.json")
data = json.loads(p.read_text(encoding="utf-8"))
print(data["category"])
print(data["topology_match"])
print(sorted(data["metrics"]))
print(Path(data["overlay_path"]).exists())
PY
```

Expected output:

```text
square_tube
True
['centerline_rmse', 'closed_path_gap', 'hausdorff']
True
```

- [ ] **Step 5: Check git status before committing**

Run:

```bash
git status --short
```

Expected: tracked code/spec/test changes are staged or unstaged; generated `results/catspec/square_tube/` artifacts may be untracked and should stay out of the commit unless the repository already tracks result artifacts.

- [ ] **Step 6: Commit verification-related source changes only**

Run:

```bash
git add catspec specs/categories scripts/validate_catspec_square_tube.py tests/catspec tests/scripts/test_validate_catspec_square_tube.py
git commit -m "test: verify square tube catspec v0"
```

Expected: a commit is created. If there are no source changes left because prior tasks already committed them, skip this commit and record the test command outputs in the final implementation summary.

---

## Self-Review Checklist

- Spec coverage: Tasks 1-4 implement YAML, parser, locus generation, static validation, JSON report, overlay PNG, and CLI. Task 5 covers real-asset verification.
- GT leakage guard: `_weld.obj` is only loaded in `catspec.validation` after generated locus creation and is used only for reference metrics/overlay.
- Type consistency: `load_catspec`, `resolve_asset_path`, `estimate_square_tube_profile`, `build_closed_rounded_rect_locus`, `sample_locus_3d`, `symmetric_rmse`, `symmetric_hausdorff`, and `validate_square_tube` are defined before use.
- Test scope: tests are CPU-only and avoid training, GPU, and model code.
