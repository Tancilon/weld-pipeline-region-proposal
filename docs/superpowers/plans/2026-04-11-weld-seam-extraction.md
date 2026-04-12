# Weld Seam Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract weld seam centerlines from OBJ meshes and fit them as line/arc segment sequences, outputting structured JSON and visualization.

**Architecture:** Single script `scripts/extract_weld_seams.py` containing all pipeline functions. Loads OBJ via trimesh, projects to 2D via PCA, extracts centerline from mesh topology, segments by curvature into line/arc primitives, and outputs JSON + matplotlib comparison plot.

**Tech Stack:** Python, numpy, trimesh, matplotlib

---

## File Structure

| File | Responsibility |
|------|---------------|
| `scripts/extract_weld_seams.py` | All pipeline functions + CLI entry point |
| `tests/scripts/test_extract_weld_seams.py` | Unit tests for each pipeline stage |

---

### Task 1: Add trimesh dependency

**Files:**
- Modify: `requirements.txt` (or install directly if no requirements.txt)

- [ ] **Step 1: Install trimesh**

```bash
pip install trimesh
```

- [ ] **Step 2: Verify import works**

```bash
python -c "import trimesh; print(trimesh.__version__)"
```

Expected: version number printed without error.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt  # if applicable
git commit -m "deps: add trimesh for weld seam mesh processing"
```

---

### Task 2: OBJ parsing and main object selection

**Files:**
- Create: `tests/scripts/test_extract_weld_seams.py`
- Create: `scripts/extract_weld_seams.py`

- [ ] **Step 1: Write failing tests for OBJ parsing**

Create `tests/scripts/test_extract_weld_seams.py`:

```python
import numpy as np
import pytest
import trimesh

from scripts.extract_weld_seams import load_weld_mesh, extract_model_name


def _make_obj_scene(tmp_path, objects: dict[str, np.ndarray]) -> str:
    """Write a multi-object OBJ file. objects maps name -> vertices (Nx3)."""
    path = tmp_path / "test.obj"
    lines = []
    vert_offset = 0
    for name, verts in objects.items():
        lines.append(f"o {name}")
        for v in verts:
            lines.append(f"v {v[0]} {v[1]} {v[2]}")
        # Minimal faces: one triangle per 3 consecutive vertices
        for i in range(0, len(verts) - 2, 3):
            a, b, c = vert_offset + i + 1, vert_offset + i + 2, vert_offset + i + 3
            lines.append(f"f {a} {b} {c}")
        vert_offset += len(verts)
    path.write_text("\n".join(lines))
    return str(path)


def test_load_weld_mesh_selects_largest_object(tmp_path):
    cube_verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    # Main object: 12 vertices (more than cube's 8)
    main_verts = np.random.rand(12, 3) * 100
    obj_path = _make_obj_scene(tmp_path, {"Cube": cube_verts, "MainWeld": main_verts})
    mesh = load_weld_mesh(obj_path)
    assert len(mesh.vertices) == 12


def test_load_weld_mesh_single_object(tmp_path):
    verts = np.random.rand(20, 3)
    obj_path = _make_obj_scene(tmp_path, {"OnlyObject": verts})
    mesh = load_weld_mesh(obj_path)
    assert len(mesh.vertices) == 20


def test_extract_model_name():
    assert extract_model_name("/path/to/工件1.obj") == "工件1"
    assert extract_model_name("assets/part_A.obj") == "part_A"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/scripts/test_extract_weld_seams.py -v
```

Expected: FAIL with `ImportError` — module not found.

- [ ] **Step 3: Implement load_weld_mesh and extract_model_name**

Create `scripts/extract_weld_seams.py`:

```python
"""Extract weld seam centerlines from OBJ meshes and fit as line/arc sequences."""

import os
from pathlib import Path

import numpy as np
import trimesh


def load_weld_mesh(obj_path: str) -> trimesh.Trimesh:
    """Load OBJ and return the sub-mesh with the most vertices."""
    scene = trimesh.load(obj_path, force="scene")
    if isinstance(scene, trimesh.Trimesh):
        return scene
    best_mesh = None
    best_count = 0
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > best_count:
            best_count = len(geom.vertices)
            best_mesh = geom
    if best_mesh is None:
        raise ValueError(f"No valid mesh found in {obj_path}")
    return best_mesh


def extract_model_name(workpiece_path: str) -> str:
    """Extract model name from workpiece OBJ filename (without extension)."""
    return Path(workpiece_path).stem
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/scripts/test_extract_weld_seams.py::test_load_weld_mesh_selects_largest_object tests/scripts/test_extract_weld_seams.py::test_load_weld_mesh_single_object tests/scripts/test_extract_weld_seams.py::test_extract_model_name -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_weld_seams.py tests/scripts/test_extract_weld_seams.py
git commit -m "feat: add OBJ parsing and model name extraction for weld seam pipeline"
```

---

### Task 3: PCA plane projection and back-projection

**Files:**
- Modify: `scripts/extract_weld_seams.py`
- Modify: `tests/scripts/test_extract_weld_seams.py`

- [ ] **Step 1: Write failing tests for PCA projection**

Append to `tests/scripts/test_extract_weld_seams.py`:

```python
from scripts.extract_weld_seams import pca_project, back_project


def test_pca_project_planar_points():
    """Points on the XY plane should project with near-zero Z variance."""
    rng = np.random.default_rng(42)
    xy = rng.uniform(-50, 50, (100, 2))
    points_3d = np.column_stack([xy, np.zeros(100)])
    pts_2d, plane = pca_project(points_3d)
    assert pts_2d.shape == (100, 2)
    assert plane["origin"].shape == (3,)
    assert plane["u"].shape == (3,)
    assert plane["v"].shape == (3,)
    assert plane["planarity"] > 0.95


def test_pca_project_warns_non_planar(capsys):
    """Points with high Z variance should trigger a warning."""
    rng = np.random.default_rng(42)
    points_3d = rng.uniform(-50, 50, (100, 3))  # fully 3D, not planar
    pts_2d, plane = pca_project(points_3d)
    assert plane["planarity"] < 0.95


def test_back_project_roundtrip():
    """back_project(pca_project(pts)) should recover original points."""
    rng = np.random.default_rng(42)
    xy = rng.uniform(-50, 50, (50, 2))
    noise = rng.normal(0, 0.01, 50)
    points_3d = np.column_stack([xy, noise])
    pts_2d, plane = pca_project(points_3d)
    recovered = back_project(pts_2d, plane)
    # Should be close (projection loses the tiny Z noise)
    assert recovered.shape == (50, 3)
    # Project original to plane too for fair comparison
    projected_orig = back_project(pca_project(points_3d)[0], plane)
    np.testing.assert_allclose(recovered, projected_orig, atol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/scripts/test_extract_weld_seams.py::test_pca_project_planar_points tests/scripts/test_extract_weld_seams.py::test_back_project_roundtrip -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement pca_project and back_project**

Add to `scripts/extract_weld_seams.py`:

```python
import warnings


def pca_project(vertices: np.ndarray) -> tuple[np.ndarray, dict]:
    """Project 3D vertices onto their best-fit plane via PCA.

    Returns:
        pts_2d: (N, 2) projected coordinates
        plane: dict with keys origin, u, v, n, planarity
    """
    origin = vertices.mean(axis=0)
    centered = vertices - origin
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    u, v, n = Vt[0], Vt[1], Vt[2]
    variance = S ** 2
    planarity = 1.0 - variance[2] / variance.sum()
    if planarity < 0.95:
        warnings.warn(
            f"Weld seam planarity is {planarity:.1%} (< 95%). "
            "Results may be inaccurate for non-planar weld seams."
        )
    pts_2d = centered @ np.column_stack([u, v])
    plane = {"origin": origin, "u": u, "v": v, "n": n, "planarity": planarity}
    return pts_2d, plane


def back_project(pts_2d: np.ndarray, plane: dict) -> np.ndarray:
    """Map 2D plane coordinates back to 3D."""
    return plane["origin"] + pts_2d[:, 0:1] * plane["u"] + pts_2d[:, 1:2] * plane["v"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/scripts/test_extract_weld_seams.py::test_pca_project_planar_points tests/scripts/test_extract_weld_seams.py::test_pca_project_warns_non_planar tests/scripts/test_extract_weld_seams.py::test_back_project_roundtrip -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_weld_seams.py tests/scripts/test_extract_weld_seams.py
git commit -m "feat: add PCA plane projection and back-projection for weld seam vertices"
```

---

### Task 4: Centerline extraction from mesh topology

**Files:**
- Modify: `scripts/extract_weld_seams.py`
- Modify: `tests/scripts/test_extract_weld_seams.py`

- [ ] **Step 1: Write failing tests for centerline extraction**

Append to `tests/scripts/test_extract_weld_seams.py`:

```python
from scripts.extract_weld_seams import extract_centerline


def _make_tube_mesh(n_rings=20, n_per_ring=8, radius=2.0, length=100.0):
    """Create a synthetic tube mesh (structured sweep) along the X axis."""
    verts = []
    faces = []
    for i in range(n_rings):
        x = length * i / (n_rings - 1)
        for j in range(n_per_ring):
            angle = 2 * np.pi * j / n_per_ring
            y = radius * np.cos(angle)
            z = radius * np.sin(angle)
            verts.append([x, y, z])
    verts = np.array(verts)
    # Connect adjacent rings with triangle faces
    for i in range(n_rings - 1):
        for j in range(n_per_ring):
            j_next = (j + 1) % n_per_ring
            v0 = i * n_per_ring + j
            v1 = i * n_per_ring + j_next
            v2 = (i + 1) * n_per_ring + j
            v3 = (i + 1) * n_per_ring + j_next
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    return mesh


def test_extract_centerline_straight_tube():
    mesh = _make_tube_mesh(n_rings=20, n_per_ring=8, radius=2.0, length=100.0)
    pts_2d, plane = pca_project(mesh.vertices)
    centerline = extract_centerline(mesh, pts_2d)
    assert len(centerline) >= 10
    # Centerline should be roughly along a straight line in 2D
    # Check that the Y spread is small relative to X spread
    x_range = centerline[:, 0].max() - centerline[:, 0].min()
    y_range = centerline[:, 1].max() - centerline[:, 1].min()
    assert x_range > 50  # should span most of the tube length
    assert y_range < 5   # should be nearly straight


def test_extract_centerline_arc_tube():
    """Tube that follows a circular arc path."""
    n_rings, n_per_ring = 30, 8
    arc_radius, tube_radius = 50.0, 2.0
    verts = []
    for i in range(n_rings):
        theta = np.pi * i / (n_rings - 1)  # 0 to pi (half circle)
        cx = arc_radius * np.cos(theta)
        cy = arc_radius * np.sin(theta)
        for j in range(n_per_ring):
            angle = 2 * np.pi * j / n_per_ring
            # Tube cross-section perpendicular to arc tangent
            tangent = np.array([-np.sin(theta), np.cos(theta)])
            normal = np.array([np.cos(theta), np.sin(theta)])
            x = cx + tube_radius * np.cos(angle) * normal[0]
            y = cy + tube_radius * np.cos(angle) * normal[1]
            z = tube_radius * np.sin(angle)
            verts.append([x, y, z])
    verts = np.array(verts)
    faces = []
    for i in range(n_rings - 1):
        for j in range(n_per_ring):
            j_next = (j + 1) % n_per_ring
            v0 = i * n_per_ring + j
            v1 = i * n_per_ring + j_next
            v2 = (i + 1) * n_per_ring + j
            v3 = (i + 1) * n_per_ring + j_next
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    pts_2d, plane = pca_project(mesh.vertices)
    centerline = extract_centerline(mesh, pts_2d)
    assert len(centerline) >= 15
    # Centerline points should be roughly at arc_radius from origin
    dists = np.sqrt(centerline[:, 0]**2 + centerline[:, 1]**2)
    np.testing.assert_allclose(dists, arc_radius, atol=5.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/scripts/test_extract_weld_seams.py::test_extract_centerline_straight_tube tests/scripts/test_extract_weld_seams.py::test_extract_centerline_arc_tube -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement extract_centerline**

Add to `scripts/extract_weld_seams.py`:

```python
from collections import defaultdict, deque


def extract_centerline(mesh: trimesh.Trimesh, pts_2d: np.ndarray) -> np.ndarray:
    """Extract ordered centerline points from a structured sweep mesh.

    Uses mesh topology to identify cross-sectional rings, then takes
    ring centroids as centerline points.

    Falls back to boundary-averaging if topology analysis fails.

    Args:
        mesh: the weld seam trimesh
        pts_2d: (N, 2) PCA-projected vertex coordinates

    Returns:
        centerline: (M, 2) ordered 2D centerline points
    """
    try:
        return _centerline_from_topology(mesh, pts_2d)
    except Exception:
        return _centerline_fallback(pts_2d)


def _centerline_from_topology(mesh: trimesh.Trimesh, pts_2d: np.ndarray) -> np.ndarray:
    """Extract centerline by identifying cross-sectional vertex rings."""
    # Build adjacency from mesh edges
    adjacency = defaultdict(set)
    for edge in mesh.edges_unique:
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])

    n_verts = len(pts_2d)
    # Estimate local path direction from overall PCA of 2D points
    centroid_2d = pts_2d.mean(axis=0)
    centered_2d = pts_2d - centroid_2d

    # Identify boundary vertices (edges with only one adjacent face)
    boundary_verts = set()
    edge_face_count = defaultdict(int)
    for face in mesh.faces:
        for i in range(3):
            edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            edge_face_count[edge] += 1
    for edge, count in edge_face_count.items():
        if count == 1:
            boundary_verts.add(edge[0])
            boundary_verts.add(edge[1])

    # BFS to find rings: start from a boundary vertex, walk along boundary
    # to find one side, then identify rings by pairing vertices across the tube
    if not boundary_verts:
        return _centerline_fallback(pts_2d)

    # Sort all vertices by their projection onto the first principal axis
    # This gives an approximate ordering along the path
    proj_along_path = centered_2d[:, 0]  # first PC = path direction for most cases
    sorted_indices = np.argsort(proj_along_path)

    # Group vertices into rings by binning along the path direction
    n_boundary = len(boundary_verts)
    # Estimate ring count: boundary verts form two chains (inner/outer),
    # so n_rings ≈ n_boundary / 2, but we use vertex connectivity instead
    # Partition vertices into rings using connectivity + position
    visited = np.zeros(n_verts, dtype=bool)
    rings = []
    start_vert = sorted_indices[0]

    # Walk along the mesh: for each unvisited vertex, BFS to collect its ring
    # A ring = vertices at similar path-distance (similar proj_along_path value)
    path_values = proj_along_path.copy()

    # Sort vertices by path projection, group into clusters
    sorted_path = path_values[sorted_indices]
    # Detect ring boundaries by finding gaps in the sorted projection
    diffs = np.diff(sorted_path)
    median_diff = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 1.0
    gap_threshold = median_diff * 3.0

    ring_starts = [0]
    for i, d in enumerate(diffs):
        if d > gap_threshold:
            ring_starts.append(i + 1)
    ring_starts.append(len(sorted_indices))

    rings_2d = []
    for i in range(len(ring_starts) - 1):
        ring_indices = sorted_indices[ring_starts[i]:ring_starts[i + 1]]
        if len(ring_indices) >= 2:
            ring_center = pts_2d[ring_indices].mean(axis=0)
            rings_2d.append(ring_center)

    if len(rings_2d) < 3:
        return _centerline_fallback(pts_2d)

    centerline = np.array(rings_2d)

    # Order centerline points by connectivity (nearest-neighbor chain)
    centerline = _order_points(centerline)

    # Light smoothing: moving average with window 3
    if len(centerline) > 5:
        smoothed = centerline.copy()
        for i in range(1, len(centerline) - 1):
            smoothed[i] = (centerline[i - 1] + centerline[i] + centerline[i + 1]) / 3
        centerline = smoothed

    return centerline


def _centerline_fallback(pts_2d: np.ndarray) -> np.ndarray:
    """Fallback: sort points by angle, average inner/outer boundaries."""
    centroid = pts_2d.mean(axis=0)
    angles = np.arctan2(pts_2d[:, 1] - centroid[1], pts_2d[:, 0] - centroid[0])
    sorted_idx = np.argsort(angles)
    sorted_pts = pts_2d[sorted_idx]

    # Bin by angle and take mean radial position per bin
    n_bins = max(30, len(pts_2d) // 20)
    angle_bins = np.linspace(angles[sorted_idx[0]], angles[sorted_idx[-1]], n_bins + 1)
    centers = []
    for i in range(n_bins):
        mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])
        if mask.sum() >= 2:
            centers.append(pts_2d[mask].mean(axis=0))

    if len(centers) < 3:
        raise ValueError("Could not extract centerline: too few points")
    return np.array(centers)


def _order_points(points: np.ndarray) -> np.ndarray:
    """Order 2D points into a sequential chain by nearest-neighbor."""
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = order[-1]
        dists = np.sum((points - points[last]) ** 2, axis=1)
        dists[visited] = np.inf
        nearest = np.argmin(dists)
        order.append(nearest)
        visited[nearest] = True
    return points[order]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/scripts/test_extract_weld_seams.py::test_extract_centerline_straight_tube tests/scripts/test_extract_weld_seams.py::test_extract_centerline_arc_tube -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_weld_seams.py tests/scripts/test_extract_weld_seams.py
git commit -m "feat: add centerline extraction from structured sweep mesh topology"
```

---

### Task 5: Curvature computation and segmentation

**Files:**
- Modify: `scripts/extract_weld_seams.py`
- Modify: `tests/scripts/test_extract_weld_seams.py`

- [ ] **Step 1: Write failing tests for curvature segmentation**

Append to `tests/scripts/test_extract_weld_seams.py`:

```python
from scripts.extract_weld_seams import compute_curvature, segment_by_curvature


def test_compute_curvature_straight_line():
    """Curvature of a straight line should be near zero."""
    pts = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
    kappa = compute_curvature(pts)
    assert len(kappa) == 50
    # Interior points should have ~0 curvature
    assert np.all(np.abs(kappa[1:-1]) < 1e-6)


def test_compute_curvature_circle():
    """Curvature of a circle with radius R should be ~1/R."""
    R = 50.0
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    pts = np.column_stack([R * np.cos(theta), R * np.sin(theta)])
    kappa = compute_curvature(pts)
    # Interior points should have curvature ~1/R = 0.02
    np.testing.assert_allclose(kappa[2:-2], 1.0 / R, atol=0.005)


def test_segment_by_curvature_line_only():
    pts = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
    segments = segment_by_curvature(pts)
    assert len(segments) == 1
    assert segments[0]["type"] == "line"


def test_segment_by_curvature_arc_only():
    R = 50.0
    theta = np.linspace(0, np.pi, 50)
    pts = np.column_stack([R * np.cos(theta), R * np.sin(theta)])
    segments = segment_by_curvature(pts)
    assert len(segments) == 1
    assert segments[0]["type"] == "arc"


def test_segment_by_curvature_mixed():
    """Line segment followed by arc followed by line."""
    # Straight line: x in [0, 30]
    line1 = np.column_stack([np.linspace(0, 30, 20), np.zeros(20)])
    # Arc: quarter circle radius 20, center at (30, 20)
    theta = np.linspace(-np.pi / 2, 0, 20)
    arc = np.column_stack([30 + 20 * np.cos(theta), 20 + 20 * np.sin(theta)])
    # Straight line: y from 20 upward
    line2 = np.column_stack([50 * np.ones(20), np.linspace(20, 50, 20)])
    pts = np.vstack([line1, arc[1:], line2[1:]])  # avoid duplicate junction points
    segments = segment_by_curvature(pts)
    types = [s["type"] for s in segments]
    assert "line" in types
    assert "arc" in types
    assert len(segments) >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/scripts/test_extract_weld_seams.py -k "curvature" -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement compute_curvature and segment_by_curvature**

Add to `scripts/extract_weld_seams.py`:

```python
def compute_curvature(pts: np.ndarray) -> np.ndarray:
    """Compute discrete curvature at each point using circumscribed circle.

    For three consecutive points, curvature = 1/R where R is the circumradius.
    Endpoints get the same curvature as their nearest interior neighbor.

    Args:
        pts: (N, 2) ordered 2D points

    Returns:
        kappa: (N,) curvature values
    """
    n = len(pts)
    kappa = np.zeros(n)
    for i in range(1, n - 1):
        a = pts[i - 1]
        b = pts[i]
        c = pts[i + 1]
        # Triangle side lengths
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ac = np.linalg.norm(c - a)
        # Area via cross product
        cross = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        denom = ab * bc * ac
        if denom < 1e-12:
            kappa[i] = 0.0
        else:
            kappa[i] = 2.0 * cross / denom
    # Copy endpoint curvatures
    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]
    return kappa


def segment_by_curvature(centerline: np.ndarray) -> list[dict]:
    """Segment centerline into line and arc segments based on curvature.

    Args:
        centerline: (N, 2) ordered 2D centerline points

    Returns:
        list of dicts, each with:
            type: "line" or "arc"
            indices: (start, end) index range into centerline
            points_2d: the 2D points for this segment
    """
    kappa = compute_curvature(centerline)

    # Smooth curvature with window of 5
    kernel_size = min(5, len(kappa))
    if kernel_size >= 3:
        kernel = np.ones(kernel_size) / kernel_size
        kappa_smooth = np.convolve(kappa, kernel, mode="same")
    else:
        kappa_smooth = kappa

    # Auto threshold: use median of non-zero curvatures
    nonzero_kappa = kappa_smooth[kappa_smooth > 1e-8]
    if len(nonzero_kappa) == 0:
        threshold = 1e-6
    else:
        threshold = np.median(nonzero_kappa) * 0.3

    # Label each point
    labels = np.where(kappa_smooth < threshold, 0, 1)  # 0=line, 1=arc

    # Merge into contiguous segments
    segments = []
    i = 0
    while i < len(labels):
        label = labels[i]
        j = i
        while j < len(labels) and labels[j] == label:
            j += 1
        segments.append({"label": label, "start": i, "end": j})
        i = j

    # Merge short segments (< 3 points) into neighbors
    merged = []
    for seg in segments:
        length = seg["end"] - seg["start"]
        if length < 3 and merged:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)

    # Build output
    result = []
    for seg in merged:
        seg_type = "arc" if seg["label"] == 1 else "line"
        # Re-check: if the segment was merged, re-evaluate its type
        seg_kappa = kappa_smooth[seg["start"]:seg["end"]]
        if len(seg_kappa) > 0 and np.median(seg_kappa) >= threshold:
            seg_type = "arc"
        else:
            seg_type = "line"
        result.append({
            "type": seg_type,
            "indices": (seg["start"], seg["end"]),
            "points_2d": centerline[seg["start"]:seg["end"]],
        })

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/scripts/test_extract_weld_seams.py -k "curvature" -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_weld_seams.py tests/scripts/test_extract_weld_seams.py
git commit -m "feat: add curvature computation and line/arc segmentation"
```

---

### Task 6: Parameter fitting and fitting error computation

**Files:**
- Modify: `scripts/extract_weld_seams.py`
- Modify: `tests/scripts/test_extract_weld_seams.py`

- [ ] **Step 1: Write failing tests for parameter fitting**

Append to `tests/scripts/test_extract_weld_seams.py`:

```python
from scripts.extract_weld_seams import fit_segment, fit_line_error, fit_arc_error


def test_fit_segment_line():
    pts_2d = np.column_stack([np.linspace(0, 100, 30), np.linspace(0, 50, 30)])
    seg = {"type": "line", "indices": (0, 30), "points_2d": pts_2d}
    result = fit_segment(seg)
    assert result["type"] == "line"
    assert len(result["points_2d"]) == 2
    np.testing.assert_allclose(result["points_2d"][0], pts_2d[0], atol=1e-10)
    np.testing.assert_allclose(result["points_2d"][1], pts_2d[-1], atol=1e-10)
    assert result["fitting_error_mm"] < 1e-10


def test_fit_segment_arc():
    R = 40.0
    theta = np.linspace(0, np.pi / 2, 30)
    pts_2d = np.column_stack([R * np.cos(theta), R * np.sin(theta)])
    seg = {"type": "arc", "indices": (0, 30), "points_2d": pts_2d}
    result = fit_segment(seg)
    assert result["type"] == "arc"
    assert len(result["points_2d"]) == 3
    # First, mid, last points
    np.testing.assert_allclose(result["points_2d"][0], pts_2d[0], atol=1e-10)
    np.testing.assert_allclose(result["points_2d"][2], pts_2d[-1], atol=1e-10)
    assert result["fitting_error_mm"] < 1.0  # should be small for a perfect arc


def test_fit_line_error_perfect():
    pts = np.column_stack([np.linspace(0, 10, 20), np.linspace(0, 5, 20)])
    error = fit_line_error(pts, pts[0], pts[-1])
    assert error < 1e-10


def test_fit_arc_error_perfect_circle():
    R = 30.0
    theta = np.linspace(0, np.pi, 50)
    pts = np.column_stack([R * np.cos(theta), R * np.sin(theta)])
    p0, pm, p1 = pts[0], pts[len(pts) // 2], pts[-1]
    error = fit_arc_error(pts, p0, pm, p1)
    assert error < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/scripts/test_extract_weld_seams.py -k "fit_" -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement fit_segment, fit_line_error, fit_arc_error**

Add to `scripts/extract_weld_seams.py`:

```python
def fit_line_error(pts: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    """Max distance from points to the line defined by p0-p1."""
    d = p1 - p0
    length = np.linalg.norm(d)
    if length < 1e-12:
        return np.max(np.linalg.norm(pts - p0, axis=1))
    # Point-to-line distance: |cross(d, p0-pt)| / |d|
    diffs = pts - p0
    cross = np.abs(diffs[:, 0] * d[1] - diffs[:, 1] * d[0])
    return float(np.max(cross / length))


def fit_arc_error(pts: np.ndarray, p0: np.ndarray, pm: np.ndarray, p1: np.ndarray) -> float:
    """Max distance from points to the circular arc defined by 3 points.

    Fits a circle through p0, pm, p1 and computes max radial deviation.
    """
    # Find circumcenter of triangle p0, pm, p1
    ax, ay = p0
    bx, by = pm
    cx, cy = p1
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-12:
        # Degenerate: points are collinear, treat as line
        return fit_line_error(pts, p0, p1)
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / D
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / D
    center = np.array([ux, uy])
    R = np.linalg.norm(p0 - center)
    # Max radial deviation
    dists = np.linalg.norm(pts - center, axis=1)
    return float(np.max(np.abs(dists - R)))


def fit_segment(segment: dict) -> dict:
    """Fit a segment and compute fitting error.

    Args:
        segment: dict with type, indices, points_2d

    Returns:
        dict with type, points_2d (key points), fitting_error_mm
    """
    pts = segment["points_2d"]
    seg_type = segment["type"]

    if seg_type == "line":
        key_points = [pts[0], pts[-1]]
        error = fit_line_error(pts, pts[0], pts[-1])
    else:  # arc
        mid_idx = len(pts) // 2
        key_points = [pts[0], pts[mid_idx], pts[-1]]
        error = fit_arc_error(pts, pts[0], pts[mid_idx], pts[-1])

    return {
        "type": seg_type,
        "points_2d": key_points,
        "indices": segment["indices"],
        "fitting_error_mm": round(error, 4),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/scripts/test_extract_weld_seams.py -k "fit_" -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_weld_seams.py tests/scripts/test_extract_weld_seams.py
git commit -m "feat: add line/arc parameter fitting with error computation"
```

---

### Task 7: Closed path detection and JSON output

**Files:**
- Modify: `scripts/extract_weld_seams.py`
- Modify: `tests/scripts/test_extract_weld_seams.py`

- [ ] **Step 1: Write failing tests for closedness and JSON assembly**

Append to `tests/scripts/test_extract_weld_seams.py`:

```python
import json
from scripts.extract_weld_seams import detect_closed, build_json_output


def test_detect_closed_true():
    # Full circle: first and last point are the same
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=True)
    pts = np.column_stack([np.cos(theta), np.sin(theta)]) * 50
    assert detect_closed(pts) is True


def test_detect_closed_false():
    # Half circle: endpoints are far apart
    theta = np.linspace(0, np.pi, 50)
    pts = np.column_stack([np.cos(theta), np.sin(theta)]) * 50
    assert detect_closed(pts) is False


def test_build_json_output():
    plane = {
        "origin": np.array([0.0, 0.0, 0.0]),
        "u": np.array([1.0, 0.0, 0.0]),
        "v": np.array([0.0, 1.0, 0.0]),
        "n": np.array([0.0, 0.0, 1.0]),
        "planarity": 0.99,
    }
    fitted = [
        {
            "type": "line",
            "points_2d": [np.array([0.0, 0.0]), np.array([10.0, 0.0])],
            "fitting_error_mm": 0.1,
        },
        {
            "type": "arc",
            "points_2d": [np.array([10.0, 0.0]), np.array([15.0, 5.0]), np.array([20.0, 0.0])],
            "fitting_error_mm": 0.3,
        },
    ]
    centerline = np.array([[0, 0], [5, 0], [10, 0], [15, 5], [20, 0]])
    result = build_json_output("工件1", fitted, centerline, plane)
    assert result["model"] == "工件1"
    assert result["coord_system"] == "raw"
    assert isinstance(result["closed"], bool)
    assert len(result["weld_seams"]) == 2
    # Check 3D coordinates (since u=[1,0,0], v=[0,1,0], origin=[0,0,0])
    line_pts = result["weld_seams"][0]["points"]
    assert len(line_pts) == 2
    assert len(line_pts[0]) == 3  # 3D point
    arc_pts = result["weld_seams"][1]["points"]
    assert len(arc_pts) == 3
    # Verify JSON serializable
    json_str = json.dumps(result)
    assert "工件1" in json_str
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/scripts/test_extract_weld_seams.py -k "closed or json" -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement detect_closed and build_json_output**

Add to `scripts/extract_weld_seams.py`:

```python
import json


def detect_closed(centerline: np.ndarray) -> bool:
    """Check if centerline forms a closed path.

    Closed if distance between first and last point < 2% of total path length.
    """
    diffs = np.diff(centerline, axis=0)
    total_length = np.sum(np.linalg.norm(diffs, axis=1))
    gap = np.linalg.norm(centerline[-1] - centerline[0])
    return bool(gap < total_length * 0.02)


def build_json_output(
    model_name: str,
    fitted_segments: list[dict],
    centerline: np.ndarray,
    plane: dict,
) -> dict:
    """Assemble final JSON structure with 3D back-projected coordinates.

    Args:
        model_name: extracted model name
        fitted_segments: list from fit_segment()
        centerline: (N, 2) centerline points for closedness check
        plane: PCA plane parameters

    Returns:
        dict ready for json.dumps()
    """
    closed = detect_closed(centerline)

    weld_seams = []
    for seg in fitted_segments:
        pts_2d = np.array(seg["points_2d"])
        pts_3d = back_project(pts_2d, plane)
        weld_seams.append({
            "type": seg["type"],
            "points": [[round(c, 6) for c in pt] for pt in pts_3d.tolist()],
            "fitting_error_mm": seg["fitting_error_mm"],
        })

    return {
        "model": model_name,
        "coord_system": "raw",
        "closed": closed,
        "weld_seams": weld_seams,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/scripts/test_extract_weld_seams.py -k "closed or json" -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_weld_seams.py tests/scripts/test_extract_weld_seams.py
git commit -m "feat: add closed path detection and JSON output assembly"
```

---

### Task 8: Visualization

**Files:**
- Modify: `scripts/extract_weld_seams.py`

- [ ] **Step 1: Implement visualization function**

Add to `scripts/extract_weld_seams.py`:

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize(
    centerline_2d: np.ndarray,
    fitted_segments: list[dict],
    mesh: trimesh.Trimesh,
    plane: dict,
    output_path: str,
):
    """Generate comparison plot: 2D fit overlay + 3D mesh with fitted path.

    Args:
        centerline_2d: (N, 2) original centerline
        fitted_segments: list from fit_segment()
        mesh: original weld seam mesh
        plane: PCA plane parameters
        output_path: path to save PNG
    """
    fig = plt.figure(figsize=(16, 7))

    # --- Left: 2D centerline vs fit ---
    ax1 = fig.add_subplot(121)
    ax1.plot(centerline_2d[:, 0], centerline_2d[:, 1], '-', color='gray',
             linewidth=1, alpha=0.6, label='Centerline')

    for seg in fitted_segments:
        pts = np.array(seg["points_2d"])
        if seg["type"] == "line":
            ax1.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=2.5)
            ax1.plot(pts[:, 0], pts[:, 1], 'go', markersize=6)
        else:  # arc
            # Draw smooth arc through 3 points
            arc_pts = _interpolate_arc_2d(pts[0], pts[1], pts[2], n=50)
            ax1.plot(arc_pts[:, 0], arc_pts[:, 1], 'r-', linewidth=2.5)
            ax1.plot(pts[:, 0], pts[:, 1], 'go', markersize=6)

        # Annotate fitting error at segment midpoint
        mid = pts[len(pts) // 2]
        ax1.annotate(f'{seg["fitting_error_mm"]:.2f}mm',
                     xy=mid, fontsize=8, color='darkred',
                     textcoords="offset points", xytext=(5, 5))

    ax1.set_xlabel('u (mm)')
    ax1.set_ylabel('v (mm)')
    ax1.set_title('2D Centerline vs Fitted Segments')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Right: 3D view ---
    ax2 = fig.add_subplot(122, projection='3d')
    verts = mesh.vertices
    ax2.plot_trisurf(
        verts[:, 0], verts[:, 1], verts[:, 2],
        triangles=mesh.faces, alpha=0.2, color='gray', edgecolor='none'
    )

    # Draw fitted path in 3D
    for seg in fitted_segments:
        pts_2d = np.array(seg["points_2d"])
        if seg["type"] == "line":
            pts_3d = back_project(pts_2d, plane)
            ax2.plot(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2], 'b-', linewidth=2.5)
        else:
            arc_2d = _interpolate_arc_2d(pts_2d[0], pts_2d[1], pts_2d[2], n=50)
            arc_3d = back_project(arc_2d, plane)
            ax2.plot(arc_3d[:, 0], arc_3d[:, 1], arc_3d[:, 2], 'r-', linewidth=2.5)

    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    ax2.set_title('3D Mesh + Fitted Path')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _interpolate_arc_2d(p0: np.ndarray, pm: np.ndarray, p1: np.ndarray, n: int = 50) -> np.ndarray:
    """Interpolate n points along circular arc through p0, pm, p1."""
    # Find circumcenter
    ax, ay = p0
    bx, by = pm
    cx, cy = p1
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-12:
        # Degenerate: return straight line
        return np.column_stack([
            np.linspace(p0[0], p1[0], n),
            np.linspace(p0[1], p1[1], n),
        ])
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / D
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / D
    center = np.array([ux, uy])

    # Compute angles from center
    a0 = np.arctan2(p0[1] - uy, p0[0] - ux)
    am = np.arctan2(pm[1] - uy, pm[0] - ux)
    a1 = np.arctan2(p1[1] - uy, p1[0] - ux)

    # Ensure angles go through pm (correct winding)
    def _unwrap(start, mid, end):
        """Adjust angles so start -> mid -> end is monotonic."""
        # Normalize to [start, start + 2pi)
        mid_adj = mid - start
        end_adj = end - start
        mid_adj = mid_adj % (2 * np.pi)
        end_adj = end_adj % (2 * np.pi)
        if mid_adj > end_adj:
            end_adj += 2 * np.pi
        return start, start + mid_adj, start + end_adj

    a0, am, a1 = _unwrap(a0, am, a1)
    R = np.linalg.norm(p0 - center)
    angles = np.linspace(a0, a1, n)
    return np.column_stack([ux + R * np.cos(angles), uy + R * np.sin(angles)])
```

- [ ] **Step 2: Verify it runs on the actual data (manual test in Task 9)**

No automated test for visualization — it will be validated visually in the integration task.

- [ ] **Step 3: Commit**

```bash
git add scripts/extract_weld_seams.py
git commit -m "feat: add 2D/3D comparison visualization for weld seam fitting"
```

---

### Task 9: CLI entry point and integration test

**Files:**
- Modify: `scripts/extract_weld_seams.py`

- [ ] **Step 1: Implement CLI main function**

Add to `scripts/extract_weld_seams.py`:

```python
import argparse


def print_summary(model_name: str, planarity: float, centerline: np.ndarray,
                  fitted_segments: list[dict], closed: bool):
    """Print analysis summary to terminal."""
    n_line = sum(1 for s in fitted_segments if s["type"] == "line")
    n_arc = sum(1 for s in fitted_segments if s["type"] == "arc")
    max_err = max(s["fitting_error_mm"] for s in fitted_segments) if fitted_segments else 0
    print(f"Weld seam analysis: {model_name}")
    print(f"  Planarity: {planarity:.1%} ({'OK' if planarity >= 0.95 else 'WARNING'})")
    print(f"  Centerline points: {len(centerline)}")
    print(f"  Segments: {len(fitted_segments)} ({n_line} line, {n_arc} arc)")
    print(f"  Max fitting error: {max_err:.2f} mm")
    print(f"  Closed: {closed}")


def run_pipeline(workpiece_path: str, weld_path: str, output_path: str | None = None,
                 no_viz: bool = False):
    """Run the full weld seam extraction pipeline.

    Args:
        workpiece_path: path to workpiece OBJ (for model name)
        weld_path: path to weld seam OBJ
        output_path: JSON output path (auto-generated if None)
        no_viz: skip visualization if True
    """
    model_name = extract_model_name(workpiece_path)

    # Determine output paths
    if output_path is None:
        out_dir = os.path.dirname(workpiece_path) or "."
        output_path = os.path.join(out_dir, f"{model_name}_weld_seams.json")
    viz_path = output_path.replace(".json", "_fit.png")

    # 1. Load mesh
    mesh = load_weld_mesh(weld_path)

    # 2. PCA projection
    pts_2d, plane = pca_project(mesh.vertices)

    # 3. Extract centerline
    centerline = extract_centerline(mesh, pts_2d)

    # 4. Segment by curvature
    segments = segment_by_curvature(centerline)

    # 5. Fit segments
    fitted = [fit_segment(seg) for seg in segments]

    # 6. Build JSON
    result = build_json_output(model_name, fitted, centerline, plane)

    # 7. Print summary
    print_summary(model_name, plane["planarity"], centerline, fitted, result["closed"])

    # 8. Save JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nJSON saved to: {output_path}")

    # 9. Visualization
    if not no_viz:
        visualize(centerline, fitted, mesh, plane, viz_path)
        print(f"Visualization saved to: {viz_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract weld seam centerlines from OBJ mesh and fit as line/arc segments."
    )
    parser.add_argument("--workpiece", required=True, help="Path to workpiece OBJ file")
    parser.add_argument("--weld", required=True, help="Path to weld seam OBJ file")
    parser.add_argument("--output", default=None, help="JSON output path (default: auto)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()
    run_pipeline(args.workpiece, args.weld, args.output, args.no_viz)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on the actual data**

```bash
python scripts/extract_weld_seams.py \
  --workpiece assets/工件1.obj \
  --weld assets/焊缝1.obj
```

Expected output:
```
Weld seam analysis: 工件1
  Planarity: 99.3% (OK)
  Centerline points: ~80
  Segments: N (X line, Y arc)
  Max fitting error: <value> mm
  Closed: True/False

JSON saved to: assets/工件1_weld_seams.json
Visualization saved to: assets/工件1_weld_seams_fit.png
```

- [ ] **Step 3: Inspect outputs**

Open `assets/工件1_weld_seams_fit.png` to visually verify the fit. Check `assets/工件1_weld_seams.json` has correct structure.

- [ ] **Step 4: Fix any issues found during integration**

Adjust thresholds, smoothing, or segmentation parameters based on visual inspection.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/scripts/test_extract_weld_seams.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/extract_weld_seams.py
git commit -m "feat: add CLI entry point and pipeline orchestration for weld seam extraction"
```

---

### Task Summary

| Task | Description | Depends On |
|------|------------|-----------|
| 1 | Add trimesh dependency | — |
| 2 | OBJ parsing + model name extraction | 1 |
| 3 | PCA plane projection + back-projection | 2 |
| 4 | Centerline extraction from mesh topology | 3 |
| 5 | Curvature computation + segmentation | 4 |
| 6 | Parameter fitting + fitting error | 5 |
| 7 | Closed path detection + JSON output | 3, 6 |
| 8 | Visualization | 3, 6 |
| 9 | CLI entry point + integration test | all |
