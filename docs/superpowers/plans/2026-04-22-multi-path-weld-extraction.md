# Multi-Path Weld Seam Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `scripts/extract_weld_seams.py` pipeline to process weld seam meshes containing multiple disconnected components (e.g., bellmouth's two independent straight lines), producing a nested JSON schema with one entry per path.

**Architecture:** Split the loaded weld mesh into connected components via `trimesh.Trimesh.split()`. For each component, run the existing single-path pipeline (PCA → centerline → segmentation → fitting) independently with its own local PCA plane. Aggregate results into a `weld_paths` array, each entry containing its own `closed` flag and `segments`. For 2D visualization, project all paths onto a single global PCA plane for unified viewing while preserving per-path local fits.

**Tech Stack:** Python, numpy, trimesh, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-04-22-multi-path-weld-extraction-design.md`

---

## File Structure

**Modify:** `scripts/extract_weld_seams.py`
- Replace `build_json_output` with `build_json_output_multi` (new nested schema)
- Replace `visualize` with `visualize_multi` (multi-path overlay on global PCA plane)
- Replace `print_summary` signature for multi-path summary
- Add helper `_process_component` (single-path pipeline, used per component)
- Refactor `run_pipeline` to split mesh and loop over components

**Modify:** `tests/scripts/test_extract_weld_seams.py`
- Update `test_build_json_output` for new schema
- Add `test_build_json_output_multi_paths` (two paths)
- Add `test_run_pipeline_multi_component` (integration with synthetic 2-line OBJ)
- Add `test_component_filter` (small noise components filtered out)

**Rerun:** `scripts/extract_all_weld_seams.sh` to regenerate all output JSON+PNG with new schema.

---

## Task 1: Replace `build_json_output` with multi-path version

**Files:**
- Modify: `scripts/extract_weld_seams.py:449-481` (remove old `build_json_output`, add `build_json_output_multi`)
- Modify: `tests/scripts/test_extract_weld_seams.py:262-293` (update `test_build_json_output` to new schema, add multi-path test)

- [ ] **Step 1: Write the new single-path test (updating existing)**

Replace the existing `test_build_json_output` in `tests/scripts/test_extract_weld_seams.py`:

```python
def test_build_json_output_single_path():
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
    paths_data = [{
        "centerline_2d": centerline,
        "plane": plane,
        "fitted": fitted,
        "closed": False,
    }]
    result = build_json_output_multi("工件1", paths_data)
    assert result["model"] == "工件1"
    assert result["coord_system"] == "raw"
    assert "closed" not in result
    assert len(result["weld_paths"]) == 1
    path = result["weld_paths"][0]
    assert path["closed"] is False
    assert len(path["segments"]) == 2
    line_pts = path["segments"][0]["points"]
    assert len(line_pts) == 2
    assert len(line_pts[0]) == 3
    arc_pts = path["segments"][1]["points"]
    assert len(arc_pts) == 3
    json_str = json.dumps(result, ensure_ascii=False)
    assert "工件1" in json_str
```

Also update the import at line 88 to use `build_json_output_multi`:

```python
from scripts.extract_weld_seams import detect_closed, build_json_output_multi
```

And remove the old `test_build_json_output` function (replaced by `test_build_json_output_single_path`).

- [ ] **Step 2: Write multi-path test**

Add after `test_build_json_output_single_path`:

```python
def test_build_json_output_multi_paths():
    plane_a = {
        "origin": np.array([0.0, 0.0, 0.0]),
        "u": np.array([1.0, 0.0, 0.0]),
        "v": np.array([0.0, 1.0, 0.0]),
        "n": np.array([0.0, 0.0, 1.0]),
        "planarity": 0.99,
    }
    plane_b = {
        "origin": np.array([100.0, 0.0, 0.0]),
        "u": np.array([0.0, 1.0, 0.0]),
        "v": np.array([0.0, 0.0, 1.0]),
        "n": np.array([1.0, 0.0, 0.0]),
        "planarity": 0.99,
    }
    fitted_a = [{
        "type": "line",
        "points_2d": [np.array([0.0, 0.0]), np.array([10.0, 0.0])],
        "fitting_error_mm": 0.05,
    }]
    fitted_b = [{
        "type": "line",
        "points_2d": [np.array([0.0, 0.0]), np.array([20.0, 0.0])],
        "fitting_error_mm": 0.08,
    }]
    paths_data = [
        {"centerline_2d": np.array([[0, 0], [10, 0]]),
         "plane": plane_a, "fitted": fitted_a, "closed": False},
        {"centerline_2d": np.array([[0, 0], [20, 0]]),
         "plane": plane_b, "fitted": fitted_b, "closed": True},
    ]
    result = build_json_output_multi("bellmouth", paths_data)
    assert len(result["weld_paths"]) == 2
    assert result["weld_paths"][0]["closed"] is False
    assert result["weld_paths"][1]["closed"] is True
    assert len(result["weld_paths"][0]["segments"]) == 1
    assert len(result["weld_paths"][1]["segments"]) == 1
    # Path A line endpoints are in plane_a coordinate system
    path_a_pts = result["weld_paths"][0]["segments"][0]["points"]
    np.testing.assert_allclose(path_a_pts[0], [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(path_a_pts[1], [10, 0, 0], atol=1e-6)
    # Path B line endpoints back-projected via plane_b (origin=(100,0,0), u=(0,1,0))
    path_b_pts = result["weld_paths"][1]["segments"][0]["points"]
    np.testing.assert_allclose(path_b_pts[0], [100, 0, 0], atol=1e-6)
    np.testing.assert_allclose(path_b_pts[1], [100, 20, 0], atol=1e-6)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/scripts/test_extract_weld_seams.py::test_build_json_output_single_path tests/scripts/test_extract_weld_seams.py::test_build_json_output_multi_paths -v`

Expected: FAIL — `build_json_output_multi` is not defined (or `build_json_output` still exists with old schema).

- [ ] **Step 4: Implement `build_json_output_multi`**

In `scripts/extract_weld_seams.py`, replace the existing `build_json_output` function (lines 463-481) with:

```python
def build_json_output_multi(model_name, paths_data):
    """Assemble final JSON with multiple weld paths.

    Args:
        model_name: model name string
        paths_data: list of dicts, each containing:
            - "fitted": list of fitted segment dicts (with points_2d)
            - "plane": PCA plane dict for this path
            - "closed": bool, whether this path is closed

    Returns:
        JSON-serializable dict with model, coord_system, and weld_paths.
    """
    weld_paths = []
    for path in paths_data:
        segments_json = []
        for seg in path["fitted"]:
            pts_2d = np.array(seg["points_2d"])
            pts_3d = back_project(pts_2d, path["plane"])
            segments_json.append({
                "type": seg["type"],
                "points": [[round(c, 6) for c in pt] for pt in pts_3d.tolist()],
                "fitting_error_mm": seg["fitting_error_mm"],
            })
        weld_paths.append({
            "closed": bool(path["closed"]),
            "segments": segments_json,
        })
    return {
        "model": model_name,
        "coord_system": "raw",
        "weld_paths": weld_paths,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_extract_weld_seams.py::test_build_json_output_single_path tests/scripts/test_extract_weld_seams.py::test_build_json_output_multi_paths -v`

Expected: 2 passed.

Note: `run_pipeline` will be broken now (still calls `build_json_output`). Next task fixes that.

- [ ] **Step 6: Commit**

```bash
git add scripts/extract_weld_seams.py tests/scripts/test_extract_weld_seams.py
git commit -m "$(cat <<'EOF'
refactor: replace build_json_output with multi-path build_json_output_multi

New JSON schema has weld_paths at top level, each path containing its
own closed flag and segments array. Each path uses its own PCA plane
for back-projection.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `_process_component` helper

**Files:**
- Modify: `scripts/extract_weld_seams.py` (add new helper before `run_pipeline`)
- Test: `tests/scripts/test_extract_weld_seams.py` (add test for component processing)

- [ ] **Step 1: Write the failing test**

Add at the end of `tests/scripts/test_extract_weld_seams.py`:

```python
from scripts.extract_weld_seams import _process_component


def test_process_component_straight_tube():
    mesh = _make_tube_mesh(n_rings=20, n_per_ring=8, radius=2.0, length=100.0)
    path = _process_component(mesh, force_close=False)
    assert "centerline_2d" in path
    assert "plane" in path
    assert "fitted" in path
    assert "closed" in path
    assert len(path["fitted"]) >= 1
    assert path["closed"] is False
    for seg in path["fitted"]:
        assert "type" in seg
        assert "points_2d" in seg
        assert "fitting_error_mm" in seg


def test_process_component_force_close_adds_segment():
    mesh = _make_tube_mesh(n_rings=20, n_per_ring=8, radius=2.0, length=100.0)
    path_open = _process_component(mesh, force_close=False)
    path_closed = _process_component(mesh, force_close=True)
    assert len(path_closed["fitted"]) == len(path_open["fitted"]) + 1
    assert path_closed["closed"] is True
    # Closing segment is a line with zero fitting error
    closing = path_closed["fitted"][-1]
    assert closing["type"] == "line"
    assert closing["fitting_error_mm"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/scripts/test_extract_weld_seams.py::test_process_component_straight_tube tests/scripts/test_extract_weld_seams.py::test_process_component_force_close_adds_segment -v`

Expected: FAIL — `_process_component` not defined.

- [ ] **Step 3: Implement `_process_component`**

In `scripts/extract_weld_seams.py`, add this function just before `run_pipeline` (around line 574):

```python
def _process_component(component_mesh, force_close=False):
    """Run the single-path pipeline (PCA → centerline → segment → fit) on one mesh component.

    Args:
        component_mesh: a trimesh.Trimesh representing one connected weld component
        force_close: if True and path not naturally closed, append a closing line

    Returns:
        dict with keys: centerline_2d, plane, fitted, closed
    """
    pts_2d, plane = pca_project(component_mesh.vertices)
    centerline = extract_centerline(component_mesh, pts_2d)
    segments = segment_by_curvature(centerline)
    fitted = [fit_segment(seg) for seg in segments]
    is_closed = detect_closed(centerline)
    if force_close and not is_closed:
        fitted = fitted + [_make_closing_segment(fitted, centerline)]
        is_closed = True
    return {
        "centerline_2d": centerline,
        "plane": plane,
        "fitted": fitted,
        "closed": is_closed,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_extract_weld_seams.py::test_process_component_straight_tube tests/scripts/test_extract_weld_seams.py::test_process_component_force_close_adds_segment -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_weld_seams.py tests/scripts/test_extract_weld_seams.py
git commit -m "$(cat <<'EOF'
feat: add _process_component helper for single-path pipeline

Encapsulates the PCA→centerline→segment→fit→force-close steps for
one mesh component. Enables run_pipeline to loop over connected
components and process each independently.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update `visualize` to multi-path overlay

**Files:**
- Modify: `scripts/extract_weld_seams.py:512-564` (replace `visualize` with `visualize_multi`)

Note: No explicit unit test for visualization correctness — matplotlib plotting is hard to assert on pixel values. Task 6 verifies by regenerating all PNGs and user inspection.

- [ ] **Step 1: Replace `visualize` with `visualize_multi`**

In `scripts/extract_weld_seams.py`, replace the entire `visualize` function (lines 512-564) with:

```python
# Path color cycle: (line_color, arc_color) per path index
_PATH_COLORS = [
    ("tab:blue", "tab:red"),
    ("tab:purple", "tab:orange"),
    ("tab:green", "tab:pink"),
    ("tab:cyan", "tab:brown"),
    ("tab:olive", "tab:gray"),
]


def _global_pca_plane(paths_data):
    """Compute a unified PCA plane across all centerlines' 3D points."""
    all_pts_3d = []
    for path in paths_data:
        pts_3d = back_project(path["centerline_2d"], path["plane"])
        all_pts_3d.append(pts_3d)
    all_pts_3d = np.vstack(all_pts_3d)
    _, global_plane = pca_project(all_pts_3d)
    return global_plane


def _project_to_plane(pts_3d, plane):
    """Project 3D points onto the given PCA plane returning 2D coords."""
    diffs = pts_3d - plane["origin"]
    u = np.dot(diffs, plane["u"])
    v = np.dot(diffs, plane["v"])
    return np.column_stack([u, v])


def visualize_multi(paths_data, mesh, output_path):
    """Generate multi-path overlay plot: all paths on a single 2D+3D panel.

    Each path is drawn in its own color (line / arc pair from _PATH_COLORS).
    The 2D panel uses a global PCA plane so all paths share coordinates.
    """
    global_plane = _global_pca_plane(paths_data)

    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    # 3D background: the full mesh in semi-transparent gray
    verts = mesh.vertices
    ax2.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                     triangles=mesh.faces, alpha=0.2, color='gray',
                     edgecolor='none')

    for idx, path in enumerate(paths_data):
        line_color, arc_color = _PATH_COLORS[idx % len(_PATH_COLORS)]
        local_plane = path["plane"]

        # 2D centerline (via global plane for unified coords)
        cl_3d = back_project(path["centerline_2d"], local_plane)
        cl_2d_global = _project_to_plane(cl_3d, global_plane)
        ax1.plot(cl_2d_global[:, 0], cl_2d_global[:, 1], '-',
                 color='gray', linewidth=1, alpha=0.5,
                 label=f'Path {idx} centerline' if idx == 0 else None)

        for seg in path["fitted"]:
            pts_local_2d = np.array(seg["points_2d"])
            pts_3d = back_project(pts_local_2d, local_plane)
            pts_global_2d = _project_to_plane(pts_3d, global_plane)
            if seg["type"] == "line":
                ax1.plot(pts_global_2d[:, 0], pts_global_2d[:, 1],
                         '-', color=line_color, linewidth=2.5)
                ax1.plot(pts_global_2d[:, 0], pts_global_2d[:, 1],
                         'o', color=line_color, markersize=5)
                ax2.plot(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
                         '-', color=line_color, linewidth=2.5)
            else:
                arc_local_2d = _interpolate_arc_2d(
                    pts_local_2d[0], pts_local_2d[1], pts_local_2d[2], n=50)
                arc_3d = back_project(arc_local_2d, local_plane)
                arc_global_2d = _project_to_plane(arc_3d, global_plane)
                ax1.plot(arc_global_2d[:, 0], arc_global_2d[:, 1],
                         '-', color=arc_color, linewidth=2.5)
                ax1.plot(pts_global_2d[:, 0], pts_global_2d[:, 1],
                         'o', color=arc_color, markersize=5)
                ax2.plot(arc_3d[:, 0], arc_3d[:, 1], arc_3d[:, 2],
                         '-', color=arc_color, linewidth=2.5)
            # Error annotation at mid-point of segment
            mid_idx = len(pts_global_2d) // 2
            mid = pts_global_2d[mid_idx]
            ax1.annotate(f'{seg["fitting_error_mm"]:.2f}mm',
                         xy=mid, fontsize=7, color='darkred',
                         textcoords="offset points", xytext=(4, 4))

    ax1.set_xlabel('u (mm)')
    ax1.set_ylabel('v (mm)')
    ax1.set_title('2D: all paths (global PCA plane)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    ax2.set_title('3D Mesh + Fitted Paths')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

- [ ] **Step 2: Verify syntax is valid by importing the module**

Run: `python -c "from scripts.extract_weld_seams import visualize_multi, _process_component, build_json_output_multi; print('OK')"`

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/extract_weld_seams.py
git commit -m "$(cat <<'EOF'
feat: add visualize_multi for multi-path overlay visualization

Replaces single-path visualize. All paths render on one 2D+3D panel
with per-path color pairs (line/arc). 2D coordinates unified via
global PCA plane so multiple paths display with correct relative
positioning.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update `print_summary` for multi-path

**Files:**
- Modify: `scripts/extract_weld_seams.py:562-576` (rewrite `print_summary`)

- [ ] **Step 1: Rewrite `print_summary`**

In `scripts/extract_weld_seams.py`, replace the existing `print_summary` function with:

```python
def print_summary(model_name, paths_data):
    """Print analysis summary for a multi-path weld seam."""
    print(f"Weld seam analysis: {model_name}")
    print(f"  Paths: {len(paths_data)}")
    all_errors = []
    for idx, path in enumerate(paths_data):
        planarity = path["plane"]["planarity"]
        planarity_ok = "OK" if planarity >= 0.95 else "WARNING"
        n_pts = len(path["centerline_2d"])
        n_line = sum(1 for s in path["fitted"] if s["type"] == "line")
        n_arc = sum(1 for s in path["fitted"] if s["type"] == "arc")
        errors = [s["fitting_error_mm"] for s in path["fitted"]]
        max_err = max(errors) if errors else 0.0
        all_errors.extend(errors)
        print(f"  Path {idx}: Planarity {planarity:.1%} ({planarity_ok}) | "
              f"{n_pts} centerline pts | "
              f"{len(path['fitted'])} segments ({n_line} line, {n_arc} arc) | "
              f"closed: {path['closed']} | max err: {max_err:.2f} mm")
    overall_max = max(all_errors) if all_errors else 0.0
    print(f"  Overall max fitting error: {overall_max:.2f} mm")
```

- [ ] **Step 2: Verify syntax is valid**

Run: `python -c "from scripts.extract_weld_seams import print_summary; print('OK')"`

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/extract_weld_seams.py
git commit -m "$(cat <<'EOF'
refactor: update print_summary for multi-path output

Prints per-path metrics (planarity, centerline size, segment count,
closed flag, max error) plus an overall max error across paths.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Refactor `run_pipeline` to split mesh and loop

**Files:**
- Modify: `scripts/extract_weld_seams.py:574-604` (rewrite `run_pipeline`)
- Test: `tests/scripts/test_extract_weld_seams.py` (integration test + filter test)

- [ ] **Step 1: Write integration test for multi-component pipeline**

Add at the end of `tests/scripts/test_extract_weld_seams.py`:

```python
from scripts.extract_weld_seams import run_pipeline


def _make_two_line_obj(tmp_path):
    """Create an OBJ with two disconnected straight tube components."""
    verts = []
    faces = []

    # First tube: along X axis from 0 to 50
    n_rings_a = 10
    n_per_ring = 6
    tube_r = 1.0
    for i in range(n_rings_a):
        x = 50 * i / (n_rings_a - 1)
        for j in range(n_per_ring):
            ang = 2 * np.pi * j / n_per_ring
            verts.append([x, tube_r * np.cos(ang), tube_r * np.sin(ang)])
    for i in range(n_rings_a - 1):
        for j in range(n_per_ring):
            jn = (j + 1) % n_per_ring
            v0 = i * n_per_ring + j
            v1 = i * n_per_ring + jn
            v2 = (i + 1) * n_per_ring + j
            v3 = (i + 1) * n_per_ring + jn
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    # Second tube: parallel, offset in Y, from 0 to 50
    offset_a = n_rings_a * n_per_ring
    for i in range(n_rings_a):
        x = 50 * i / (n_rings_a - 1)
        for j in range(n_per_ring):
            ang = 2 * np.pi * j / n_per_ring
            verts.append([x, 30.0 + tube_r * np.cos(ang), tube_r * np.sin(ang)])
    for i in range(n_rings_a - 1):
        for j in range(n_per_ring):
            jn = (j + 1) % n_per_ring
            v0 = offset_a + i * n_per_ring + j
            v1 = offset_a + i * n_per_ring + jn
            v2 = offset_a + (i + 1) * n_per_ring + j
            v3 = offset_a + (i + 1) * n_per_ring + jn
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    lines = ["o 焊缝"]
    for v in verts:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")

    obj_path = tmp_path / "two_lines_weld.obj"
    obj_path.write_text("\n".join(lines))

    # Also need a minimal workpiece OBJ so extract_model_name works
    wp_path = tmp_path / "dual.obj"
    wp_path.write_text("o dummy\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    return str(wp_path), str(obj_path)


def test_run_pipeline_multi_component(tmp_path):
    wp_path, weld_path = _make_two_line_obj(tmp_path)
    output = tmp_path / "out.json"
    run_pipeline(wp_path, weld_path, str(output), no_viz=True, force_close=False)
    with open(output) as f:
        data = json.load(f)
    assert data["model"] == "dual"
    assert "weld_paths" in data
    assert len(data["weld_paths"]) == 2
    for path in data["weld_paths"]:
        assert path["closed"] is False
        assert len(path["segments"]) >= 1
        # Each path should be predominantly a line
        assert any(s["type"] == "line" for s in path["segments"])


def test_component_filter_drops_tiny(tmp_path):
    """A mesh with one real component + one tiny noise component should output 1 path."""
    verts = []
    faces = []

    # Real tube component (60 verts)
    n_rings = 10
    n_per_ring = 6
    tube_r = 1.0
    for i in range(n_rings):
        x = 50 * i / (n_rings - 1)
        for j in range(n_per_ring):
            ang = 2 * np.pi * j / n_per_ring
            verts.append([x, tube_r * np.cos(ang), tube_r * np.sin(ang)])
    for i in range(n_rings - 1):
        for j in range(n_per_ring):
            jn = (j + 1) % n_per_ring
            v0 = i * n_per_ring + j
            v1 = i * n_per_ring + jn
            v2 = (i + 1) * n_per_ring + j
            v3 = (i + 1) * n_per_ring + jn
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    # Tiny noise component (4 verts, 2 triangles — should be filtered out)
    base = len(verts)
    verts.extend([
        [1000, 1000, 1000],
        [1001, 1000, 1000],
        [1000, 1001, 1000],
        [1000, 1000, 1001],
    ])
    faces.append([base, base + 1, base + 2])
    faces.append([base, base + 1, base + 3])

    lines = ["o 焊缝"]
    for v in verts:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")

    weld_path = tmp_path / "noisy_weld.obj"
    weld_path.write_text("\n".join(lines))

    wp_path = tmp_path / "noisy.obj"
    wp_path.write_text("o dummy\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    output = tmp_path / "out.json"
    run_pipeline(str(wp_path), str(weld_path), str(output), no_viz=True, force_close=False)
    with open(output) as f:
        data = json.load(f)
    # Small 4-vertex noise component should be filtered out (< 10 verts)
    assert len(data["weld_paths"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/scripts/test_extract_weld_seams.py::test_run_pipeline_multi_component tests/scripts/test_extract_weld_seams.py::test_component_filter_drops_tiny -v`

Expected: FAIL — `run_pipeline` still uses old single-path flow with `build_json_output` (removed in Task 1), so it errors or produces old schema.

- [ ] **Step 3: Rewrite `run_pipeline`**

Replace the `run_pipeline` function (lines 574-604 in the original; actual line numbers may have shifted). Locate it by searching for `def run_pipeline(`. Replace entire function body with:

```python
MIN_COMPONENT_VERTICES = 10


def run_pipeline(workpiece_path, weld_path, output_path=None, no_viz=False,
                 force_close=False):
    model_name = extract_model_name(workpiece_path)
    if output_path is None:
        out_dir = os.path.dirname(workpiece_path) or "."
        output_path = os.path.join(out_dir, f"{model_name}_weld_seams.json")
    viz_path = output_path.replace(".json", "_fit.png")

    mesh = load_weld_mesh(weld_path)
    components = mesh.split(only_watertight=False)
    components = [c for c in components if len(c.vertices) >= MIN_COMPONENT_VERTICES]
    if len(components) == 0:
        raise ValueError(
            f"No valid mesh components found (all < {MIN_COMPONENT_VERTICES} vertices)"
        )

    paths_data = [_process_component(c, force_close=force_close) for c in components]

    result = build_json_output_multi(model_name, paths_data)
    print_summary(model_name, paths_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nJSON saved to: {output_path}")

    if not no_viz:
        visualize_multi(paths_data, mesh, viz_path)
        print(f"Visualization saved to: {viz_path}")
```

- [ ] **Step 4: Run all tests to verify full suite passes**

Run: `python -m pytest tests/scripts/test_extract_weld_seams.py -v`

Expected: all tests pass (previously 20 + new ones from Task 1, 2, 5 = 25 total).

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_weld_seams.py tests/scripts/test_extract_weld_seams.py
git commit -m "$(cat <<'EOF'
feat: run_pipeline splits mesh into components and processes each

Uses trimesh.split(only_watertight=False) to separate disconnected
weld paths. Each component runs the full pipeline independently
with its own PCA plane. Components with fewer than 10 vertices are
filtered as noise.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Regenerate all outputs and verify

**Files:**
- Regenerate: `assets/weld_seams_output/*.json` and `*.png`

- [ ] **Step 1: Run the batch script**

Run: `bash scripts/extract_all_weld_seams.sh`

Expected output per category:
```
=== Processing bellmouth ===
Weld seam analysis: bellmouth
  Paths: 2
  Path 0: ... 0 arc ...
  Path 1: ... 0 arc ...
  Overall max fitting error: <small>
...
```

For bellmouth/channel_steel/H_beam expect `Paths: 2`. For square_tube/cover_plate expect `Paths: 1`.

- [ ] **Step 2: Verify one JSON has the new schema**

Run: `python -c "import json; d = json.load(open('assets/weld_seams_output/bellmouth_weld_seams.json')); print('weld_paths' in d, 'closed' not in d, len(d['weld_paths']))"`

Expected output: `True True 2`

- [ ] **Step 3: Inspect each PNG visually**

Open each `assets/weld_seams_output/*_fit.png` and confirm:
- bellmouth: two separate straight lines (path 0 blue, path 1 purple)
- channel_steel: two separate paths
- H_beam: two separate paths
- square_tube: one path (closed circle or near-circle)
- cover_plate: one path (closed racetrack with 4 arcs + 4 lines)

Record any visual issues for follow-up but do not block commit on them — the multi-path extraction is the core deliverable.

- [ ] **Step 4: Verify workpiece 工件1 still works**

Run: `python scripts/extract_weld_seams.py --workpiece "assets/工件1.obj" --weld "assets/焊缝1.obj" --output "assets/工件1_weld_seams.json" --force-close`

Expected: JSON saved, PNG saved, no errors. The new JSON has the nested `weld_paths` schema.

- [ ] **Step 5: Commit regenerated outputs**

Note: JSON/PNG files in `assets/weld_seams_output/` may or may not already be tracked by git. Only add files if the user has explicitly decided to track them.

Skip this step if the assets output files are listed in `.gitignore` or have never been committed. Otherwise:

```bash
git status assets/weld_seams_output/
```

If files show as modified/untracked and the user wants them tracked:

```bash
git add assets/weld_seams_output/
git commit -m "$(cat <<'EOF'
chore: regenerate weld seam outputs with multi-path schema

All 5 categories reprocessed with the new pipeline. Bellmouth,
channel_steel, and H_beam now correctly split into 2 paths each.
Square_tube and cover_plate remain single-path.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

Otherwise skip the add/commit (outputs are local artifacts only).
