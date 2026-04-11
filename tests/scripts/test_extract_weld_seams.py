import json

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


from scripts.extract_weld_seams import pca_project, back_project


def test_pca_project_planar_points():
    rng = np.random.default_rng(42)
    xy = rng.uniform(-50, 50, (100, 2))
    points_3d = np.column_stack([xy, np.zeros(100)])
    pts_2d, plane = pca_project(points_3d)
    assert pts_2d.shape == (100, 2)
    assert plane["origin"].shape == (3,)
    assert plane["u"].shape == (3,)
    assert plane["v"].shape == (3,)
    assert plane["planarity"] > 0.95


def test_pca_project_warns_non_planar():
    rng = np.random.default_rng(42)
    points_3d = rng.uniform(-50, 50, (100, 3))
    with pytest.warns(UserWarning, match="planarity"):
        pts_2d, plane = pca_project(points_3d)
    assert plane["planarity"] < 0.95


def test_back_project_roundtrip():
    rng = np.random.default_rng(42)
    xy = rng.uniform(-50, 50, (50, 2))
    noise = rng.normal(0, 0.01, 50)
    points_3d = np.column_stack([xy, noise])
    pts_2d, plane = pca_project(points_3d)
    recovered = back_project(pts_2d, plane)
    assert recovered.shape == (50, 3)
    projected_orig = back_project(pca_project(points_3d)[0], plane)
    np.testing.assert_allclose(recovered, projected_orig, atol=1e-10)


from scripts.extract_weld_seams import compute_curvature, segment_by_curvature
from scripts.extract_weld_seams import fit_segment, fit_line_error, fit_arc_error
from scripts.extract_weld_seams import extract_centerline
from scripts.extract_weld_seams import detect_closed, build_json_output


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
    x_range = centerline[:, 0].max() - centerline[:, 0].min()
    y_range = centerline[:, 1].max() - centerline[:, 1].min()
    assert x_range > 50
    assert y_range < 5


def test_extract_centerline_arc_tube():
    n_rings, n_per_ring = 30, 8
    arc_radius, tube_radius = 50.0, 2.0
    verts = []
    for i in range(n_rings):
        theta = np.pi * i / (n_rings - 1)
        cx = arc_radius * np.cos(theta)
        cy = arc_radius * np.sin(theta)
        for j in range(n_per_ring):
            angle = 2 * np.pi * j / n_per_ring
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
    # Fit a circle to the centerline and verify the radius matches arc_radius.
    # This is more robust than checking individual point distances.
    from scripts.extract_weld_seams import _fit_circle_center
    center, fitted_radius = _fit_circle_center(centerline)
    assert abs(fitted_radius - arc_radius) < 5.0, (
        f"Fitted radius {fitted_radius:.1f} differs from expected {arc_radius}"
    )


def test_compute_curvature_straight_line():
    pts = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
    kappa = compute_curvature(pts)
    assert len(kappa) == 50
    assert np.all(np.abs(kappa[1:-1]) < 1e-6)


def test_compute_curvature_circle():
    R = 50.0
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    pts = np.column_stack([R * np.cos(theta), R * np.sin(theta)])
    kappa = compute_curvature(pts)
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
    line1 = np.column_stack([np.linspace(0, 30, 20), np.zeros(20)])
    theta = np.linspace(-np.pi / 2, 0, 20)
    arc = np.column_stack([30 + 20 * np.cos(theta), 20 + 20 * np.sin(theta)])
    line2 = np.column_stack([50 * np.ones(20), np.linspace(20, 50, 20)])
    pts = np.vstack([line1, arc[1:], line2[1:]])
    segments = segment_by_curvature(pts)
    types = [s["type"] for s in segments]
    assert "line" in types
    assert "arc" in types
    assert len(segments) >= 2


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
    np.testing.assert_allclose(result["points_2d"][0], pts_2d[0], atol=1e-10)
    np.testing.assert_allclose(result["points_2d"][2], pts_2d[-1], atol=1e-10)
    assert result["fitting_error_mm"] < 1.0


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


def test_detect_closed_true():
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=True)
    pts = np.column_stack([np.cos(theta), np.sin(theta)]) * 50
    assert detect_closed(pts) is True


def test_detect_closed_false():
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
    line_pts = result["weld_seams"][0]["points"]
    assert len(line_pts) == 2
    assert len(line_pts[0]) == 3
    arc_pts = result["weld_seams"][1]["points"]
    assert len(arc_pts) == 3
    json_str = json.dumps(result, ensure_ascii=False)
    assert "工件1" in json_str
