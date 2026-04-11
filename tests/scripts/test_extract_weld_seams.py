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


def test_pca_project_warns_non_planar(capsys):
    rng = np.random.default_rng(42)
    points_3d = rng.uniform(-50, 50, (100, 3))
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
