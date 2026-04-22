import numpy as np
import trimesh

from weld.strategies import get_strategy
from weld.strategies.base import Strategy, GenericStrategy


def _make_tube_mesh(n_rings=20, n_per_ring=8, radius=2.0, length=100.0):
    verts = []
    faces = []
    for i in range(n_rings):
        x = length * i / (n_rings - 1)
        for j in range(n_per_ring):
            angle = 2 * np.pi * j / n_per_ring
            verts.append([x, radius * np.cos(angle), radius * np.sin(angle)])
    verts = np.array(verts)
    for i in range(n_rings - 1):
        for j in range(n_per_ring):
            jn = (j + 1) % n_per_ring
            v0 = i * n_per_ring + j
            v1 = i * n_per_ring + jn
            v2 = (i + 1) * n_per_ring + j
            v3 = (i + 1) * n_per_ring + jn
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces))


def test_strategy_base_is_abstract():
    s = Strategy()
    try:
        s.process(None)
    except NotImplementedError:
        return
    raise AssertionError("Strategy.process should raise NotImplementedError")


def test_generic_strategy_processes_mesh():
    mesh = _make_tube_mesh()
    result = GenericStrategy().process(mesh)
    assert isinstance(result, list)
    assert len(result) == 1
    path = result[0]
    for key in ("centerline_2d", "plane", "fitted", "closed"):
        assert key in path


def test_generic_strategy_force_close_attribute_default_false():
    assert GenericStrategy.force_close is False


def test_get_strategy_unknown_returns_generic():
    assert isinstance(get_strategy("not_a_category"), GenericStrategy)


def test_get_strategy_none_returns_generic():
    assert isinstance(get_strategy(None), GenericStrategy)


import json
from pathlib import Path


def _make_two_line_obj(tmp_path):
    verts = []
    faces = []
    n_rings_a = 20
    n_per_ring = 8
    tube_r = 2.0
    for i in range(n_rings_a):
        x = 100 * i / (n_rings_a - 1)
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
    offset_a = n_rings_a * n_per_ring
    for i in range(n_rings_a):
        x = 100 * i / (n_rings_a - 1)
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
    obj_path = tmp_path / "weld.obj"
    obj_path.write_text("\n".join(lines))
    wp_path = tmp_path / "dual.obj"
    wp_path.write_text("o dummy\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    return str(wp_path), str(obj_path)


def test_run_pipeline_uses_generic_when_no_category(tmp_path):
    from weld.pipeline import run_pipeline
    wp, weld = _make_two_line_obj(tmp_path)
    output = tmp_path / "out.json"
    run_pipeline(wp, weld, str(output), no_viz=True, category=None)
    data = json.loads(Path(output).read_text())
    assert "weld_paths" in data
    assert len(data["weld_paths"]) == 2


def test_bellmouth_strategy_fits_single_line_per_component():
    from weld.strategies.bellmouth import BellmouthStrategy
    m1 = _make_tube_mesh()
    v2 = m1.vertices.copy()
    v2[:, 1] += 30
    mesh = trimesh.Trimesh(
        vertices=np.vstack([m1.vertices, v2]),
        faces=np.vstack([m1.faces, m1.faces + len(m1.vertices)]),
    )
    paths = BellmouthStrategy().process(mesh)
    assert len(paths) == 2
    for p in paths:
        assert p["closed"] is False
        assert len(p["fitted"]) == 1
        assert p["fitted"][0]["type"] == "line"
        # Straight tube of length 100 should fit with near-zero error
        assert p["fitted"][0]["fitting_error_mm"] < 1.0


def test_get_strategy_returns_bellmouth():
    from weld.strategies.bellmouth import BellmouthStrategy
    assert isinstance(get_strategy("bellmouth"), BellmouthStrategy)
