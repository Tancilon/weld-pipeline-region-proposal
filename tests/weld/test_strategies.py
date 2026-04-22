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
