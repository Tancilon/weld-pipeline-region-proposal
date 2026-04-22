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


def _make_rounded_rect_mesh(half_w=50.0, half_h=50.0, corner_r=10.0, tube_r=1.0,
                            n_along=120, n_per_ring=6):
    """Make a mesh approximating a rounded-rectangle sweep."""
    # Build centerline of rounded rectangle (clockwise starting at top-left tangent)
    tl = (-half_w + corner_r, half_h - corner_r)
    tr = (half_w - corner_r, half_h - corner_r)
    br = (half_w - corner_r, -half_h + corner_r)
    bl = (-half_w + corner_r, -half_h + corner_r)
    # Total perimeter: 2(2(half_w-r) + 2(half_h-r)) + 4 * (π/2 r)
    #                = 4(half_w - r) + 4(half_h - r) + 2πr
    centerline_pts = []
    # Top edge: from (-half_w+r, half_h) to (half_w-r, half_h)
    n_top = max(2, int(n_along * (half_w - corner_r) / (half_w + half_h)))
    for i in range(n_top):
        t = i / n_top
        centerline_pts.append((tl[0] + 2*(half_w - corner_r)*t, half_h))
    # Top-right arc: 90° → 0°
    n_arc = max(2, n_along // 12)
    for i in range(n_arc):
        a = np.pi/2 - (np.pi/2) * (i / n_arc)
        centerline_pts.append((tr[0] + corner_r*np.cos(a), tr[1] + corner_r*np.sin(a)))
    # Right edge
    for i in range(n_top):
        t = i / n_top
        centerline_pts.append((half_w, tr[1] - 2*(half_h - corner_r)*t))
    # Bottom-right arc: 0° → -90°
    for i in range(n_arc):
        a = -(np.pi/2) * (i / n_arc)
        centerline_pts.append((br[0] + corner_r*np.cos(a), br[1] + corner_r*np.sin(a)))
    # Bottom edge
    for i in range(n_top):
        t = i / n_top
        centerline_pts.append((br[0] - 2*(half_w - corner_r)*t, -half_h))
    # Bottom-left arc: -90° → -180°
    for i in range(n_arc):
        a = -np.pi/2 - (np.pi/2) * (i / n_arc)
        centerline_pts.append((bl[0] + corner_r*np.cos(a), bl[1] + corner_r*np.sin(a)))
    # Left edge
    for i in range(n_top):
        t = i / n_top
        centerline_pts.append((-half_w, bl[1] + 2*(half_h - corner_r)*t))
    # Top-left arc: 180° → 90°
    for i in range(n_arc):
        a = np.pi - (np.pi/2) * (i / n_arc)
        centerline_pts.append((tl[0] + corner_r*np.cos(a), tl[1] + corner_r*np.sin(a)))
    centerline_pts = np.array(centerline_pts)

    # Build a thin planar ring: centerline replicated at z=0 and z=tube_r
    # with triangles connecting the two layers (like the real square_tube weld).
    n_cl = len(centerline_pts)
    verts = []
    for cx, cy in centerline_pts:
        verts.append([cx, cy, 0.0])
    for cx, cy in centerline_pts:
        verts.append([cx, cy, tube_r])
    faces = []
    for i in range(n_cl):
        i_next = (i + 1) % n_cl
        v0 = i
        v1 = i_next
        v2 = i + n_cl
        v3 = i_next + n_cl
        faces.append([v0, v1, v2])
        faces.append([v1, v3, v2])
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces))


def test_square_tube_strategy_fits_rounded_rect():
    from weld.strategies.square_tube import SquareTubeStrategy
    mesh = _make_rounded_rect_mesh(half_w=50.0, half_h=50.0, corner_r=10.0)
    paths = SquareTubeStrategy().process(mesh)
    assert len(paths) == 1
    path = paths[0]
    assert path["closed"] is True
    assert len(path["fitted"]) == 8
    types = [s["type"] for s in path["fitted"]]
    assert types.count("line") == 4
    assert types.count("arc") == 4
    # Alternating line/arc
    assert types == ["line", "arc", "line", "arc", "line", "arc", "line", "arc"]
    for seg in path["fitted"]:
        assert seg["fitting_error_mm"] < 3.0


def test_get_strategy_returns_square_tube():
    from weld.strategies.square_tube import SquareTubeStrategy
    assert isinstance(get_strategy("square_tube"), SquareTubeStrategy)


def test_cover_plate_strategy_has_force_close_true():
    from weld.strategies.cover_plate import CoverPlateStrategy
    assert CoverPlateStrategy.force_close is True
    from weld.strategies.base import GenericStrategy
    assert issubclass(CoverPlateStrategy, GenericStrategy)


def test_channel_steel_strategy_is_strategy_subclass():
    from weld.strategies.channel_steel import ChannelSteelStrategy
    from weld.strategies.base import Strategy
    assert issubclass(ChannelSteelStrategy, Strategy)


def test_h_beam_strategy_is_generic_subclass():
    from weld.strategies.h_beam import HBeamStrategy
    from weld.strategies.base import GenericStrategy
    assert issubclass(HBeamStrategy, GenericStrategy)


def test_registry_has_all_five_categories():
    from weld.strategies.bellmouth import BellmouthStrategy
    from weld.strategies.channel_steel import ChannelSteelStrategy
    from weld.strategies.h_beam import HBeamStrategy
    from weld.strategies.square_tube import SquareTubeStrategy
    from weld.strategies.cover_plate import CoverPlateStrategy

    assert isinstance(get_strategy("bellmouth"), BellmouthStrategy)
    assert isinstance(get_strategy("channel_steel"), ChannelSteelStrategy)
    assert isinstance(get_strategy("h_beam"), HBeamStrategy)
    assert isinstance(get_strategy("H_beam"), HBeamStrategy)  # legacy alias
    assert isinstance(get_strategy("square_tube"), SquareTubeStrategy)
    assert isinstance(get_strategy("cover_plate"), CoverPlateStrategy)
