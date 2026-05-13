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
