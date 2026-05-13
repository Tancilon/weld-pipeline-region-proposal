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


@pytest.mark.parametrize(
    ("spec_name", "category", "locus_type"),
    [
        ("channel_steel.yaml", "channel_steel", "open_line_arc_line_arc_line"),
        ("H_beam.yaml", "H_beam", "open_line_arc_line_arc_line"),
        ("bellmouth.yaml", "bellmouth", "parallel_open_lines"),
    ],
)
def test_load_catspec_accepts_static_open_profile_categories(spec_name, category, locus_type):
    spec = load_catspec(Path("specs/categories") / spec_name)

    assert spec["schema_version"] == "catspec.v0"
    assert spec["category"] == category
    assert spec["welds"][0]["locus"]["type"] == locus_type
    assert spec["welds"][0]["locus"]["params"]["path_count"] >= 1


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


def test_load_catspec_rejects_missing_corner_radius_source(tmp_path):
    spec_path = tmp_path / "bad.yaml"
    spec_path.write_text(
        VALID_YAML.replace("        corner_radius_source: estimate_from_workpiece_mesh\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(CatSpecError, match=r"welds\[0\]\.locus\.params\.corner_radius_source"):
        load_catspec(spec_path)


def test_load_catspec_rejects_open_profile_missing_plane_values(tmp_path):
    spec_path = tmp_path / "bad.yaml"
    spec_path.write_text(
        """\
schema_version: catspec.v0
category: channel_steel
units: meter
provenance:
  source_mesh: mesh.obj
  source_weld_mesh: weld.obj
parts:
  - id: channel_body
    primitive: channel_steel
welds:
  - id: inner_open_profile
    parts: [channel_body]
    locus:
      type: open_line_arc_line_arc_line
      source: analytic_from_profile
      frame: canonical_profile
      params:
        plane_axis: x
        profile_axes: [y, z]
        path_count: 2
        sample_points_per_segment: 16
    weld_meta:
      weld_type_prior: fillet
      torch_constraints: default_single_pass
      is_load_bearing: true
      confidence: medium
""",
        encoding="utf-8",
    )

    with pytest.raises(CatSpecError, match=r"welds\[0\]\.locus\.params\.plane_values"):
        load_catspec(spec_path)


def test_cover_plate_schema_gap_is_documented():
    gap_doc = Path("docs/catspec_v0_2_asset_review.md")

    assert gap_doc.exists()
    text = gap_doc.read_text(encoding="utf-8")
    assert "cover_plate" in text
    assert "schema gap" in text
    assert "closed freeform" in text


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


def test_resolve_asset_path_uses_absolute_spec_path_when_cwd_is_elsewhere(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    sibling = tmp_path / "datasets" / "obj_share_models"
    sibling.mkdir(parents=True)
    asset = sibling / "mesh.obj"
    asset.write_text("o mesh\n", encoding="utf-8")
    spec_dir = repo_root / "specs" / "categories"
    spec_dir.mkdir(parents=True)
    spec_path = spec_dir / "spec.yaml"
    outside = tmp_path / "unrelated" / "outside" / "nested"
    outside.mkdir(parents=True)
    monkeypatch.chdir(outside)

    resolved = resolve_asset_path("../datasets/obj_share_models/mesh.obj", spec_path=spec_path.resolve())

    assert resolved == asset.resolve()
