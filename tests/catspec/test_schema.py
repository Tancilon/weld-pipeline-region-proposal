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


def test_load_catspec_rejects_missing_corner_radius_source(tmp_path):
    spec_path = tmp_path / "bad.yaml"
    spec_path.write_text(
        VALID_YAML.replace("        corner_radius_source: estimate_from_workpiece_mesh\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(CatSpecError, match=r"welds\[0\]\.locus\.params\.corner_radius_source"):
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
