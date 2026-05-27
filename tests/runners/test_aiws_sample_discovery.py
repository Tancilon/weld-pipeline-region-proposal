from __future__ import annotations

import sys
from pathlib import Path

REGION_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REGION_ROOT.parent
if str(REGION_ROOT) not in sys.path:
    sys.path.insert(0, str(REGION_ROOT))

from runners.aiws_sample_discovery import discover_rgbd_samples, prepare_intermediate_inputs


def test_discover_real_tmp_samples_finds_three_categories():
    samples = discover_rgbd_samples(WORKSPACE / "tmp")

    by_id = {sample.sample_id: sample for sample in samples}
    assert set(by_id) == {"0034", "0035", "0054", "0055", "0091", "0092"}
    assert by_id["0034"].workpiece_type == "square_tube"
    assert by_id["0054"].workpiece_type == "cover_plate"
    assert by_id["0091"].workpiece_type == "bellmouth"


def test_prepare_intermediate_inputs_copies_rgb_and_depth(tmp_path):
    source = tmp_path / "tmp" / "square_tube"
    source.mkdir(parents=True)
    rgb = source / "0034_color.png"
    depth = source / "0034_depth.exr"
    rgb.write_bytes(b"rgb")
    depth.write_bytes(b"depth")

    sample = discover_rgbd_samples(tmp_path / "tmp")[0]
    paths = prepare_intermediate_inputs(sample, tmp_path / "output")

    assert paths["rgb_path"].read_bytes() == b"rgb"
    assert paths["depth_path"].read_bytes() == b"depth"
    assert paths["part_masks_dir"].is_dir()
