from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REGION_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REGION_ROOT.parent
if str(REGION_ROOT) not in sys.path:
    sys.path.insert(0, str(REGION_ROOT))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from components.aiws_pipeline_contracts import validate_region_proposal
from components.workpiece_priors import WorkpiecePriorRegistry
from runners.aiws_region_proposal_contract import (
    genpose_result_to_part_payload,
    write_region_proposal,
)


def test_genpose_result_to_part_payload_converts_meters_to_mm():
    registry = WorkpiecePriorRegistry(
        WORKSPACE / "workpiece_priors/workpiece_info.yaml", repo_root=WORKSPACE
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [0.1, 0.2, 0.3]

    payload = genpose_result_to_part_payload(
        registry=registry,
        category="bellmouth",
        part_name="tube",
        mask_path=Path("output/0091/part_masks/tube.png"),
        genpose_result={
            "R": pose[:3, :3],
            "t": pose[:3, 3],
            "size": np.array([0.147, 0.199, 0.149], dtype=np.float64),
            "size_source": "scale_net",
        },
    )

    assert payload["raw_size_xyz_m"] == [0.147, 0.199, 0.149]
    assert payload["raw_size_xyz_mm"] == [147.0, 199.0, 149.0]
    assert payload["matched_size_xyz_mm"] == [148.0, 200.0, 148.0]
    assert payload["coarse_pose_source"] == "genpose2"


def test_write_region_proposal_writes_valid_json(tmp_path):
    registry = WorkpiecePriorRegistry(
        WORKSPACE / "workpiece_priors/workpiece_info.yaml", repo_root=WORKSPACE
    )
    part_payload = genpose_result_to_part_payload(
        registry=registry,
        category="square_tube",
        part_name="tube",
        mask_path=tmp_path / "part_masks/tube.png",
        genpose_result={
            "R": np.eye(3, dtype=np.float64),
            "t": np.array([0.1, 0.2, 0.3], dtype=np.float64),
            "size": np.array([0.121, 0.201, 0.119], dtype=np.float64),
            "size_source": "scale_net",
        },
    )

    output_path = write_region_proposal(
        output_dir=tmp_path,
        sample_id="0034",
        workpiece_type="square_tube",
        camera_path=Path("workpiece_priors/camera.json"),
        rgb_path=tmp_path / "rgb.png",
        depth_path=tmp_path / "depth.exr",
        object_mask_path=tmp_path / "object_mask.png",
        focused_parts={"tube": part_payload},
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    validate_region_proposal(payload)
    assert payload["focused_parts"]["tube"]["size_source"] == "scale_net"
