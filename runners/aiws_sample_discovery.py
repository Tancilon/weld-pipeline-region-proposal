from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RGBDSample:
    sample_id: str
    workpiece_type: str
    rgb_path: Path
    depth_path: Path


def discover_rgbd_samples(input_root: str | Path) -> list[RGBDSample]:
    input_root = Path(input_root)
    samples: list[RGBDSample] = []
    for rgb_path in sorted(input_root.glob("*/*_color.png")):
        sample_id = rgb_path.name.removesuffix("_color.png")
        depth_path = rgb_path.with_name(f"{sample_id}_depth.exr")
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth for sample {sample_id}: {depth_path}")
        samples.append(
            RGBDSample(
                sample_id=sample_id,
                workpiece_type=rgb_path.parent.name,
                rgb_path=rgb_path,
                depth_path=depth_path,
            )
        )
    return samples


def prepare_intermediate_inputs(
    sample: RGBDSample, output_root: str | Path
) -> dict[str, Path]:
    sample_dir = Path(output_root) / sample.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    part_masks_dir = sample_dir / "part_masks"
    part_masks_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = sample_dir / "rgb.png"
    depth_path = sample_dir / "depth.exr"
    shutil.copy2(sample.rgb_path, rgb_path)
    shutil.copy2(sample.depth_path, depth_path)
    return {
        "sample_dir": sample_dir,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "part_masks_dir": part_masks_dir,
    }
