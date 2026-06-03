#!/usr/bin/env python
"""Run Semantic-SAM automatic mask generation on one RGB image."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

if not hasattr(Image, "LINEAR"):
    Image.LINEAR = Image.BILINEAR

from semantic_sam.build_semantic_sam import build_semantic_sam, prepare_image
from tasks.automatic_mask_generator import SemanticSamAutomaticMaskGenerator
from tasks.interactive_idino_m2m_auto import show_anns


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_CKPTS = {
    "T": REPO_DIR / "ckpts" / "swint_only_sam_many2many.pth",
    "L": REPO_DIR / "ckpts" / "swinl_only_sam_many2many.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Semantic-SAM masks for one RGB image and write results "
            "to Semantic-SAM/vis by default."
        )
    )
    parser.add_argument("image", help="Path to the input RGB image.")
    parser.add_argument(
        "--model-type",
        choices=("T", "L"),
        default="T",
        help="Backbone/model size: T uses SwinT, L uses SwinL. Default: T.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint path. Defaults to ckpts/swint... or ckpts/swinl... by model type.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[4, 5, 6],
        help="Semantic-SAM granularity levels to run. Default: 4 5 6 for part-level masks.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_DIR / "vis"),
        help="Directory for output files. Default: Semantic-SAM/vis.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Filename prefix. Default: <image_stem>_levels_<levels>.",
    )
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=200)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--min-mask-region-area", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="Color seed for the overlay.")
    return parser.parse_args()


def resolve_existing_path(raw_path: str, bases: Iterable[Path]) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve(strict=True)

    checked: List[Path] = []
    for base in bases:
        candidate = (base / path).resolve()
        checked.append(candidate)
        if candidate.exists():
            return candidate

    checked_text = "\n  ".join(str(item) for item in checked)
    raise FileNotFoundError(f"Could not find {raw_path!r}. Checked:\n  {checked_text}")


def validate_levels(levels: List[int]) -> List[int]:
    invalid = [level for level in levels if level < 1 or level > 6]
    if invalid:
        raise ValueError(f"Levels must be in [1, 6], got: {invalid}")
    return levels


def make_prefix(image_path: Path, levels: List[int], explicit_prefix: str = None) -> str:
    if explicit_prefix:
        return explicit_prefix
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem).strip("._")
    if not safe_stem:
        safe_stem = "image"
    levels_label = "".join(str(level) for level in levels)
    return f"{safe_stem}_levels_{levels_label}"


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def save_overlay(image: np.ndarray, masks: List[Dict[str, Any]], output_path: Path, seed: int) -> None:
    height, width = image.shape[:2]
    dpi = 100

    np.random.seed(seed)
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image)
    plt.sca(ax)
    show_anns(masks)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_mask_pngs(
    masks: List[Dict[str, Any]],
    masks_dir: Path,
    prefix: str,
) -> List[Dict[str, Any]]:
    masks_dir.mkdir(parents=True, exist_ok=True)
    metadata: List[Dict[str, Any]] = []

    for index, ann in enumerate(masks):
        mask = np.asarray(ann["segmentation"]).astype(np.uint8) * 255
        mask_name = f"{prefix}_mask_{index:03d}.png"
        mask_path = masks_dir / mask_name
        Image.fromarray(mask, mode="L").save(mask_path)

        record = {
            key: to_builtin(value)
            for key, value in ann.items()
            if key != "segmentation"
        }
        record["mask_index"] = index
        record["mask_path"] = str(mask_path)
        metadata.append(record)

    return metadata


def main() -> None:
    args = parse_args()
    initial_cwd = Path.cwd()

    levels = validate_levels(args.levels)
    image_path = resolve_existing_path(args.image, bases=[initial_cwd, REPO_DIR])
    ckpt_path = (
        resolve_existing_path(args.ckpt, bases=[initial_cwd, REPO_DIR])
        if args.ckpt
        else DEFAULT_CKPTS[args.model_type].resolve(strict=True)
    )
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (initial_cwd / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = make_prefix(image_path, levels, args.output_prefix)
    input_path = output_dir / f"{prefix}_input.png"
    overlay_path = output_dir / f"{prefix}_overlay.png"
    metadata_path = output_dir / f"{prefix}_metadata.json"
    masks_dir = output_dir / f"{prefix}_masks"

    if not torch.cuda.is_available():
        raise RuntimeError("Semantic-SAM in this repository expects CUDA, but torch.cuda is not available.")

    os.chdir(REPO_DIR)
    original_image, image_tensor = prepare_image(str(image_path))
    model = build_semantic_sam(args.model_type, str(ckpt_path))
    mask_generator = SemanticSamAutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
        level=levels,
    )
    masks = mask_generator.generate(image_tensor)

    Image.fromarray(original_image).save(input_path)
    save_overlay(original_image, masks, overlay_path, args.seed)
    mask_metadata = save_mask_pngs(masks, masks_dir, prefix)

    height, width = original_image.shape[:2]
    metadata = {
        "image": str(image_path),
        "model_type": args.model_type,
        "checkpoint": str(ckpt_path),
        "levels": levels,
        "input_size": {"height": int(height), "width": int(width)},
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "min_mask_region_area": args.min_mask_region_area,
        "mask_count": len(masks),
        "input_path": str(input_path),
        "overlay_path": str(overlay_path),
        "masks_dir": str(masks_dir),
        "masks": mask_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Generated {len(masks)} masks")
    print(f"Input image: {input_path}")
    print(f"Overlay: {overlay_path}")
    print(f"Masks: {masks_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
