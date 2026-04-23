"""Offline data augmentation for AIWS5.2 COCO segmentation dataset."""

import json
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# EXR support must be enabled before cv2 is imported.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import albumentations as A
from pycocotools.coco import COCO

logger = logging.getLogger(__name__)


def update_K_flip(K: np.ndarray, width: int) -> np.ndarray:
    """Horizontal flip: cx' = W - 1 - cx; fx/fy/cy unchanged."""
    K2 = K.copy()
    K2[0, 2] = (width - 1) - K[0, 2]
    return K2


def update_K_crop(K: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """Crop with top-left (x0, y0): cx -= x0, cy -= y0; fx/fy unchanged."""
    K2 = K.copy()
    K2[0, 2] = K[0, 2] - x0
    K2[1, 2] = K[1, 2] - y0
    return K2


def update_K_resize(K: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    """Resize src_w x src_h -> dst_w x dst_h. Uses pixel-center-aligned scaling."""
    sx = dst_w / src_w
    sy = dst_h / src_h
    K2 = K.copy()
    K2[0, 0] = K[0, 0] * sx
    K2[1, 1] = K[1, 1] * sy
    K2[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K2[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K2


def update_K_translate(K: np.ndarray, tx: float, ty: float) -> np.ndarray:
    """Pixel-space translate (tx right, ty down): cx += tx, cy += ty."""
    K2 = K.copy()
    K2[0, 2] = K[0, 2] + tx
    K2[1, 2] = K[1, 2] + ty
    return K2


@dataclass(frozen=True)
class GeomParams:
    """Parameters for one geometric-augmentation pass.

    All ops are applied in canonical order: flip -> crop -> resize -> translate.
    Any op can be disabled by passing the identity (flip=False, crop_box=None,
    resize_to=None, translate=(0.0, 0.0)).
    """
    flip: bool
    crop_box: Optional[Tuple[int, int, int, int]]  # (x0, y0, w, h) or None
    resize_to: Optional[Tuple[int, int]]           # (dst_w, dst_h) or None
    translate: Tuple[float, float]                  # (tx, ty) in pixels


def compose_K(
    K: np.ndarray, width: int, height: int, params: GeomParams,
) -> Tuple[np.ndarray, int, int]:
    """Apply params to K in canonical order and return (K', W', H')."""
    W, H = width, height
    if params.flip:
        K = update_K_flip(K, width=W)
    if params.crop_box is not None:
        x0, y0, w_crop, h_crop = params.crop_box
        K = update_K_crop(K, x0=x0, y0=y0)
        W, H = w_crop, h_crop
    if params.resize_to is not None:
        dst_w, dst_h = params.resize_to
        K = update_K_resize(K, src_w=W, src_h=H, dst_w=dst_w, dst_h=dst_h)
        W, H = dst_w, dst_h
    tx, ty = params.translate
    if tx != 0.0 or ty != 0.0:
        K = update_K_translate(K, tx=tx, ty=ty)
    return K, W, H


def read_source_meta(meta_dir, image_filename: str):
    """Load meta/<stem>.json; return (K, W, H, annotation_dict)."""
    stem = os.path.splitext(image_filename)[0]
    meta_path = Path(meta_dir) / f"{stem}.json"
    with open(meta_path) as f:
        d = json.load(f)
    i = d["camera"]["intrinsics"]
    K = np.array([
        [i["fx"], 0.0, i["cx"]],
        [0.0, i["fy"], i["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K, int(i["width"]), int(i["height"]), d["annotation"]


def write_augmented_meta(meta_dir, image_filename: str, K, W: int, H: int, annotation: dict) -> None:
    """Write meta/<stem>.json matching the source schema."""
    stem = os.path.splitext(image_filename)[0]
    out = {
        "camera": {"intrinsics": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "width": int(W),
            "height": int(H),
        }},
        "annotation": {
            "class_name": annotation["class_name"],
            "dimensions": list(annotation["dimensions"]),
        },
    }
    out_path = Path(meta_dir) / f"{stem}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def copy_dataset_files(input_dir, output_dir) -> None:
    """Copy images/, depth/, meta/ directories wholesale from input to output."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    for sub in ("images", "depth", "meta"):
        src_sub = input_dir / sub
        dst_sub = output_dir / sub
        dst_sub.mkdir(parents=True, exist_ok=True)
        if not src_sub.exists():
            continue
        for entry in src_sub.iterdir():
            if entry.is_file():
                shutil.copy2(entry, dst_sub / entry.name)


def sample_geom_params(width: int, height: int, rng: random.Random) -> GeomParams:
    """Sample a GeomParams instance using the probabilities and ranges from spec §5.1."""
    # HorizontalFlip p=0.5
    flip = rng.random() < 0.5

    # RandomResizedCrop p=0.5; scale in [0.8, 1.0], aspect ratio locked to source W/H
    crop_box = None
    resize_to = None
    if rng.random() < 0.5:
        scale = rng.uniform(0.8, 1.0)
        # Keep aspect ratio: area = scale * W * H, same AR => w_crop = sqrt(scale) * W, h_crop = sqrt(scale) * H
        w_crop = max(1, int(round(width * (scale ** 0.5))))
        h_crop = max(1, int(round(height * (scale ** 0.5))))
        w_crop = min(w_crop, width)
        h_crop = min(h_crop, height)
        x0 = rng.randint(0, width - w_crop)
        y0 = rng.randint(0, height - h_crop)
        crop_box = (x0, y0, w_crop, h_crop)
        resize_to = (width, height)  # resize back to source

    # Translate p=0.3; +/-5% on each axis independently
    tx = ty = 0.0
    if rng.random() < 0.3:
        tx = rng.uniform(-0.05, 0.05) * width
        ty = rng.uniform(-0.05, 0.05) * height

    return GeomParams(flip=flip, crop_box=crop_box, resize_to=resize_to, translate=(tx, ty))


def apply_geom(
    rgb: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    params: GeomParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply params to (rgb, mask, depth) consistently and propagate K.

    RGB uses linear interpolation, mask and depth use nearest neighbor
    (nearest prevents flying pixels at depth discontinuities). Borders
    revealed by translate fill with zero.
    """
    H, W = rgb.shape[:2]
    assert mask.shape == (H, W), f"mask shape {mask.shape} != rgb HxW {(H, W)}"
    assert depth.shape == (H, W), f"depth shape {depth.shape} != rgb HxW {(H, W)}"

    # Canonical order: flip -> crop -> resize -> translate
    if params.flip:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
        depth = np.ascontiguousarray(depth[:, ::-1])

    if params.crop_box is not None:
        x0, y0, w_crop, h_crop = params.crop_box
        rgb = rgb[y0:y0 + h_crop, x0:x0 + w_crop]
        mask = mask[y0:y0 + h_crop, x0:x0 + w_crop]
        depth = depth[y0:y0 + h_crop, x0:x0 + w_crop]

    if params.resize_to is not None:
        dst_w, dst_h = params.resize_to
        rgb = cv2.resize(rgb, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)

    tx, ty = params.translate
    if tx != 0.0 or ty != 0.0:
        H_cur, W_cur = rgb.shape[:2]
        M = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
        rgb = cv2.warpAffine(
            rgb, M, (W_cur, H_cur),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        mask = cv2.warpAffine(
            mask, M, (W_cur, H_cur),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        depth = cv2.warpAffine(
            depth, M, (W_cur, H_cur),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )

    K_out, _, _ = compose_K(K, width=W, height=H, params=params)
    return rgb, mask, depth, K_out


def mask_to_polygons(mask: np.ndarray, min_area: float = 50.0) -> list[list[float]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        poly = contour.astype(np.float32).reshape(-1).tolist()
        if len(poly) >= 6:
            polygons.append(poly)
    return polygons


def compute_bbox_area(polygons: list[list[float]]) -> tuple[list[float], float]:
    if not polygons:
        return [0.0, 0.0, 0.0, 0.0], 0.0
    all_x = []
    all_y = []
    total_area = 0.0
    for poly in polygons:
        xs = poly[0::2]
        ys = poly[1::2]
        all_x.extend(xs)
        all_y.extend(ys)
        n = len(xs)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += xs[i] * ys[j]
            area -= xs[j] * ys[i]
        total_area += abs(area) / 2.0
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    return bbox, total_area


def is_mask_safe(aug_area: float, original_area: float, min_ratio: float = 0.1) -> bool:
    if original_area <= 0 or aug_area <= 0:
        return False
    return (aug_area / original_area) >= min_ratio


def is_depth_safe(aug_depth: np.ndarray, original_depth: np.ndarray, min_ratio: float = 0.1) -> bool:
    """Require at least min_ratio * source-valid-pixel-count non-zero pixels in aug_depth."""
    src_valid = int(np.count_nonzero(original_depth))
    if src_valid == 0:
        return False
    aug_valid = int(np.count_nonzero(aug_depth))
    return (aug_valid / src_valid) >= min_ratio


def compute_quotas_balanced(coco, target: int, rng: random.Random) -> dict[int, int]:
    """Per-category balancing: each category's image pool gets base + random remainder.

    Categories with 0 samples or already >= target are skipped (no down-sampling).
    Returns a dict mapping image_id -> how many augmented copies to make from it.
    """
    quotas: dict[int, int] = {}
    for cat_id in coco.getCatIds():
        img_ids = list(set(coco.getImgIds(catIds=[cat_id])))
        count = len(img_ids)
        if count == 0:
            logger.warning(f"category {cat_id}: 0 samples, cannot augment, skipping")
            continue
        if count >= target:
            logger.info(f"category {cat_id}: {count} >= target {target}, skipping")
            continue
        need = target - count
        base = need // count
        remainder = need % count
        for img_id in img_ids:
            quotas[img_id] = quotas.get(img_id, 0) + base
        for img_id in rng.sample(img_ids, remainder):
            quotas[img_id] = quotas.get(img_id, 0) + 1
    return quotas


def compute_quotas_uniform(coco, num_aug: int) -> dict[int, int]:
    """Legacy mode: every image gets num_aug augmentations."""
    return {img_id: num_aug for img_id in coco.getImgIds()}


def build_color_transform() -> A.Compose:
    """RGB-only color augmentations; no geometric ops (those are hand-rolled)."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
        A.GaussNoise(std_range=(0.02, 0.05), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    ])


def augment_single_image(
    image: np.ndarray,
    mask: np.ndarray,
    original_area: float,
    transform: A.Compose,
    max_retries: int = 5,
    min_area_ratio: float = 0.1,
) -> tuple[np.ndarray, list[list[float]], list[float], float] | None:
    for _ in range(max_retries):
        result = transform(image=image, mask=mask)
        aug_image = result['image']
        aug_mask = result['mask']
        polygons = mask_to_polygons(aug_mask, min_area=50.0)
        if not polygons:
            continue
        bbox, area = compute_bbox_area(polygons)
        if is_mask_safe(area, original_area, min_ratio=min_area_ratio):
            return aug_image, polygons, bbox, area
    return None


def augment_split(
    input_dir: str,
    output_dir: str,
    split: str,
    num_aug: int,
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    ann_path = os.path.join(input_dir, 'annotations', f'{split}.json')
    with open(ann_path) as f:
        coco_data = json.load(f)

    coco = COCO(ann_path)

    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    # Copy original images
    for img_info in coco_data['images']:
        src = os.path.join(input_dir, 'images', img_info['file_name'])
        dst = os.path.join(output_dir, 'images', img_info['file_name'])
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Determine image dimensions from first image
    first_img = coco_data['images'][0]
    height, width = first_img['height'], first_img['width']
    transform = build_color_transform()

    aug_images = []
    aug_annotations = []
    aug_img_id = 100000
    aug_ann_id = 100000
    aug_file_counter = 0

    total = len(coco_data['images'])
    for idx, img_info in enumerate(coco_data['images']):
        img_id = img_info['id']
        img_path = os.path.join(input_dir, 'images', img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"Could not read image: {img_path}, skipping.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotation for this image (exactly one per image)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        if not ann_ids:
            logger.warning(f"No annotation for image_id={img_id}, skipping.")
            continue
        ann = coco.loadAnns(ann_ids)[0]
        mask = coco.annToMask(ann)
        original_area = ann['area']

        for aug_idx in range(num_aug):
            result = augment_single_image(
                image, mask, original_area, transform,
                max_retries=5, min_area_ratio=0.1,
            )
            if result is None:
                logger.warning(
                    f"[{split}] Failed to augment image_id={img_id} "
                    f"(slot {aug_idx+1}/{num_aug}) after max retries, skipping."
                )
                continue

            aug_image, polygons, bbox, area = result

            # Save augmented image
            aug_filename = f"AUG_{split}_{aug_file_counter:04d}.png"
            aug_file_counter += 1
            aug_path = os.path.join(output_dir, 'images', aug_filename)
            cv2.imwrite(aug_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

            # Create image entry
            aug_images.append({
                'license': img_info.get('license', ''),
                'url': '',
                'file_name': aug_filename,
                'height': height,
                'width': width,
                'date_captured': '',
                'id': aug_img_id,
            })

            # Create annotation entry
            aug_annotations.append({
                'iscrowd': False,
                'image_id': aug_img_id,
                'image_name': aug_filename,
                'category_id': ann['category_id'],
                'id': aug_ann_id,
                'segmentation': polygons,
                'area': area,
                'bbox': bbox,
                'source_image_id': img_id,
            })

            aug_img_id += 1
            aug_ann_id += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            logger.info(f"[{split}] Processed {idx+1}/{total} images, "
                        f"generated {aug_file_counter} augmented images so far.")

    # Merge original + augmented
    merged = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': coco_data['images'] + aug_images,
        'annotations': coco_data['annotations'] + aug_annotations,
    }

    out_ann_path = os.path.join(output_dir, 'annotations', f'{split}.json')
    with open(out_ann_path, 'w') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[{split}] Done. Original: {len(coco_data['images'])}, "
        f"Augmented: {len(aug_images)}, "
        f"Total: {len(merged['images'])} images."
    )


def _sanitize_depth(depth: np.ndarray, max_meters: float = 4.0) -> np.ndarray:
    """Match datasets_nuclear.py:186-191: NaN/Inf/<0/>max_meters -> 0."""
    invalid = ~np.isfinite(depth)
    invalid |= depth < 0
    invalid |= depth > max_meters
    if np.any(invalid):
        depth = depth.copy()
        depth[invalid] = 0
    return depth


def _try_generate_augmented_sample(
    rgb, mask, depth, K, W, H, original_area,
    color_transform, rng: random.Random, max_retries: int,
):
    """Sample GeomParams, apply full pipeline, check safety, retry on failure.

    Returns the augmented quadruple + derived polygons/bbox/area, or None if
    every retry fails.
    """
    for _ in range(max_retries):
        params = sample_geom_params(width=W, height=H, rng=rng)
        rgb_g, mask_g, depth_g, K_g = apply_geom(rgb, mask, depth, K, params)
        # Color jitter (RGB only)
        rgb_g = color_transform(image=rgb_g)["image"]
        # Re-sanitize depth after geometric ops
        depth_g = _sanitize_depth(depth_g)

        polygons = mask_to_polygons(mask_g, min_area=50.0)
        if not polygons:
            continue
        bbox, area = compute_bbox_area(polygons)
        if not is_mask_safe(aug_area=area, original_area=original_area, min_ratio=0.1):
            continue
        if not is_depth_safe(aug_depth=depth_g, original_depth=depth, min_ratio=0.1):
            continue
        H_out, W_out = rgb_g.shape[:2]
        return rgb_g, mask_g, depth_g, K_g, W_out, H_out, polygons, bbox, area
    return None


def augment_train_split(
    input_dir,
    output_dir,
    quotas: dict,
    seed: int = 42,
) -> None:
    """Augment the train split of a COCO dataset per pre-computed quotas.

    - Copies images/, depth/, meta/ and val/test annotations to output_dir.
    - For each source train image with quota > 0, generates that many augmented
      quadruples (RGB, mask, depth, meta-K), appending COCO entries.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError(
            f"input_dir and output_dir must differ; got {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)

    # 1) Copy images/, depth/, meta/ wholesale
    copy_dataset_files(input_dir, output_dir)

    # 2) Copy val/test annotations as-is
    for split in ("val", "test"):
        src = input_dir / "annotations" / f"{split}.json"
        if src.is_file():
            shutil.copy2(src, output_dir / "annotations" / f"{split}.json")

    # 3) Process train split
    train_ann = input_dir / "annotations" / "train.json"
    with open(train_ann) as f:
        coco_data = json.load(f)
    coco = COCO(str(train_ann))

    color_transform = build_color_transform()
    rng = random.Random(seed)
    np.random.seed(seed)

    aug_images = []
    aug_annotations = []
    aug_img_id = 100000
    aug_ann_id = 100000
    aug_file_counter = 0

    total_quota = sum(quotas.values())
    produced = 0
    logger.info(f"train split: {len(coco_data['images'])} source images, total quota {total_quota}")

    for img_info in coco_data["images"]:
        img_id = img_info["id"]
        quota = quotas.get(img_id, 0)
        if quota == 0:
            continue

        # Load source RGB + mask + depth + K + annotation (meta)
        fname = img_info["file_name"]
        stem = os.path.splitext(fname)[0]
        rgb_path = input_dir / "images" / fname
        depth_path = input_dir / "depth" / f"{stem}.exr"
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            logger.warning(f"cannot read {rgb_path}, skipping")
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            logger.warning(f"cannot read {depth_path}, skipping")
            continue
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = depth.astype(np.float32, copy=False)
        depth = _sanitize_depth(depth)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        if not ann_ids:
            logger.warning(f"no annotation for image_id={img_id}, skipping")
            continue
        ann = coco.loadAnns(ann_ids)[0]
        mask = coco.annToMask(ann).astype(np.uint8)
        original_area = ann["area"]

        try:
            K, W, H, meta_annotation = read_source_meta(input_dir / "meta", fname)
        except FileNotFoundError:
            logger.warning(f"no meta for {fname}, skipping")
            continue

        for slot in range(quota):
            out_sample = _try_generate_augmented_sample(
                rgb, mask, depth, K, W, H, original_area,
                color_transform, rng, max_retries=5,
            )
            if out_sample is None:
                logger.warning(
                    f"image_id={img_id} slot {slot+1}/{quota}: all retries failed; dropping"
                )
                continue
            aug_rgb, aug_mask, aug_depth, aug_K, aug_W, aug_H, polygons, bbox, area = out_sample

            aug_filename = f"AUG_train_{aug_file_counter:04d}.png"
            aug_file_counter += 1

            # Save RGB
            cv2.imwrite(
                str(output_dir / "images" / aug_filename),
                cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR),
            )
            # Save depth
            cv2.imwrite(
                str(output_dir / "depth" / f"{os.path.splitext(aug_filename)[0]}.exr"),
                aug_depth,
            )
            # Save meta
            write_augmented_meta(
                output_dir / "meta", aug_filename,
                K=aug_K, W=aug_W, H=aug_H, annotation=meta_annotation,
            )

            aug_images.append({
                "license": img_info.get("license", ""),
                "url": "",
                "file_name": aug_filename,
                "height": int(aug_H),
                "width": int(aug_W),
                "date_captured": "",
                "id": aug_img_id,
            })
            aug_annotations.append({
                "iscrowd": False,
                "image_id": aug_img_id,
                "image_name": aug_filename,
                "category_id": ann["category_id"],
                "id": aug_ann_id,
                "segmentation": polygons,
                "area": area,
                "bbox": bbox,
                "source_image_id": img_id,
            })
            aug_img_id += 1
            aug_ann_id += 1
            produced += 1

    # Merge and write train.json
    merged = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data["categories"],
        "images": coco_data["images"] + aug_images,
        "annotations": coco_data["annotations"] + aug_annotations,
    }
    with open(output_dir / "annotations" / "train.json", "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    logger.info(
        f"train split done. Originals: {len(coco_data['images'])}, "
        f"augmented: {produced}/{total_quota}, total: {len(merged['images'])}."
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Offline data augmentation for COCO segmentation datasets."
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Path to input dataset (must contain annotations/ and images/).',
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Path to output augmented dataset.',
    )
    parser.add_argument(
        '--num_aug', type=int, default=3,
        help='Number of augmented images per original image (default: 3).',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42).',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    for split in ['train', 'val']:
        ann_path = os.path.join(args.input_dir, 'annotations', f'{split}.json')
        if not os.path.exists(ann_path):
            logger.warning(f"Annotation file not found: {ann_path}, skipping {split}.")
            continue
        augment_split(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            split=split,
            num_aug=args.num_aug,
            seed=args.seed,
        )

    logger.info("Augmentation complete.")


if __name__ == '__main__':
    main()
