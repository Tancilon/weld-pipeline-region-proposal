"""Offline data augmentation for AIWS5.2 COCO segmentation dataset."""

import json
import logging
import os
import shutil
from dataclasses import dataclass
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


def is_augmentation_safe(aug_area: float, original_area: float, min_ratio: float = 0.1) -> bool:
    if original_area <= 0 or aug_area <= 0:
        return False
    return (aug_area / original_area) >= min_ratio


def build_transform(height: int, width: int) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.4),
        A.RandomResizedCrop(
            size=(height, width), scale=(0.8, 1.0),
            ratio=(width / height, width / height), p=0.4,
        ),
        A.Affine(
            translate_percent=(-0.05, 0.05), shear=(-5, 5),
            border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.3,
        ),
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
        if is_augmentation_safe(area, original_area, min_ratio=min_area_ratio):
            return aug_image, polygons, bbox, area
    return None


def augment_split(
    input_dir: str,
    output_dir: str,
    split: str,
    num_aug: int,
    seed: int,
) -> None:
    import random
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
    transform = build_transform(height=height, width=width)

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
