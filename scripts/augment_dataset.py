"""Offline data augmentation for AIWS5.2 COCO segmentation dataset."""

import cv2
import numpy as np
import albumentations as A


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
