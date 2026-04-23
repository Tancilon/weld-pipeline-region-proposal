"""Tests for scripts/augment_dataset.py."""

import os

# EXR support must be enabled before cv2 is imported anywhere in the test process.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.augment_dataset import (
    mask_to_polygons,
    update_K_flip,
    update_K_crop,
    update_K_resize,
    update_K_translate,
)


def test_mask_to_polygons_simple_square():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 1
    polygons = mask_to_polygons(mask, min_area=50)
    assert len(polygons) == 1
    assert len(polygons[0]) >= 8

def test_mask_to_polygons_filters_small_fragments():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 1
    mask[0:3, 0:3] = 1
    polygons = mask_to_polygons(mask, min_area=50)
    assert len(polygons) == 1

def test_mask_to_polygons_empty_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    polygons = mask_to_polygons(mask, min_area=50)
    assert len(polygons) == 0


from scripts.augment_dataset import compute_bbox_area, is_augmentation_safe

def test_compute_bbox_area_simple():
    polygon = [10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]
    bbox, area = compute_bbox_area([polygon])
    assert bbox == [10.0, 10.0, 40.0, 40.0]
    assert area == 1600.0

def test_is_augmentation_safe_pass():
    assert is_augmentation_safe(500.0, 1000.0, min_ratio=0.1) is True

def test_is_augmentation_safe_fail():
    assert is_augmentation_safe(50.0, 1000.0, min_ratio=0.1) is False

def test_is_augmentation_safe_zero_area():
    assert is_augmentation_safe(0.0, 1000.0, min_ratio=0.1) is False


import albumentations as A
from scripts.augment_dataset import build_transform

def test_build_transform_returns_compose():
    transform = build_transform(height=1080, width=1920)
    assert isinstance(transform, A.Compose)

def test_build_transform_applies_to_image_and_mask():
    transform = build_transform(height=100, width=200)
    image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[25:75, 50:150] = 1
    result = transform(image=image, mask=mask)
    assert result['image'].shape == (100, 200, 3)
    assert result['mask'].shape == (100, 200)


from scripts.augment_dataset import augment_single_image

def test_augment_single_image_produces_valid_output():
    image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[25:75, 50:150] = 1
    original_area = float(mask.sum())
    transform = build_transform(height=100, width=200)
    result = augment_single_image(image, mask, original_area, transform, max_retries=5, min_area_ratio=0.1)
    assert result is not None
    aug_img, polygons, bbox, area = result
    assert aug_img.shape == (100, 200, 3)
    assert len(polygons) >= 1
    assert len(bbox) == 4
    assert area > 0

def test_augment_single_image_returns_none_on_impossible():
    image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[0, 0] = 1
    original_area = 5000.0
    transform = build_transform(height=100, width=200)
    result = augment_single_image(image, mask, original_area, transform, max_retries=3, min_area_ratio=0.1)
    assert result is None


def _K(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def test_update_K_flip():
    K = _K(100.0, 110.0, 30.0, 40.0)
    K2 = update_K_flip(K, width=100)
    assert K2[0, 0] == 100.0
    assert K2[1, 1] == 110.0
    assert K2[0, 2] == pytest.approx(100 - 1 - 30.0)  # W-1-cx
    assert K2[1, 2] == 40.0


def test_update_K_crop():
    K = _K(100.0, 110.0, 30.0, 40.0)
    K2 = update_K_crop(K, x0=5, y0=7)
    assert K2[0, 0] == 100.0
    assert K2[1, 1] == 110.0
    assert K2[0, 2] == pytest.approx(30.0 - 5)
    assert K2[1, 2] == pytest.approx(40.0 - 7)


def test_update_K_resize_half_pixel():
    K = _K(100.0, 100.0, 10.0, 20.0)
    # shrink 100->50, scale=0.5
    K2 = update_K_resize(K, src_w=100, src_h=100, dst_w=50, dst_h=50)
    assert K2[0, 0] == pytest.approx(50.0)
    assert K2[1, 1] == pytest.approx(50.0)
    # cx' = (cx + 0.5) * sx - 0.5 = (10.5) * 0.5 - 0.5 = 4.75
    assert K2[0, 2] == pytest.approx(4.75)
    # cy' = (20.5) * 0.5 - 0.5 = 9.75
    assert K2[1, 2] == pytest.approx(9.75)


def test_update_K_resize_identity():
    K = _K(100.0, 110.0, 30.0, 40.0)
    K2 = update_K_resize(K, src_w=100, src_h=100, dst_w=100, dst_h=100)
    np.testing.assert_allclose(K2, K)


def test_update_K_translate():
    K = _K(100.0, 110.0, 30.0, 40.0)
    K2 = update_K_translate(K, tx=3.0, ty=-2.0)
    assert K2[0, 0] == 100.0
    assert K2[1, 1] == 110.0
    assert K2[0, 2] == pytest.approx(33.0)
    assert K2[1, 2] == pytest.approx(38.0)


from scripts.augment_dataset import GeomParams, compose_K


def test_compose_K_flip_only():
    K = _K(100.0, 110.0, 30.0, 40.0)
    params = GeomParams(flip=True, crop_box=None, resize_to=None, translate=(0.0, 0.0))
    K2, W2, H2 = compose_K(K, width=100, height=80, params=params)
    assert W2 == 100 and H2 == 80
    assert K2[0, 2] == pytest.approx(99 - 30.0)


def test_compose_K_crop_then_resize_equivalent_to_stepwise():
    K = _K(100.0, 100.0, 30.0, 40.0)
    # Crop (x0=10, y0=5) to 80x70, then resize back to 100x100
    params = GeomParams(
        flip=False,
        crop_box=(10, 5, 80, 70),
        resize_to=(100, 100),
        translate=(0.0, 0.0),
    )
    K2, W2, H2 = compose_K(K, width=100, height=100, params=params)

    # Manual two-step
    K_cropped = update_K_crop(K, x0=10, y0=5)
    K_resized = update_K_resize(K_cropped, src_w=80, src_h=70, dst_w=100, dst_h=100)

    assert W2 == 100 and H2 == 100
    np.testing.assert_allclose(K2, K_resized)


def test_compose_K_all_ops():
    K = _K(100.0, 100.0, 30.0, 40.0)
    params = GeomParams(
        flip=True,
        crop_box=(5, 5, 90, 90),
        resize_to=(100, 100),
        translate=(2.0, -3.0),
    )
    K2, W2, H2 = compose_K(K, width=100, height=100, params=params)
    # Apply in canonical order: flip (cx := 99-30 = 69), crop (cx -= 5 -> 64, cy -= 5 -> 35),
    # resize 90->100 (sx=sy=10/9), cx' = (64+0.5)*10/9 - 0.5, then translate (+2, -3)
    expected_cx_after_crop = 64.0
    expected_cy_after_crop = 35.0
    sx = 100 / 90
    expected_cx_after_resize = (expected_cx_after_crop + 0.5) * sx - 0.5
    expected_cy_after_resize = (expected_cy_after_crop + 0.5) * sx - 0.5
    expected_cx = expected_cx_after_resize + 2.0
    expected_cy = expected_cy_after_resize - 3.0
    assert K2[0, 2] == pytest.approx(expected_cx)
    assert K2[1, 2] == pytest.approx(expected_cy)


import random

from scripts.augment_dataset import sample_geom_params


def test_sample_geom_params_reproducible():
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    p_a = sample_geom_params(width=1920, height=1080, rng=rng_a)
    p_b = sample_geom_params(width=1920, height=1080, rng=rng_b)
    assert p_a == p_b


def test_sample_geom_params_ranges():
    rng = random.Random(0)
    for _ in range(200):
        p = sample_geom_params(width=1920, height=1080, rng=rng)
        # translate within +-5%
        assert -1920 * 0.05 - 1 <= p.translate[0] <= 1920 * 0.05 + 1
        assert -1080 * 0.05 - 1 <= p.translate[1] <= 1080 * 0.05 + 1
        if p.crop_box is not None:
            x0, y0, w, h = p.crop_box
            # crop must stay inside source
            assert 0 <= x0 and x0 + w <= 1920
            assert 0 <= y0 and y0 + h <= 1080
            # scale in [0.8, 1.0]
            scale = (w * h) / (1920 * 1080)
            assert 0.8 - 1e-6 <= scale <= 1.0 + 1e-6
            # aspect ratio locked to W/H (within floor rounding error of 1 px)
            source_ar = 1920 / 1080
            cropped_ar = w / h
            assert abs(cropped_ar - source_ar) / source_ar < 0.002
        if p.resize_to is not None:
            # crop+resize always brings output back to source size
            assert p.resize_to == (1920, 1080)
