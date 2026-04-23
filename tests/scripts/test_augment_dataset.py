"""Tests for scripts/augment_dataset.py."""

import json
import os

# EXR support must be enabled before cv2 is imported anywhere in the test process.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
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


from scripts.augment_dataset import compute_bbox_area, is_mask_safe

def test_compute_bbox_area_simple():
    polygon = [10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]
    bbox, area = compute_bbox_area([polygon])
    assert bbox == [10.0, 10.0, 40.0, 40.0]
    assert area == 1600.0

def test_is_mask_safe_pass():
    assert is_mask_safe(500.0, 1000.0, min_ratio=0.1) is True

def test_is_mask_safe_fail():
    assert is_mask_safe(50.0, 1000.0, min_ratio=0.1) is False

def test_is_mask_safe_zero_area():
    assert is_mask_safe(0.0, 1000.0, min_ratio=0.1) is False


import albumentations as A
from scripts.augment_dataset import build_color_transform

def test_build_color_transform_returns_compose():
    transform = build_color_transform()
    assert isinstance(transform, A.Compose)

def test_build_color_transform_applies_to_image():
    transform = build_color_transform()
    image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    result = transform(image=image)
    assert result['image'].shape == (100, 200, 3)


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


from scripts.augment_dataset import apply_geom


def _synthetic_sample(W=32, H=24):
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[:, :W // 2] = 200  # left half bright, right half dark
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1
    depth = np.full((H, W), 1.5, dtype=np.float32)
    depth[:, W // 2:] = 2.5  # left 1.5m, right 2.5m
    K = _K(fx=20.0, fy=20.0, cx=W / 2 - 0.5, cy=H / 2 - 0.5)
    return rgb, mask, depth, K


def test_apply_geom_identity():
    rgb, mask, depth, K = _synthetic_sample()
    params = GeomParams(flip=False, crop_box=None, resize_to=None, translate=(0.0, 0.0))
    rgb2, mask2, depth2, K2 = apply_geom(rgb, mask, depth, K, params)
    np.testing.assert_array_equal(rgb2, rgb)
    np.testing.assert_array_equal(mask2, mask)
    np.testing.assert_allclose(depth2, depth)
    np.testing.assert_allclose(K2, K)


def test_apply_geom_flip_swaps_bright_half():
    rgb, mask, depth, K = _synthetic_sample()
    params = GeomParams(flip=True, crop_box=None, resize_to=None, translate=(0.0, 0.0))
    rgb2, mask2, depth2, K2 = apply_geom(rgb, mask, depth, K, params)
    # After flip, bright half is now on the right
    assert rgb2[:, -1, 0].mean() > 100
    assert rgb2[:, 0, 0].mean() < 50
    # Depth on the left was 2.5 (it was the right half originally)
    assert depth2[:, 0].mean() == pytest.approx(2.5)
    assert depth2[:, -1].mean() == pytest.approx(1.5)


def test_apply_geom_crop_resize_preserves_shape():
    rgb, mask, depth, K = _synthetic_sample()
    params = GeomParams(
        flip=False, crop_box=(2, 2, 28, 20), resize_to=(32, 24), translate=(0.0, 0.0)
    )
    rgb2, mask2, depth2, K2 = apply_geom(rgb, mask, depth, K, params)
    assert rgb2.shape == rgb.shape
    assert mask2.shape == mask.shape
    assert depth2.shape == depth.shape


def test_apply_geom_translate_fills_zero_borders():
    rgb, mask, depth, K = _synthetic_sample()
    params = GeomParams(
        flip=False, crop_box=None, resize_to=None, translate=(3.0, 0.0)
    )
    rgb2, mask2, depth2, K2 = apply_geom(rgb, mask, depth, K, params)
    # First 3 columns should be zeros on all three channels
    assert rgb2[:, :3].sum() == 0
    assert mask2[:, :3].sum() == 0
    assert (depth2[:, :3] == 0).all()


def _unproject(K, pixel_xy, depth_map):
    """Unproject pixel (x, y) to 3D camera frame using depth[y, x]. Returns (3,) or None if invalid."""
    x, y = pixel_xy
    if not (0 <= int(round(y)) < depth_map.shape[0] and 0 <= int(round(x)) < depth_map.shape[1]):
        return None
    z = float(depth_map[int(round(y)), int(round(x))])
    if z <= 0:
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    return np.array([X, Y, z], dtype=np.float64)


def _apply_params_to_pixel(W, H, params, x, y):
    """Map a source pixel (x, y) to its post-aug location under params."""
    if params.flip:
        x = (W - 1) - x
    if params.crop_box is not None:
        x0, y0, _, _ = params.crop_box
        x = x - x0
        y = y - y0
        W = params.crop_box[2]
        H = params.crop_box[3]
    if params.resize_to is not None:
        dst_w, dst_h = params.resize_to
        sx = dst_w / W
        sy = dst_h / H
        x = (x + 0.5) * sx - 0.5
        y = (y + 0.5) * sy - 0.5
        W, H = dst_w, dst_h
    tx, ty = params.translate
    x += tx
    y += ty
    return x, y, W, H


def _roundtrip_consistency(params, mirror_x: bool = False):
    # Synthetic scene: a fronto-parallel plane at z=1.5m, 64x48, cx/cy off-center
    W, H = 64, 48
    K = np.array([[40.0, 0, 28.0], [0, 40.0, 22.0], [0, 0, 1]], dtype=np.float64)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[10:38, 12:52] = 1  # central rectangle
    depth = np.full((H, W), 1.5, dtype=np.float32)

    rgb2, mask2, depth2, K2 = apply_geom(rgb, mask, depth, K, params)

    # Sample 10 mask pixels (deterministic via fixed grid)
    ys, xs = np.where(mask > 0)
    idxs = np.linspace(0, len(xs) - 1, 10).astype(int)
    src_pixels = [(int(xs[i]), int(ys[i])) for i in idxs]

    max_err = 0.0
    compared = 0
    for (x, y) in src_pixels:
        P_src = _unproject(K, (x, y), depth)
        if P_src is None:
            continue
        x2, y2, _, _ = _apply_params_to_pixel(W, H, params, x, y)
        P_aug = _unproject(K2, (x2, y2), depth2)
        if P_aug is None:
            continue  # pixel fell outside after translate -- acceptable
        # Horizontal flip mirrors the X axis: P_aug = (-P_src.X, P_src.Y, P_src.Z)
        if mirror_x:
            expected = np.array([-P_src[0], P_src[1], P_src[2]])
        else:
            expected = P_src
        err = np.linalg.norm(P_aug - expected)
        max_err = max(max_err, err)
        compared += 1

    assert compared >= 5, "Too few comparable pixels after augmentation"
    scene_scale = 1.5  # plane depth
    assert max_err / scene_scale < 0.05, f"max_err={max_err} exceeds 5% of scene scale"


def test_roundtrip_flip():
    _roundtrip_consistency(
        GeomParams(flip=True, crop_box=None, resize_to=None, translate=(0.0, 0.0)),
        mirror_x=True,
    )


def test_roundtrip_crop_resize():
    _roundtrip_consistency(GeomParams(
        flip=False, crop_box=(4, 3, 56, 42), resize_to=(64, 48), translate=(0.0, 0.0),
    ))


def test_roundtrip_translate():
    _roundtrip_consistency(GeomParams(
        flip=False, crop_box=None, resize_to=None, translate=(2.0, -1.0),
    ))


from scripts.augment_dataset import build_color_transform


def test_color_transform_preserves_shape():
    rgb = (np.random.rand(24, 32, 3) * 255).astype(np.uint8)
    t = build_color_transform()
    out = t(image=rgb)["image"]
    assert out.shape == rgb.shape
    assert out.dtype == rgb.dtype


from scripts.augment_dataset import is_mask_safe, is_depth_safe


def test_is_mask_safe_accepts_large():
    assert is_mask_safe(aug_area=90.0, original_area=100.0, min_ratio=0.1) is True


def test_is_mask_safe_rejects_tiny():
    assert is_mask_safe(aug_area=5.0, original_area=100.0, min_ratio=0.1) is False


def test_is_mask_safe_rejects_zero_original():
    assert is_mask_safe(aug_area=5.0, original_area=0.0, min_ratio=0.1) is False


def test_is_depth_safe_accepts_many_valid():
    src = np.full((10, 10), 1.5, dtype=np.float32)      # 100 valid
    aug = np.full((10, 10), 1.5, dtype=np.float32)
    aug[:, :3] = 0  # 70 valid
    assert is_depth_safe(aug_depth=aug, original_depth=src, min_ratio=0.1) is True


def test_is_depth_safe_rejects_mostly_invalid():
    src = np.full((10, 10), 1.5, dtype=np.float32)      # 100 valid
    aug = np.zeros((10, 10), dtype=np.float32)
    aug[0, :5] = 1.5  # 5 valid, 5% of src
    assert is_depth_safe(aug_depth=aug, original_depth=src, min_ratio=0.1) is False


from types import SimpleNamespace

from scripts.augment_dataset import compute_quotas_balanced, compute_quotas_uniform


def _fake_coco(images_per_category: dict[int, int]):
    """Build a minimal object exposing the pycocotools interface we use."""
    images = []
    anns = []
    img_id = 0
    ann_id = 0
    for cat_id, n in images_per_category.items():
        for _ in range(n):
            images.append({"id": img_id})
            anns.append({"id": ann_id, "image_id": img_id, "category_id": cat_id})
            img_id += 1
            ann_id += 1
    cats = [{"id": cid, "name": f"cat{cid}"} for cid in images_per_category.keys()]
    fake = SimpleNamespace()
    fake.imgs = {im["id"]: im for im in images}
    fake.anns = {a["id"]: a for a in anns}
    fake.cats = {c["id"]: c for c in cats}
    fake.getCatIds = lambda: list(fake.cats.keys())
    fake.getImgIds = lambda catIds=None: [
        a["image_id"] for a in anns if (catIds is None or a["category_id"] in catIds)
    ]
    return fake


def test_compute_quotas_balanced_small_remainder():
    # 1 category with 5 images, target 8 -> need 3
    coco = _fake_coco({1: 5})
    rng = random.Random(0)
    quotas = compute_quotas_balanced(coco, target=8, rng=rng)
    assert sum(quotas.values()) == 3
    # Each image quota is 0 or 1 since base=0, remainder=3
    assert all(q in (0, 1) for q in quotas.values())
    # Exactly 3 images got the +1
    assert list(quotas.values()).count(1) == 3


def test_compute_quotas_balanced_base_plus_remainder():
    # 4 images, target 10 -> need 6, base=1, remainder=2
    coco = _fake_coco({1: 4})
    rng = random.Random(0)
    quotas = compute_quotas_balanced(coco, target=10, rng=rng)
    assert sum(quotas.values()) == 6
    assert all(q in (1, 2) for q in quotas.values())
    assert list(quotas.values()).count(2) == 2


def test_compute_quotas_balanced_skips_already_met():
    coco = _fake_coco({1: 12})  # already > target=10
    rng = random.Random(0)
    quotas = compute_quotas_balanced(coco, target=10, rng=rng)
    assert sum(quotas.values()) == 0


def test_compute_quotas_balanced_skips_zero_samples():
    # two cats declared; one has 0 images
    coco = _fake_coco({1: 3, 2: 0})
    rng = random.Random(0)
    quotas = compute_quotas_balanced(coco, target=5, rng=rng)
    # Only category 1 produces quota entries
    assert sum(quotas.values()) == 2


def test_compute_quotas_uniform():
    coco = _fake_coco({1: 3, 2: 4})
    quotas = compute_quotas_uniform(coco, num_aug=5)
    assert sum(quotas.values()) == 7 * 5
    assert all(q == 5 for q in quotas.values())


from scripts.augment_dataset import read_source_meta, write_augmented_meta, copy_dataset_files


def test_read_source_meta_returns_intrinsics_and_annotation(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    sample = {
        "camera": {"intrinsics": {
            "fx": 1171.28, "fy": 1170.44, "cx": 970.16, "cy": 542.53,
            "width": 1920, "height": 1080}},
        "annotation": {"class_name": "fangguan", "dimensions": [0.22, 0.21, 0.22]},
    }
    (meta_dir / "img1.json").write_text(json.dumps(sample))
    K, W, H, annotation = read_source_meta(meta_dir, "img1.png")
    assert K[0, 0] == pytest.approx(1171.28)
    assert K[0, 2] == pytest.approx(970.16)
    assert (W, H) == (1920, 1080)
    assert annotation["class_name"] == "fangguan"


def test_write_augmented_meta_schema(tmp_path):
    out_dir = tmp_path / "meta"
    out_dir.mkdir()
    K = _K(100.0, 110.0, 50.0, 40.0)
    annotation = {"class_name": "gaiban", "dimensions": [1.0, 2.0, 3.0]}
    write_augmented_meta(out_dir, "AUG_train_0000.png", K=K, W=1920, H=1080, annotation=annotation)
    data = json.loads((out_dir / "AUG_train_0000.json").read_text())
    assert data["camera"]["intrinsics"]["fx"] == pytest.approx(100.0)
    assert data["camera"]["intrinsics"]["cx"] == pytest.approx(50.0)
    assert data["camera"]["intrinsics"]["width"] == 1920
    assert data["annotation"]["class_name"] == "gaiban"
    assert data["annotation"]["dimensions"] == [1.0, 2.0, 3.0]


def test_copy_dataset_files_copies_all_three_dirs(tmp_path):
    src = tmp_path / "in"
    dst = tmp_path / "out"
    for sub in ("images", "depth", "meta"):
        (src / sub).mkdir(parents=True)
    (src / "images" / "a.png").write_bytes(b"fake")
    (src / "depth" / "a.exr").write_bytes(b"fakedepth")
    (src / "meta" / "a.json").write_text("{}")
    copy_dataset_files(src, dst)
    assert (dst / "images" / "a.png").read_bytes() == b"fake"
    assert (dst / "depth" / "a.exr").read_bytes() == b"fakedepth"
    assert (dst / "meta" / "a.json").read_text() == "{}"


from pycocotools.coco import COCO
from scripts.augment_dataset import augment_train_split


def _write_png(path, W=64, H=48, color=(120, 200, 80)):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = color
    cv2.imwrite(str(path), img)


def _write_exr(path, W=64, H=48, value=1.5):
    depth = np.full((H, W), value, dtype=np.float32)
    cv2.imwrite(str(path), depth)


def _write_meta(path, W=64, H=48, class_name="gaiban"):
    meta = {
        "camera": {"intrinsics": {
            "fx": 40.0, "fy": 40.0, "cx": W / 2 - 0.5, "cy": H / 2 - 0.5,
            "width": W, "height": H}},
        "annotation": {"class_name": class_name, "dimensions": [1.0, 1.0, 1.0]},
    }
    path.write_text(json.dumps(meta))


def _build_mini_dataset(tmp_path):
    d = tmp_path / "mini"
    for sub in ("images", "depth", "meta", "annotations"):
        (d / sub).mkdir(parents=True)

    # 2 categories, 2 images each
    images = []
    anns = []
    for idx, (cat_id, class_name) in enumerate([(1, "gaiban"), (1, "gaiban"), (2, "fangguan"), (2, "fangguan")]):
        fname = f"img{idx}.png"
        _write_png(d / "images" / fname)
        _write_exr(d / "depth" / f"img{idx}.exr")
        _write_meta(d / "meta" / f"img{idx}.json", class_name=class_name)
        images.append({
            "license": "", "url": "", "file_name": fname,
            "height": 48, "width": 64, "date_captured": "", "id": idx,
        })
        # rectangle mask covering most of the image
        seg_poly = [10.0, 10.0, 54.0, 10.0, 54.0, 38.0, 10.0, 38.0]
        anns.append({
            "iscrowd": False, "image_id": idx, "image_name": fname,
            "category_id": cat_id, "id": idx,
            "segmentation": [seg_poly],
            "area": 44.0 * 28.0,
            "bbox": [10.0, 10.0, 44.0, 28.0],
        })

    coco_dict = {
        "info": {}, "licenses": [],
        "categories": [
            {"id": 1, "name": "gaiban", "supercategory": None},
            {"id": 2, "name": "fangguan", "supercategory": None},
        ],
        "images": images, "annotations": anns,
    }
    (d / "annotations" / "train.json").write_text(json.dumps(coco_dict))
    # also write val/test so copy_dataset_files has something to handle
    (d / "annotations" / "val.json").write_text(json.dumps({
        **coco_dict, "images": [], "annotations": []}))
    (d / "annotations" / "test.json").write_text(json.dumps({
        **coco_dict, "images": [], "annotations": []}))
    return d


def test_augment_train_split_end_to_end(tmp_path):
    src = _build_mini_dataset(tmp_path)
    out = tmp_path / "out"

    # target=5 per category -> each category needs 3 more from its 2 images
    coco = COCO(str(src / "annotations" / "train.json"))
    quotas = compute_quotas_balanced(coco, target=5, rng=random.Random(0))
    augment_train_split(input_dir=src, output_dir=out, quotas=quotas, seed=0)

    # Check output layout
    for sub in ("images", "depth", "meta"):
        assert (out / sub).is_dir()
    assert (out / "annotations" / "train.json").is_file()
    assert (out / "annotations" / "val.json").is_file()
    assert (out / "annotations" / "test.json").is_file()

    # Re-load augmented train.json and verify category counts
    aug = COCO(str(out / "annotations" / "train.json"))
    counts = {1: 0, 2: 0}
    for a in aug.dataset["annotations"]:
        counts[a["category_id"]] = counts.get(a["category_id"], 0) + 1
    assert counts[1] == 5
    assert counts[2] == 5

    # Every augmented sample has matching png / exr / json
    for img in aug.dataset["images"]:
        fname = img["file_name"]
        if not fname.startswith("AUG_"):
            continue
        stem = os.path.splitext(fname)[0]
        assert (out / "images" / f"{stem}.png").is_file()
        assert (out / "depth" / f"{stem}.exr").is_file()
        assert (out / "meta" / f"{stem}.json").is_file()
        # meta schema
        mdata = json.loads((out / "meta" / f"{stem}.json").read_text())
        assert set(mdata["camera"]["intrinsics"].keys()) == {
            "fx", "fy", "cx", "cy", "width", "height"}
        assert "class_name" in mdata["annotation"]
        assert "dimensions" in mdata["annotation"]

    # source_image_id present and valid
    orig_ids = {img["id"] for img in aug.dataset["images"] if not img["file_name"].startswith("AUG_")}
    for a in aug.dataset["annotations"]:
        if a["id"] >= 100000:  # augmented
            assert a["source_image_id"] in orig_ids


import argparse
from scripts.augment_dataset import build_arg_parser


def test_cli_accepts_target_per_category():
    p = build_arg_parser()
    args = p.parse_args([
        "--input_dir", "/tmp/in", "--output_dir", "/tmp/out",
        "--target_per_category", "900", "--seed", "7",
    ])
    assert args.target_per_category == 900
    assert args.seed == 7
    assert args.num_aug == 3  # default unchanged


def test_cli_legacy_num_aug_still_works():
    p = build_arg_parser()
    args = p.parse_args([
        "--input_dir", "/tmp/in", "--output_dir", "/tmp/out",
        "--num_aug", "2",
    ])
    assert args.target_per_category is None
    assert args.num_aug == 2
