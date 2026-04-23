# Train-Split Category-Balanced Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `scripts/augment_dataset.py` so it balances the train split of `data/aiws5.2-dataset` to a configurable per-category target (default 900), producing geometrically-consistent quadruples (RGB, mask, depth, intrinsics K) suitable for future seg+pose training.

**Architecture:** Hand-rolled geometric augmentation (flip / crop+resize / translate) with explicit K propagation; albumentations used only for color ops. Depth synchronizes with RGB+mask via nearest-neighbor resize and constant-zero border fill. Each augmented sample writes a matching `images/*.png`, `depth/*.exr`, `meta/*.json`, and COCO annotation entry. Val / test splits pass through unchanged.

**Tech Stack:** Python 3, NumPy, OpenCV (with `OPENCV_IO_ENABLE_OPENEXR=1`), albumentations, pycocotools, pytest.

**Spec:** `docs/superpowers/specs/2026-04-22-train-category-balance-augment-design.md`

---

## File Structure

**Modified:**
- `scripts/augment_dataset.py` — significant rewrite. New pure-function core for intrinsic propagation and geometric transforms; rewritten orchestration that takes a pre-computed quota map; extended CLI.

**Created:**
- `tests/scripts/test_augment_dataset.py` — unit tests for intrinsic propagation, synthetic-data round-trip tests for geometric consistency, and an end-to-end integration test with a tmp-path mini COCO.

**Unchanged (but read/respected):**
- `datasets/datasets_nuclear.py` — consumer of the augmented data; depth sanitization rules (lines 186–191) must match what the augmentation script produces.

---

## Task 1: Intrinsic Update Pure Functions

Implement the four per-operator K update formulas from spec §6 as pure functions. No I/O, no state — easy to test exhaustively.

**Files:**
- Modify: `scripts/augment_dataset.py` (add new functions)
- Create: `tests/scripts/test_augment_dataset.py`

- [ ] **Step 1: Write failing tests for all four ops**

Create `tests/scripts/test_augment_dataset.py`:

```python
"""Tests for scripts/augment_dataset.py."""

import os

# EXR support must be enabled before cv2 is imported anywhere in the test process.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
import pytest

from scripts.augment_dataset import (
    update_K_flip,
    update_K_crop,
    update_K_resize,
    update_K_translate,
)


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
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `pytest tests/scripts/test_augment_dataset.py -v`
Expected: 5 errors like `ImportError: cannot import name 'update_K_flip' ...`

- [ ] **Step 3: Implement the four pure functions**

Add to the top of `scripts/augment_dataset.py` (after existing imports, before `mask_to_polygons`):

```python
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


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
```

Note: the `os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")` line must appear **before** `import cv2` in the file. The existing file imports cv2 at the top — move the env setdefault to precede that import.

- [ ] **Step 4: Run tests, confirm they pass**

Run: `pytest tests/scripts/test_augment_dataset.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/augment_dataset.py tests/scripts/test_augment_dataset.py
git commit -m "feat(augment): pure intrinsic-update functions for flip/crop/resize/translate"
```

---

## Task 2: GeomParams Dataclass + Composed K Update

Define the parameter container and a composition function that applies ops in canonical order (flip → crop → resize → translate). This is the single entry point later code uses to propagate K.

**Files:**
- Modify: `scripts/augment_dataset.py`
- Modify: `tests/scripts/test_augment_dataset.py`

- [ ] **Step 1: Write failing test for composition**

Append to `tests/scripts/test_augment_dataset.py`:

```python
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
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `pytest tests/scripts/test_augment_dataset.py::test_compose_K_flip_only -v`
Expected: `ImportError: cannot import name 'GeomParams' ...`

- [ ] **Step 3: Implement GeomParams and compose_K**

Add to `scripts/augment_dataset.py` (after the four `update_K_*` functions):

```python
from dataclasses import dataclass
from typing import Optional, Tuple


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
```

- [ ] **Step 4: Run tests, confirm they pass**

Run: `pytest tests/scripts/test_augment_dataset.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/augment_dataset.py tests/scripts/test_augment_dataset.py
git commit -m "feat(augment): GeomParams dataclass and compose_K chaining"
```

---

## Task 3: Geometric Parameter Sampling

Sample `GeomParams` randomly per the probabilities and ranges in spec §5.1. Must be seeded for reproducibility and stay within documented ranges.

**Files:**
- Modify: `scripts/augment_dataset.py`
- Modify: `tests/scripts/test_augment_dataset.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/scripts/test_augment_dataset.py`:

```python
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
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `pytest tests/scripts/test_augment_dataset.py::test_sample_geom_params_reproducible -v`
Expected: ImportError.

- [ ] **Step 3: Implement sample_geom_params**

Add to `scripts/augment_dataset.py`:

```python
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
```

Also add `import random` to the imports block of `scripts/augment_dataset.py` (it is already imported inside `augment_split` in the original code; move it to module level).

- [ ] **Step 4: Run tests, confirm they pass**

Run: `pytest tests/scripts/test_augment_dataset.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/augment_dataset.py tests/scripts/test_augment_dataset.py
git commit -m "feat(augment): seeded geometric-param sampling"
```

---

## Task 4: Apply Geometric Transform to RGB / Mask / Depth

Implement `apply_geom(rgb, mask, depth, K, params) → (rgb', mask', depth', K')`. RGB uses linear interpolation for cosmetic ops, mask and depth use nearest-neighbor. Borders fill with 0.

**Files:**
- Modify: `scripts/augment_dataset.py`
- Modify: `tests/scripts/test_augment_dataset.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/scripts/test_augment_dataset.py`:

```python
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
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `pytest tests/scripts/test_augment_dataset.py -v -k apply_geom`
Expected: ImportError.

- [ ] **Step 3: Implement apply_geom**

Add to `scripts/augment_dataset.py`:

```python
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
```

- [ ] **Step 4: Run tests, confirm they pass**

Run: `pytest tests/scripts/test_augment_dataset.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/augment_dataset.py tests/scripts/test_augment_dataset.py
git commit -m "feat(augment): apply_geom transforms RGB/mask/depth consistently"
```

---

## Task 5: Geometric-Consistency Round-Trip Tests

This is the **core correctness gate** (spec §11.2). Unproject foreground pixels via (K, depth); apply the augmentation; unproject the same pixels' new locations via (K', depth'); assert the 3D points match within a small tolerance. Uses synthetic planar-scene depth for reproducibility — no real EXR file needed.

**Files:**
- Modify: `tests/scripts/test_augment_dataset.py`

- [ ] **Step 1: Add the round-trip helper and three tests**

Append to `tests/scripts/test_augment_dataset.py`:

```python
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


def _roundtrip_consistency(params):
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
        err = np.linalg.norm(P_aug - P_src)
        max_err = max(max_err, err)
        compared += 1

    assert compared >= 5, "Too few comparable pixels after augmentation"
    scene_scale = 1.5  # plane depth
    assert max_err / scene_scale < 0.05, f"max_err={max_err} exceeds 5% of scene scale"


def test_roundtrip_flip():
    _roundtrip_consistency(GeomParams(
        flip=True, crop_box=None, resize_to=None, translate=(0.0, 0.0)
    ))


def test_roundtrip_crop_resize():
    _roundtrip_consistency(GeomParams(
        flip=False, crop_box=(4, 3, 56, 42), resize_to=(64, 48), translate=(0.0, 0.0),
    ))


def test_roundtrip_translate():
    _roundtrip_consistency(GeomParams(
        flip=False, crop_box=None, resize_to=None, translate=(2.0, -1.0),
    ))
```

- [ ] **Step 2: Run round-trip tests**

Run: `pytest tests/scripts/test_augment_dataset.py -v -k roundtrip`
Expected: 3 passed.

If any fails: the K-propagation formula in Task 1 is wrong for that op. Debug by printing `P_src`, `P_aug`, `K`, `K2` and comparing to the formula in spec §6. Do not loosen the tolerance to pass — fix the formula.

- [ ] **Step 3: Commit**

```bash
git add tests/scripts/test_augment_dataset.py
git commit -m "test(augment): 3D round-trip consistency for flip/crop-resize/translate"
```

---

## Task 6: Color Transform Builder

Rebuild the color-only albumentations pipeline. This drops Rotate and Affine from the original script, and is applied to RGB only (not mask/depth).

**Files:**
- Modify: `scripts/augment_dataset.py`
- Modify: `tests/scripts/test_augment_dataset.py`

- [ ] **Step 1: Write test**

Append to `tests/scripts/test_augment_dataset.py`:

```python
from scripts.augment_dataset import build_color_transform


def test_color_transform_preserves_shape():
    rgb = (np.random.rand(24, 32, 3) * 255).astype(np.uint8)
    t = build_color_transform()
    out = t(image=rgb)["image"]
    assert out.shape == rgb.shape
    assert out.dtype == rgb.dtype
```

- [ ] **Step 2: Run test, confirm failure**

Run: `pytest tests/scripts/test_augment_dataset.py -v -k color_transform`
Expected: ImportError.

- [ ] **Step 3: Replace `build_transform` with `build_color_transform`**

In `scripts/augment_dataset.py`, find the existing `build_transform` (around lines 60–76) and replace it with:

```python
def build_color_transform() -> A.Compose:
    """RGB-only color augmentations; no geometric ops (those are hand-rolled)."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
        A.GaussNoise(std_range=(0.02, 0.05), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    ])
```

Also remove the now-unused `build_transform` import sites inside `augment_split` (they will be rewritten in Task 9 anyway, but drop any direct references now).

- [ ] **Step 4: Run test, confirm pass**

Run: `pytest tests/scripts/test_augment_dataset.py -v -k color_transform`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/augment_dataset.py tests/scripts/test_augment_dataset.py
git commit -m "feat(augment): RGB-only color transform (drops Rotate/Affine)"
```

---

## Task 7: Safety Checks (Mask + Depth)

Extend the safety logic to also check that the augmented depth map retains enough valid pixels. Rename the existing `is_augmentation_safe` → `is_mask_safe`, add a new `is_depth_safe`.

**Files:**
- Modify: `scripts/augment_dataset.py`
- Modify: `tests/scripts/test_augment_dataset.py`

- [ ] **Step 1: Write tests**

Append to `tests/scripts/test_augment_dataset.py`:

```python
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
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `pytest tests/scripts/test_augment_dataset.py -v -k "mask_safe or depth_safe"`
Expected: ImportError.

- [ ] **Step 3: Rename and add new function**

In `scripts/augment_dataset.py`, rename `is_augmentation_safe` to `is_mask_safe` (keep the existing implementation). Add immediately after:

```python
def is_depth_safe(aug_depth: np.ndarray, original_depth: np.ndarray, min_ratio: float = 0.1) -> bool:
    """Require at least min_ratio * source-valid-pixel-count non-zero pixels in aug_depth."""
    src_valid = int(np.count_nonzero(original_depth))
    if src_valid == 0:
        return True  # all-zero source: pass through (sensor data unavailable)
    aug_valid = int(np.count_nonzero(aug_depth))
    return (aug_valid / src_valid) >= min_ratio
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/scripts/test_augment_dataset.py -v`
Expected: all tests pass (including the renamed `is_mask_safe`).

- [ ] **Step 5: Commit**

```bash
git add scripts/augment_dataset.py tests/scripts/test_augment_dataset.py
git commit -m "feat(augment): split safety checks into is_mask_safe + is_depth_safe"
```

---

## Task 8: Per-Category Quota Calculator

Implement `compute_quotas_balanced(coco, target, rng)` per spec §4 and a trivial `compute_quotas_uniform(coco, num_aug)` for legacy-mode compatibility.

**Files:**
- Modify: `scripts/augment_dataset.py`
- Modify: `tests/scripts/test_augment_dataset.py`

- [ ] **Step 1: Write tests**

Append to `tests/scripts/test_augment_dataset.py`:

```python
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
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `pytest tests/scripts/test_augment_dataset.py -v -k quotas`
Expected: ImportError.

- [ ] **Step 3: Implement the quota calculators**

Add to `scripts/augment_dataset.py`:

```python
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
```

Note: `logger` is already defined at module scope (line 13 of the original file).

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/scripts/test_augment_dataset.py -v -k quotas`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/augment_dataset.py tests/scripts/test_augment_dataset.py
git commit -m "feat(augment): per-category quota calculators"
```

---

## Task 9: Dataset File-Copy and Meta I/O Helpers

Add helpers to (a) copy `images/`, `depth/`, `meta/` from input to output, (b) read a source meta JSON, (c) write an augmented meta JSON with updated intrinsics.

**Files:**
- Modify: `scripts/augment_dataset.py`
- Modify: `tests/scripts/test_augment_dataset.py`

- [ ] **Step 1: Write tests**

Append to `tests/scripts/test_augment_dataset.py`:

```python
import json

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
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `pytest tests/scripts/test_augment_dataset.py -v -k "meta or copy_dataset"`
Expected: ImportError.

- [ ] **Step 3: Implement the helpers**

Add to `scripts/augment_dataset.py`:

```python
from pathlib import Path


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
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/scripts/test_augment_dataset.py -v`
Expected: all previous tests + 3 new ones pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/augment_dataset.py tests/scripts/test_augment_dataset.py
git commit -m "feat(augment): meta I/O and dataset-file copy helpers"
```

---

## Task 10: augment_train_split Orchestration

Rewrite the top-level `augment_split` into `augment_train_split(input_dir, output_dir, quotas, seed, num_categories)`. This is where all the pieces come together: read source quadruples, sample params, apply geom+color, safety check, retry, write outputs, append COCO entries.

**Files:**
- Modify: `scripts/augment_dataset.py`
- Modify: `tests/scripts/test_augment_dataset.py`

- [ ] **Step 1: Write integration test using synthetic mini dataset**

Append to `tests/scripts/test_augment_dataset.py`:

```python
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
```

- [ ] **Step 2: Run test, confirm failure**

Run: `pytest tests/scripts/test_augment_dataset.py::test_augment_train_split_end_to_end -v`
Expected: `ImportError: cannot import name 'augment_train_split'` (or similar).

- [ ] **Step 3: Implement augment_train_split**

Delete the existing `augment_split` function body in `scripts/augment_dataset.py` (lines ~100–225 of the original). Replace with:

```python
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
```

- [ ] **Step 4: Run integration test, confirm pass**

Run: `pytest tests/scripts/test_augment_dataset.py::test_augment_train_split_end_to_end -v`
Expected: 1 passed.

If the test fails because `cv2.imwrite` can't write EXR, verify `OPENCV_IO_ENABLE_OPENEXR=1` is set at the top of the test file AND at the top of `scripts/augment_dataset.py` (both places must be before any `import cv2`).

- [ ] **Step 5: Commit**

```bash
git add scripts/augment_dataset.py tests/scripts/test_augment_dataset.py
git commit -m "feat(augment): augment_train_split orchestration with quota map"
```

---

## Task 11: CLI Wiring

Extend `main()` to accept `--target_per_category` and dispatch:
- If `--target_per_category` is set: train-only, balanced quotas.
- Else fall back to legacy uniform `--num_aug` mode on all splits.
- Abort if input_dir == output_dir.

**Files:**
- Modify: `scripts/augment_dataset.py`
- Modify: `tests/scripts/test_augment_dataset.py`

- [ ] **Step 1: Write CLI parse test**

Append to `tests/scripts/test_augment_dataset.py`:

```python
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
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `pytest tests/scripts/test_augment_dataset.py -v -k cli`
Expected: ImportError.

- [ ] **Step 3: Replace main() with parser extraction + dispatch**

Replace the existing `main()` in `scripts/augment_dataset.py` with:

```python
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline data augmentation for AIWS5.2 COCO segmentation datasets."
    )
    p.add_argument("--input_dir", type=str, required=True,
                   help="Path to input dataset (must contain annotations/, images/, depth/, meta/).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Path to output augmented dataset (must differ from input_dir).")
    p.add_argument("--target_per_category", type=int, default=None,
                   help="If set, balance each non-empty train category to this count "
                        "(only the train split is augmented).")
    p.add_argument("--num_aug", type=int, default=3,
                   help="Legacy uniform multiplier (applied to train+val when "
                        "--target_per_category is not set).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p


def main() -> None:
    import argparse as _argparse_marker  # keep the import close to its usage
    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise SystemExit("ERROR: --input_dir and --output_dir must differ.")

    if args.target_per_category is not None:
        # Balanced mode: train only
        train_ann = input_dir / "annotations" / "train.json"
        if not train_ann.is_file():
            raise SystemExit(f"ERROR: {train_ann} not found.")
        coco = COCO(str(train_ann))
        rng = random.Random(args.seed)
        quotas = compute_quotas_balanced(coco, target=args.target_per_category, rng=rng)
        augment_train_split(
            input_dir=input_dir, output_dir=output_dir,
            quotas=quotas, seed=args.seed,
        )
    else:
        # Legacy uniform mode: train + val
        for split in ("train", "val"):
            split_ann = input_dir / "annotations" / f"{split}.json"
            if not split_ann.is_file():
                logger.warning(f"{split_ann} not found, skipping {split}")
                continue
            coco = COCO(str(split_ann))
            quotas = compute_quotas_uniform(coco, num_aug=args.num_aug)
            # augment_train_split handles train; for legacy val we need a variant
            # Reuse augment_train_split but point it at the named split:
            _augment_named_split(
                input_dir=input_dir, output_dir=output_dir,
                split=split, quotas=quotas, seed=args.seed,
            )

    logger.info("Augmentation complete.")


def _augment_named_split(input_dir, output_dir, split: str, quotas: dict, seed: int) -> None:
    """Legacy-mode variant that augments an arbitrary named split.

    Implemented by monkey-setting the train_ann path inside augment_train_split.
    To keep things simple we duplicate the tiny amount of logic instead.
    """
    # Copy originals + other splits (idempotent across calls)
    output_dir_p = Path(output_dir)
    input_dir_p = Path(input_dir)
    (output_dir_p / "annotations").mkdir(parents=True, exist_ok=True)
    copy_dataset_files(input_dir_p, output_dir_p)
    for other in ("train", "val", "test"):
        if other == split:
            continue
        src = input_dir_p / "annotations" / f"{other}.json"
        if src.is_file():
            shutil.copy2(src, output_dir_p / "annotations" / f"{other}.json")

    # Augment this one split using the same inner pipeline
    split_ann = input_dir_p / "annotations" / f"{split}.json"
    with open(split_ann) as f:
        coco_data = json.load(f)
    coco = COCO(str(split_ann))
    color_transform = build_color_transform()
    rng = random.Random(seed)
    np.random.seed(seed)

    aug_images, aug_annotations = [], []
    aug_img_id = aug_ann_id = 100000
    counter = 0
    for img_info in coco_data["images"]:
        q = quotas.get(img_info["id"], 0)
        if q == 0:
            continue
        fname = img_info["file_name"]
        stem = os.path.splitext(fname)[0]
        rgb = cv2.imread(str(input_dir_p / "images" / fname), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(
            str(input_dir_p / "depth" / f"{stem}.exr"),
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
        )
        if depth is None:
            continue
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = _sanitize_depth(depth.astype(np.float32, copy=False))
        ann_ids = coco.getAnnIds(imgIds=img_info["id"])
        if not ann_ids:
            continue
        ann = coco.loadAnns(ann_ids)[0]
        mask = coco.annToMask(ann).astype(np.uint8)
        try:
            K, W, H, meta_annotation = read_source_meta(input_dir_p / "meta", fname)
        except FileNotFoundError:
            continue
        for slot in range(q):
            out = _try_generate_augmented_sample(
                rgb, mask, depth, K, W, H, ann["area"],
                color_transform, rng, max_retries=5,
            )
            if out is None:
                continue
            aug_rgb, aug_mask, aug_depth, aug_K, aug_W, aug_H, polygons, bbox, area = out
            aug_filename = f"AUG_{split}_{counter:04d}.png"
            counter += 1
            cv2.imwrite(
                str(output_dir_p / "images" / aug_filename),
                cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(output_dir_p / "depth" / f"{os.path.splitext(aug_filename)[0]}.exr"),
                aug_depth,
            )
            write_augmented_meta(
                output_dir_p / "meta", aug_filename,
                K=aug_K, W=aug_W, H=aug_H, annotation=meta_annotation,
            )
            aug_images.append({
                "license": img_info.get("license", ""), "url": "",
                "file_name": aug_filename, "height": int(aug_H), "width": int(aug_W),
                "date_captured": "", "id": aug_img_id,
            })
            aug_annotations.append({
                "iscrowd": False, "image_id": aug_img_id, "image_name": aug_filename,
                "category_id": ann["category_id"], "id": aug_ann_id,
                "segmentation": polygons, "area": area, "bbox": bbox,
                "source_image_id": img_info["id"],
            })
            aug_img_id += 1
            aug_ann_id += 1
    merged = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data["categories"],
        "images": coco_data["images"] + aug_images,
        "annotations": coco_data["annotations"] + aug_annotations,
    }
    with open(output_dir_p / "annotations" / f"{split}.json", "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    logger.info(
        f"[{split}] legacy-mode: {len(coco_data['images'])} originals, "
        f"{len(aug_images)} augmented."
    )
```

Also at the top of `scripts/augment_dataset.py` ensure `import argparse` is present.

- [ ] **Step 4: Run CLI tests, confirm pass**

Run: `pytest tests/scripts/test_augment_dataset.py -v -k cli`
Expected: 2 passed.

- [ ] **Step 5: Full-suite smoke run**

Run: `pytest tests/scripts/test_augment_dataset.py -v`
Expected: all tests pass (roughly 30+ tests total).

- [ ] **Step 6: Commit**

```bash
git add scripts/augment_dataset.py tests/scripts/test_augment_dataset.py
git commit -m "feat(augment): CLI with --target_per_category + legacy fallback"
```

---

## Task 12: Real-Dataset Smoke Run and Documentation

Run the balanced augmentation against the real dataset, verify counts and sanity-check a few augmented samples.

**Files:**
- Modify: `scripts/augment_dataset.py` (docstring at top only — no behavior change)

- [ ] **Step 1: Run the balanced augmentation on the real dataset**

```bash
python -m scripts.augment_dataset \
    --input_dir data/aiws5.2-dataset \
    --output_dir data/aiws5.2-dataset-balanced \
    --target_per_category 900 \
    --seed 42
```

Expected log lines:
- "category 5: 0 samples, cannot augment, skipping" (槽钢)
- "category 6: 0 samples, cannot augment, skipping" (坡口)
- "train split done. Originals: 1037, augmented: ~2560, total: ~3600."

- [ ] **Step 2: Verify per-category counts match target**

```bash
python3 - <<'PY'
import json, collections
with open("data/aiws5.2-dataset-balanced/annotations/train.json") as f:
    d = json.load(f)
id2name = {c["id"]: c["name"] for c in d["categories"]}
c = collections.Counter(a["category_id"] for a in d["annotations"])
for cid, n in sorted(c.items()):
    print(f"{id2name[cid]} (id={cid}): {n}")
PY
```

Expected output: each of 盖板, 方管, 喇叭口, H型钢 → 900. 槽钢, 坡口 absent or 0.

- [ ] **Step 3: Load augmented data via the existing dataset class**

```bash
python3 - <<'PY'
import sys, os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
sys.path.insert(0, ".")
from types import SimpleNamespace
from datasets.datasets_nuclear import NuclearWorkpieceDataset
ds = NuclearWorkpieceDataset(
    cfg=SimpleNamespace(agent_type="full"),
    data_dir="data/aiws5.2-dataset-balanced",
    annotation_file="data/aiws5.2-dataset-balanced/annotations/train.json",
    mode="full",
    img_size=224,
)
sample = ds[len(ds) - 1]  # very likely an AUG_ sample
print("keys:", list(sample.keys()))
print("depth shape:", sample["depth"].shape if "depth" in sample else "missing")
PY
```

Expected: prints at least `roi_rgb`, `gt_masks`, `gt_classes`, `num_instances`, `image_id`, `depth` keys. No exceptions.

- [ ] **Step 4: Visual inspection of 3 random augmented samples**

```bash
python3 - <<'PY'
import os, random, glob
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2, numpy as np
import matplotlib.pyplot as plt

aug_dir = "data/aiws5.2-dataset-balanced"
pngs = sorted(glob.glob(f"{aug_dir}/images/AUG_train_*.png"))
random.seed(0)
picks = random.sample(pngs, 3)
fig, axes = plt.subplots(3, 2, figsize=(10, 12))
for i, p in enumerate(picks):
    stem = os.path.splitext(os.path.basename(p))[0]
    rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    d = cv2.imread(f"{aug_dir}/depth/{stem}.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if d.ndim == 3:
        d = d[:, :, 0]
    axes[i, 0].imshow(rgb); axes[i, 0].set_title(stem)
    axes[i, 1].imshow(d, cmap="viridis"); axes[i, 1].set_title("depth")
plt.tight_layout()
plt.savefig("/tmp/aug_smoke.png", dpi=120)
print("saved /tmp/aug_smoke.png")
PY
```

Open `/tmp/aug_smoke.png` and confirm RGB and depth look visually consistent (same geometry, no ghost features, black borders where expected).

- [ ] **Step 5: Update the top-of-file docstring**

Replace the first line of `scripts/augment_dataset.py`:

```python
"""Offline data augmentation for AIWS5.2 COCO segmentation dataset."""
```

with:

```python
"""Offline data augmentation for AIWS5.2 COCO seg+pose dataset.

Supports two modes:
  * --target_per_category N : balance each non-empty train category to N samples,
    producing geometrically-consistent (RGB, mask, depth, meta-K) quadruples.
    Only the train split is augmented; val/test pass through unchanged.
  * --num_aug N (legacy) : apply uniform N augmentations per image, to train+val.

See docs/superpowers/specs/2026-04-22-train-category-balance-augment-design.md
for the full design rationale (operator choice, K-propagation formulas, etc.).
"""
```

- [ ] **Step 6: Commit the docstring update**

```bash
git add scripts/augment_dataset.py
git commit -m "docs(augment): update module docstring to describe balanced mode"
```

The generated dataset under `data/aiws5.2-dataset-balanced/` is not committed (it is large and follows the existing `.gitignore` pattern for `data/`).

---

## Self-Review

- **Spec coverage**: every section of the design doc has a task — §2 scope → Task 11 dispatch; §3 output layout → Task 10 orchestration; §4 quotas → Task 8; §5 operators → Tasks 3, 4, 6; §6 K formulas → Tasks 1, 2; §7 depth handling → Task 4 + `_sanitize_depth` in Task 10; §8 meta schema → Task 9; §9 COCO updates → Task 10; §10 architecture → distributed across all tasks; §11 testing → Tasks 1–11 tests + Task 12 smoke; §12 risks → acknowledged in design; §13 acceptance → Task 12.
- **Placeholder scan**: no TODO / TBD / "implement later" markers. Every code block is complete.
- **Type consistency**: `GeomParams` signature (flip/crop_box/resize_to/translate) is identical across Tasks 2, 3, 4, 5, 10. `compute_quotas_*` return `dict[int, int]` consistently. `augment_train_split` signature `(input_dir, output_dir, quotas, seed)` is the same in Tasks 10 and 11. `apply_geom` returns 4-tuple consistently. `_try_generate_augmented_sample` returns a 9-tuple that Task 10 unpacks the same way.
