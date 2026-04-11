# Offline Data Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an offline augmentation script that expands the balanced AIWS5.2 dataset (~396 images) to ~4x size (~1584 images) with synchronized COCO segmentation masks.

**Architecture:** A single standalone script `scripts/augment_dataset.py` reads the input dataset, applies Albumentations transforms to each image+mask pair, converts masks back to COCO polygon format, and writes a merged output dataset. No changes to existing training code required.

**Tech Stack:** Python, Albumentations, OpenCV, pycocotools, NumPy

**Spec:** `docs/superpowers/specs/2026-04-11-data-augmentation-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `scripts/augment_dataset.py` | Main augmentation script — CLI entry point, orchestration |
| Create | `tests/scripts/test_augment_dataset.py` | Unit tests for mask conversion, safety checks, ID generation |

---

### Task 1: Install Albumentations dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add albumentations to requirements.txt**

Add the following line to `requirements.txt`:

```
albumentations>=1.3.0
```

- [ ] **Step 2: Install the dependency**

Run: `pip install albumentations>=1.3.0`
Expected: Successful installation with no conflicts.

- [ ] **Step 3: Verify import**

Run: `python -c "import albumentations; print(albumentations.__version__)"`
Expected: Prints version number (e.g., `1.3.1` or higher).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "deps: add albumentations for offline data augmentation"
```

---

### Task 2: Implement mask conversion utilities

**Files:**
- Create: `scripts/augment_dataset.py` (initial skeleton with conversion functions)
- Create: `tests/scripts/test_augment_dataset.py`

- [ ] **Step 1: Write failing tests for mask_to_polygons**

Create `tests/scripts/test_augment_dataset.py`:

```python
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.augment_dataset import mask_to_polygons


def test_mask_to_polygons_simple_square():
    """A 50x50 white square on a 100x100 black image should produce one polygon."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 1
    polygons = mask_to_polygons(mask, min_area=50)
    assert len(polygons) == 1
    # Polygon is a flat list of [x1, y1, x2, y2, ...]
    assert len(polygons[0]) >= 8  # At least 4 points


def test_mask_to_polygons_filters_small_fragments():
    """Fragments smaller than min_area should be filtered out."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 1       # Large region: 2500 px
    mask[0:3, 0:3] = 1           # Tiny fragment: 9 px
    polygons = mask_to_polygons(mask, min_area=50)
    assert len(polygons) == 1    # Only the large region


def test_mask_to_polygons_empty_mask():
    """An all-zero mask should return an empty list."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    polygons = mask_to_polygons(mask, min_area=50)
    assert len(polygons) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/scripts/test_augment_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError` because `scripts/augment_dataset.py` doesn't exist yet.

- [ ] **Step 3: Implement mask_to_polygons**

Create `scripts/augment_dataset.py`:

```python
"""Offline data augmentation for AIWS5.2 COCO segmentation dataset."""

import cv2
import numpy as np


def mask_to_polygons(mask: np.ndarray, min_area: float = 50.0) -> list[list[float]]:
    """Convert a binary mask to COCO-format polygon list.

    Args:
        mask: Binary mask (H, W) with values 0 or 1, dtype uint8.
        min_area: Minimum contour area to keep (filters fragments).

    Returns:
        List of polygons, each polygon is a flat list [x1, y1, x2, y2, ...].
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        # Flatten contour from (N, 1, 2) to flat list [x1, y1, x2, y2, ...]
        poly = contour.reshape(-1).tolist()
        if len(poly) >= 6:  # At least 3 points to form a polygon
            polygons.append(poly)
    return polygons
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_augment_dataset.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/augment_dataset.py tests/scripts/test_augment_dataset.py
git commit -m "feat: add mask_to_polygons conversion utility"
```

---

### Task 3: Implement safety check and bbox/area computation

**Files:**
- Modify: `scripts/augment_dataset.py`
- Modify: `tests/scripts/test_augment_dataset.py`

- [ ] **Step 1: Write failing tests for compute_bbox_area and is_augmentation_safe**

Append to `tests/scripts/test_augment_dataset.py`:

```python
from scripts.augment_dataset import compute_bbox_area, is_augmentation_safe


def test_compute_bbox_area_simple():
    """Bbox and area for a known polygon."""
    # Square: (10,10), (50,10), (50,50), (10,50)
    polygon = [10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]
    bbox, area = compute_bbox_area([polygon])
    # COCO bbox format: [x, y, width, height]
    assert bbox == [10.0, 10.0, 40.0, 40.0]
    assert area == 1600.0


def test_is_augmentation_safe_pass():
    """Augmented area >= 10% of original should pass."""
    assert is_augmentation_safe(500.0, 1000.0, min_ratio=0.1) is True


def test_is_augmentation_safe_fail():
    """Augmented area < 10% of original should fail."""
    assert is_augmentation_safe(50.0, 1000.0, min_ratio=0.1) is False


def test_is_augmentation_safe_zero_area():
    """Zero augmented area should fail."""
    assert is_augmentation_safe(0.0, 1000.0, min_ratio=0.1) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/scripts/test_augment_dataset.py::test_compute_bbox_area_simple tests/scripts/test_augment_dataset.py::test_is_augmentation_safe_pass -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement compute_bbox_area and is_augmentation_safe**

Add to `scripts/augment_dataset.py` after `mask_to_polygons`:

```python
def compute_bbox_area(polygons: list[list[float]]) -> tuple[list[float], float]:
    """Compute COCO-format bbox and area from polygons.

    Args:
        polygons: List of polygons, each a flat list [x1, y1, x2, y2, ...].

    Returns:
        (bbox, area) where bbox is [x, y, w, h] and area is the polygon area.
    """
    all_x = []
    all_y = []
    total_area = 0.0
    for poly in polygons:
        xs = poly[0::2]
        ys = poly[1::2]
        all_x.extend(xs)
        all_y.extend(ys)
        # Shoelace formula for polygon area
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


def is_augmentation_safe(
    aug_area: float, original_area: float, min_ratio: float = 0.1
) -> bool:
    """Check whether augmented mask retains enough area.

    Args:
        aug_area: Area of the augmented mask.
        original_area: Area of the original mask.
        min_ratio: Minimum ratio of augmented/original area to accept.

    Returns:
        True if safe, False if the augmentation should be discarded.
    """
    if original_area <= 0 or aug_area <= 0:
        return False
    return (aug_area / original_area) >= min_ratio
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_augment_dataset.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/augment_dataset.py tests/scripts/test_augment_dataset.py
git commit -m "feat: add bbox/area computation and safety check"
```

---

### Task 4: Implement the augmentation transform builder

**Files:**
- Modify: `scripts/augment_dataset.py`
- Modify: `tests/scripts/test_augment_dataset.py`

- [ ] **Step 1: Write failing test for build_transform**

Append to `tests/scripts/test_augment_dataset.py`:

```python
import albumentations as A
from scripts.augment_dataset import build_transform


def test_build_transform_returns_compose():
    """build_transform should return an Albumentations Compose object."""
    transform = build_transform(height=1080, width=1920)
    assert isinstance(transform, A.Compose)


def test_build_transform_applies_to_image_and_mask():
    """Transform should accept image and mask, return both with correct shapes."""
    transform = build_transform(height=100, width=200)
    image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[25:75, 50:150] = 1
    result = transform(image=image, mask=mask)
    assert result['image'].shape == (100, 200, 3)
    assert result['mask'].shape == (100, 200)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/scripts/test_augment_dataset.py::test_build_transform_returns_compose -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement build_transform**

Add to `scripts/augment_dataset.py` after the existing functions, and add the import at the top:

```python
import albumentations as A
```

Then add the function:

```python
def build_transform(height: int, width: int) -> A.Compose:
    """Build the Albumentations augmentation pipeline.

    Args:
        height: Target image height.
        width: Target image width.

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.4),
        A.RandomResizedCrop(
            size=(height, width), scale=(0.8, 1.2),
            ratio=(width / height, width / height), p=0.4,
        ),
        A.Affine(
            translate_percent=(-0.05, 0.05), shear=(-5, 5),
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.3,
        ),
        # Color transforms
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4,
        ),
        A.GaussNoise(std_range=(0.02, 0.05), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    ])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_augment_dataset.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/augment_dataset.py tests/scripts/test_augment_dataset.py
git commit -m "feat: add Albumentations transform builder"
```

---

### Task 5: Implement single-image augmentation function

**Files:**
- Modify: `scripts/augment_dataset.py`
- Modify: `tests/scripts/test_augment_dataset.py`

- [ ] **Step 1: Write failing test for augment_single_image**

Append to `tests/scripts/test_augment_dataset.py`:

```python
from scripts.augment_dataset import augment_single_image


def test_augment_single_image_produces_valid_output():
    """augment_single_image should return an augmented image, polygons, bbox, and area."""
    image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[25:75, 50:150] = 1
    original_area = float(mask.sum())
    transform = build_transform(height=100, width=200)

    result = augment_single_image(
        image, mask, original_area, transform, max_retries=5, min_area_ratio=0.1
    )
    assert result is not None
    aug_img, polygons, bbox, area = result
    assert aug_img.shape == (100, 200, 3)
    assert len(polygons) >= 1
    assert len(bbox) == 4
    assert area > 0


def test_augment_single_image_returns_none_on_impossible():
    """If mask is tiny and transform always destroys it, should return None after retries."""
    image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[0, 0] = 1  # Single pixel — will almost certainly be destroyed
    original_area = 5000.0  # Pretend original was large so ratio check always fails
    transform = build_transform(height=100, width=200)

    result = augment_single_image(
        image, mask, original_area, transform, max_retries=3, min_area_ratio=0.1
    )
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/scripts/test_augment_dataset.py::test_augment_single_image_produces_valid_output -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement augment_single_image**

Add to `scripts/augment_dataset.py`:

```python
def augment_single_image(
    image: np.ndarray,
    mask: np.ndarray,
    original_area: float,
    transform: A.Compose,
    max_retries: int = 5,
    min_area_ratio: float = 0.1,
) -> tuple[np.ndarray, list[list[float]], list[float], float] | None:
    """Apply augmentation to one image+mask pair with safety retries.

    Args:
        image: RGB image (H, W, 3), uint8.
        mask: Binary mask (H, W), uint8.
        original_area: Area of the original annotation (for safety check).
        transform: Albumentations Compose pipeline.
        max_retries: Maximum attempts to produce a valid augmentation.
        min_area_ratio: Minimum augmented/original area ratio.

    Returns:
        (augmented_image, polygons, bbox, area) or None if all retries failed.
    """
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_augment_dataset.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/augment_dataset.py tests/scripts/test_augment_dataset.py
git commit -m "feat: add single-image augmentation with safety retries"
```

---

### Task 6: Implement the dataset-level augmentation orchestrator

**Files:**
- Modify: `scripts/augment_dataset.py`

- [ ] **Step 1: Implement augment_split to process one train/val split**

Add to `scripts/augment_dataset.py`:

```python
import json
import os
import shutil
import logging
from pycocotools.coco import COCO

logger = logging.getLogger(__name__)


def augment_split(
    input_dir: str,
    output_dir: str,
    split: str,
    num_aug: int,
    seed: int,
) -> None:
    """Augment one dataset split (train or val).

    Reads the original COCO JSON + images, generates augmented samples,
    and writes a merged dataset (original + augmented) to output_dir.

    Args:
        input_dir: Path to input dataset root (contains annotations/, images/).
        output_dir: Path to output dataset root.
        split: Split name ('train' or 'val').
        num_aug: Number of augmented images to generate per original image.
        seed: Random seed for reproducibility.
    """
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
            aug_filename = f"AUG_{aug_file_counter:04d}.png"
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
```

- [ ] **Step 2: Run existing tests to verify nothing broke**

Run: `python -m pytest tests/scripts/test_augment_dataset.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/augment_dataset.py
git commit -m "feat: add dataset-level augmentation orchestrator"
```

---

### Task 7: Implement CLI entry point

**Files:**
- Modify: `scripts/augment_dataset.py`

- [ ] **Step 1: Add argparse CLI at the bottom of augment_dataset.py**

Add to the bottom of `scripts/augment_dataset.py`:

```python
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
```

- [ ] **Step 2: Test CLI help**

Run: `python scripts/augment_dataset.py --help`
Expected: Prints usage message showing `--input_dir`, `--output_dir`, `--num_aug`, `--seed` arguments.

- [ ] **Step 3: Commit**

```bash
git add scripts/augment_dataset.py
git commit -m "feat: add CLI entry point for augmentation script"
```

---

### Task 8: End-to-end run and validation

**Files:**
- No new files — runs the script on real data and validates output.

- [ ] **Step 1: Run the augmentation script**

Run:
```bash
python scripts/augment_dataset.py \
    --input_dir data/aiws5.2-dataset-v1 \
    --output_dir data/aiws5.2-dataset-v1-aug \
    --num_aug 3 \
    --seed 42
```
Expected: Script completes with log messages showing progress. No errors.

- [ ] **Step 2: Validate output file structure**

Run:
```bash
ls data/aiws5.2-dataset-v1-aug/annotations/
ls data/aiws5.2-dataset-v1-aug/images/ | wc -l
ls data/aiws5.2-dataset-v1-aug/images/ | grep "^AUG_" | wc -l
```
Expected:
- `train.json` and `val.json` exist
- Total image count ~1584 (396 original + ~1188 augmented)
- AUG_ prefixed files ~1188

- [ ] **Step 3: Validate COCO annotation integrity**

Run:
```python
python3 -c "
import json
from collections import Counter

for split in ['train', 'val']:
    with open(f'data/aiws5.2-dataset-v1-aug/annotations/{split}.json') as f:
        data = json.load(f)
    n_img = len(data['images'])
    n_ann = len(data['annotations'])
    print(f'{split}: {n_img} images, {n_ann} annotations')
    assert n_img == n_ann, f'Mismatch: {n_img} images vs {n_ann} annotations'

    # Check every image has exactly one annotation
    img_ids = {img['id'] for img in data['images']}
    ann_img_ids = {ann['image_id'] for ann in data['annotations']}
    assert img_ids == ann_img_ids, 'Image/annotation ID mismatch'

    # Category distribution
    cat_names = {c['id']: c['name'] for c in data['categories']}
    counts = Counter(a['category_id'] for a in data['annotations'])
    print('  Category distribution:')
    for cat_id in sorted(counts):
        print(f'    {cat_names.get(cat_id, cat_id)}: {counts[cat_id]}')

    # Check augmented annotations have source_image_id
    aug_anns = [a for a in data['annotations'] if a['id'] >= 100000]
    for a in aug_anns:
        assert 'source_image_id' in a, f'Missing source_image_id in ann {a[\"id\"]}'
    print(f'  Augmented annotations with source_image_id: {len(aug_anns)}')
    print()
print('All validations passed.')
"
```
Expected: All assertions pass. Category distribution roughly uniform across 4 classes. Each class ~4x original count.

- [ ] **Step 4: Spot-check a few augmented images visually**

Run:
```bash
python3 -c "
import cv2
import json
import numpy as np
from pycocotools.coco import COCO

coco = COCO('data/aiws5.2-dataset-v1-aug/annotations/train.json')
# Pick first 3 AUG images
aug_imgs = [img for img in coco.dataset['images'] if img['file_name'].startswith('AUG_')][:3]
for img_info in aug_imgs:
    img = cv2.imread(f'data/aiws5.2-dataset-v1-aug/images/{img_info[\"file_name\"]}')
    ann_ids = coco.getAnnIds(imgIds=img_info['id'])
    ann = coco.loadAnns(ann_ids)[0]
    mask = coco.annToMask(ann)
    # Overlay mask on image
    overlay = img.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    cv2.imwrite(f'data/aiws5.2-dataset-v1-aug/debug_{img_info[\"file_name\"]}', overlay)
    print(f'Saved debug overlay: debug_{img_info[\"file_name\"]}')
"
```
Expected: Debug overlay images saved. Open them to verify masks align with objects.

- [ ] **Step 5: Commit**

```bash
git add scripts/augment_dataset.py tests/scripts/test_augment_dataset.py
git commit -m "feat: complete offline data augmentation pipeline"
```
