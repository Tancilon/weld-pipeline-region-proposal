# Train Split Category-Balanced Augmentation with Geometric Consistency

**Date**: 2026-04-22
**Scope**: Extend `scripts/augment_dataset.py` so it can balance the train split of
`data/aiws5.2-dataset` to a user-specified per-category sample count (default 900),
while keeping RGB / mask / depth / camera intrinsics geometrically consistent so the
augmented data can feed the full seg+pose pipeline once pose GT becomes available.

---

## 1. Motivation

`data/aiws5.2-dataset` is an AIWS 5.2 COCO-style dataset for 6 nuclear-workpiece
categories. Current train-split class distribution is heavily skewed:

| category | train | val | test |
|---|---|---|---|
| 盖板 | 500 | 248 | 84 |
| 方管 | 324 | 162 | 54 |
| 喇叭口 | 144 | 72 | 24 |
| H型钢 | 72 | 36 | 12 |
| 槽钢 | 0 | 0 | 0 |
| 坡口 | 0 | 0 | 0 |

The existing `scripts/augment_dataset.py` applies a fixed `--num_aug` multiplier
uniformly to every source image; it cannot balance categories. It also only
touches `images/` and `annotations/`, leaving `depth/` and `meta/` (camera
intrinsics) untouched — acceptable for seg-only training (`datasets_nuclear.py`
never reads `intrinsics` in seg mode) but incorrect for the full pipeline, which
unprojects depth using the per-image intrinsics matrix K.

This design extends the script to:

1. Compute per-category augmentation quotas so every present category hits a
   target sample count (default 900) in the train split.
2. Produce augmented samples as geometrically consistent quadruples
   (RGB, mask, depth, K) so the full pipeline can consume them directly.

Val / test splits are **not** augmented — they must reflect the real data
distribution for unbiased evaluation.

---

## 2. Scope

### 2.1 In scope

- Category balancing for the train split only.
- RGB + mask + depth synchronized geometric augmentation.
- Per-sample `meta/*.json` generation with correct intrinsics.
- Augmentation operator set restricted to pinhole-preserving transforms.

### 2.2 Out of scope

- Val / test augmentation (pass through unchanged).
- Down-sampling categories that already exceed the target (leave as-is).
- Augmenting categories with zero samples (槽钢, 坡口) — impossible to
  synthesize without source data.
- Pose-aware augmentation of `rotation` / `translation` fields — pose GT does
  not exist in this dataset yet. When pose GT is added later, the geometric
  consistency established here will make that extension straightforward.

---

## 3. Output layout

```
output_dir/
  images/        # copy of originals + AUG_train_NNNN.png
  depth/         # copy of originals + AUG_train_NNNN.exr
  meta/          # copy of originals + AUG_train_NNNN.json
  annotations/
    train.json   # original entries + augmented entries
    val.json     # copied as-is
    test.json    # copied as-is
```

- Originals are copied via `shutil.copy2` to keep mtimes.
- If `output_dir == input_dir` the script aborts with a clear error.
- Augmented filenames use `AUG_train_{counter:04d}` prefix so future extensions
  to other splits remain possible without a naming collision.

---

## 4. Per-category quota calculation

**CLI**: new `--target_per_category <int>` flag (default 900). When set, the
legacy `--num_aug` flag is ignored. When absent, fall back to the legacy
uniform-multiplier behavior.

**Algorithm** (applied only to the train split):

```python
for category in train:
    count = current_sample_count(category)
    if count == 0:
        log.warning(f"{category}: 0 samples, cannot augment, skipping")
        continue
    if count >= target:
        log.info(f"{category}: {count} >= {target}, skipping (no down-sampling)")
        continue

    need = target - count
    src_images = all_source_images_of(category)
    base = need // len(src_images)
    remainder = need % len(src_images)

    quotas = {img_id: base for img_id in src_images}
    for img_id in random.sample(src_images, remainder):
        quotas[img_id] += 1
```

**Expected outcome** (target=900):

| category | need | src images | base | remainder | effective per image |
|---|---|---|---|---|---|
| 盖板 | 400 | 500 | 0 | 400 | 80% of images get +1 |
| 方管 | 576 | 324 | 1 | 252 | each +1, 78% get +2 |
| 喇叭口 | 756 | 144 | 5 | 36 | each +5, 25% get +6 |
| H型钢 | 828 | 72 | 11 | 36 | each +11, 50% get +12 |

Randomness (which images get the +1 remainder, and each augmentation's per-op
parameters) is seeded by `--seed` for reproducibility.

---

## 5. Augmentation operator set

Geometric ops are hand-rolled so we have access to the exact transform
parameters and can propagate them to the intrinsics K. Color ops are delegated
to `albumentations` and applied to RGB only.

### 5.1 Geometric ops (pinhole-preserving)

| op | probability | parameters | rationale |
|---|---|---|---|
| HorizontalFlip | 0.5 | — | mirror about vertical axis; K updates cleanly |
| RandomResizedCrop | 0.5 | scale ∈ [0.8, 1.0], aspect ratio locked to source W/H (no distortion) | equivalent to crop-then-resize, both pinhole-preserving |
| Translate | 0.3 | translate_percent ∈ [-0.05, 0.05] independently on x, y | shifts principal point |

### 5.2 Geometric ops explicitly excluded

- **Rotate**: rotation about the image center is not a valid camera motion
  unless the axis passes exactly through the principal point. The dataset's
  (cx=970.16, cy=542.53) ≠ (W/2, H/2). Dropping Rotate eliminates a silent
  source of pinhole-model violation.
- **Affine shear**: breaks the pinhole model entirely; no valid K update
  exists for a sheared image.

### 5.3 Color ops (RGB only, K unchanged)

Applied after the geometric stage:

| op | probability | parameters |
|---|---|---|
| RandomBrightnessContrast | 0.5 | brightness_limit=0.2, contrast_limit=0.2 |
| HueSaturationValue | 0.4 | hue ±10, sat ±20, val ±20 |
| GaussNoise | 0.3 | std_range (0.02, 0.05) |
| GaussianBlur | 0.2 | blur_limit (3, 7) |

### 5.4 Border fill policy (confirmed during brainstorm)

Translation exposes out-of-frame regions. We use `cv2.BORDER_CONSTANT` with
value 0 for all of RGB, mask, and depth. Rationale: depth=0 is the canonical
"sensor invalid" marker, and a black RGB border paired with zero-depth is a
realistic sensor-blind-region pattern. Alternatives (border reflect, micro-crop
rescale) were rejected as either physically implausible (reflected depth) or
as adding extra complexity without clear benefit.

---

## 6. Intrinsic propagation

Let `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` and `W, H` be current image
size. Each op updates K as follows:

### 6.1 Horizontal flip
```
cx' = W - 1 - cx
fx, fy, cy  unchanged
```

### 6.2 Crop (origin x0, y0, size W_crop × H_crop)
```
cx' = cx - x0
cy' = cy - y0
fx, fy  unchanged
W' = W_crop,  H' = H_crop
```

### 6.3 Resize (W,H → W',H')
```
sx = W' / W,  sy = H' / H
fx' = fx * sx
fy' = fy * sy
cx' = (cx + 0.5) * sx - 0.5
cy' = (cy + 0.5) * sy - 0.5
```

(The `+ 0.5 / − 0.5` correction aligns pixel-center conventions under scaling;
dropping it biases the principal point by half a sub-pixel which compounds on
each resize.)

### 6.4 Translate (tx, ty pixels, positive = right/down)
```
cx' = cx + tx
cy' = cy + ty
fx, fy  unchanged
```

### 6.5 Composition

`RandomResizedCrop` is implemented as crop (6.2) followed by resize (6.3) back
to the original `(W, H)`. Applying ops in a fixed order —
`[flip, crop+resize, translate]` — and chaining the K updates gives the final
K' for the augmented sample.

---

## 7. Depth synchronization

Depth is stored as single-channel `.exr` in meters, clamped to
`MAX_DEPTH_METERS = 4.0` (per `datasets_nuclear.py:34`).

### 7.1 EXR I/O

`scripts/augment_dataset.py` must set `os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')`
**before** importing `cv2`, matching what the dataset loader already assumes.

### 7.2 Per-op depth transform

Uses the same geometric parameters as RGB / mask:

| op | depth operation |
|---|---|
| Flip | `np.fliplr(depth)`; values unchanged |
| Crop | array slicing `depth[y0:y0+H_crop, x0:x0+W_crop]` |
| Resize | `cv2.resize(..., interpolation=cv2.INTER_NEAREST)` — **nearest only**, to avoid flying pixels from interpolating across depth discontinuities |
| Translate | `cv2.warpAffine(..., borderMode=BORDER_CONSTANT, borderValue=0)` |

### 7.3 Invalid depth sanitization

After geometric transform, re-apply the dataset loader's sanitization
(`~isfinite | <0 | >4.0` → 0). This matches
[datasets_nuclear.py:186-191](../../datasets/datasets_nuclear.py#L186-L191)
and keeps the augmented data indistinguishable from cleanly-loaded originals.

### 7.4 Multi-channel EXR

If a source depth file is 3-channel (some EXR writers emit RGB-packed depth
with only ch0 valid), the script reads `depth[:,:,0]` and writes back
single-channel. This matches how the loader collapses multi-channel depth.

---

## 8. Meta JSON schema

Each augmented sample writes `meta/AUG_train_NNNN.json` with the same schema
as the source meta files:

```json
{
  "camera": {
    "intrinsics": {
      "fx": <updated>,
      "fy": <updated>,
      "cx": <updated>,
      "cy": <updated>,
      "width": <post-aug W>,
      "height": <post-aug H>
    }
  },
  "annotation": {
    "class_name": "<copied from source meta>",
    "dimensions": [<copied from source meta>]
  }
}
```

- `class_name` and `dimensions` are physical properties of the object, not the
  image — copied verbatim from the source.
- `width` / `height` reflect the post-augmentation image size. For the current
  operator set these are always 1920×1080 (RandomResizedCrop resizes back to
  original), but the code reads them from the actual output shape so future
  multi-scale ops work without framework changes.

---

## 9. COCO annotation update

Each augmented sample adds one `images` entry and one `annotations` entry to
`train.json`.

### 9.1 images entry
```json
{
  "license": "",
  "url": "",
  "file_name": "AUG_train_NNNN.png",
  "height": <post-aug H>,
  "width": <post-aug W>,
  "date_captured": "",
  "id": <new, starting at 100000>
}
```

### 9.2 annotations entry
```json
{
  "iscrowd": false,
  "image_id": <matches images entry id>,
  "image_name": "AUG_train_NNNN.png",
  "category_id": <same as source>,
  "id": <new, starting at 100000>,
  "segmentation": [<polygons derived from transformed mask>],
  "area": <polygon area post-transform>,
  "bbox": <bbox post-transform>,
  "source_image_id": <original img id, for traceability>
}
```

Mask → polygon conversion reuses the existing `mask_to_polygons` helper
(min area 50 px, CHAIN_APPROX_SIMPLE).

### 9.3 Safety re-sampling

The existing `is_augmentation_safe` check (polygon area ≥ 10% of source area)
is retained. We extend it to also check that post-transform depth has at
least 10% as many non-zero pixels (i.e. valid-depth pixels, after §7.3
sanitization) as the source depth map. On failure, re-sample all geometric
parameters up to 5 times; if all retries fail, log a warning and move on
without inflating that quota slot.

---

## 10. Architecture

```
scripts/augment_dataset.py
├── [new] GeomParams dataclass
│       flip: bool
│       crop_box: (x0, y0, W_crop, H_crop) | None
│       resize_to: (W', H') | None
│       translate: (tx, ty)
│
├── [new] apply_geom(rgb, mask, depth, K, params)
│       → (rgb', mask', depth', K')
│       Applies ops in canonical order: flip → crop → resize → translate
│
├── [new] update_intrinsics(K, W, H, params) → (K', W', H')
│       Pure function, thoroughly unit-tested
│
├── [new] sample_geom_params(W, H, rng) → GeomParams
│       Stochastic sampling driven by the per-op probabilities in §5.1
│
├── [kept] build_color_transform() → A.Compose
│       RGB-only albumentations pipeline per §5.3
│
├── [kept, modified] mask_to_polygons, compute_bbox_area, is_augmentation_safe
│       is_augmentation_safe gains a depth-validity check (§9.3)
│
├── [new] compute_quotas(coco, target, rng) → dict[img_id, int]
│       Per-category quota calculation per §4
│
├── [rewritten] augment_split(input_dir, output_dir, target, seed)
│       Orchestrates the full pipeline:
│       1. Load COCO, copy originals (images/depth/meta) to output
│       2. Compute quotas
│       3. For each src image with quota > 0:
│            For each slot in quota:
│              sample GeomParams → apply_geom → color transform
│              → safety check (retry up to 5×)
│              → write png/exr/json, append COCO entries
│       4. Merge and write train.json; copy val.json, test.json
│
└── main()
      Parses CLI: --input_dir, --output_dir, --target_per_category (new),
                  --num_aug (legacy), --seed
      Dispatches to augment_split (train only when target is set; all splits
      when legacy num_aug is used, matching old behavior)
```

---

## 11. Testing strategy

Tests live in `tests/scripts/test_augment_dataset.py`.

### 11.1 Unit tests (fast, no real data)

Per §6 formulas, assert each op's K update in isolation:
- `test_flip_intrinsics`
- `test_crop_intrinsics`
- `test_resize_intrinsics` (check the half-pixel correction explicitly)
- `test_translate_intrinsics`
- `test_compose_crop_then_resize` — verify `RandomResizedCrop` composition
  matches step-by-step computation

### 11.2 Geometric consistency round-trip (core correctness gate)

This is the most important test — directly validates that augmented samples
are geometrically equivalent to originals, which is the whole point of
approach B.

For each of `{flip, crop+resize, translate}`:
1. Load a real sample (e.g., `f101_dajiaodu1_1`).
2. Pick N well-distributed foreground pixels from the source mask.
3. Unproject them to the camera frame using source K and source depth → `P_src`.
4. Apply the geometric transform to (RGB, mask, depth, K), tracking each
   sampled pixel's new location.
5. Unproject the corresponding post-transform pixels using the new K and new
   depth → `P_aug`.
6. Assert `||P_aug - P_src|| / object_scale < 0.05` (tolerance budgeted for
   nearest-neighbor depth resampling).

Three test cases: `test_roundtrip_flip`, `test_roundtrip_crop_resize`,
`test_roundtrip_translate`.

### 11.3 End-to-end integration

Build a synthetic mini COCO split (2–3 images, 4 categories), run
`augment_split(target=10)`, and assert:
- Output dir contains `images/`, `depth/`, `meta/`, `annotations/`.
- Per-category counts equal the target.
- Every AUG sample has matching png / exr / json.
- Every `meta/AUG_*.json` matches the source meta schema.
- `pycocotools.COCO(...)` loads the output `train.json` without error.
- Every annotation's `source_image_id` exists in the original id set.

### 11.4 Manual smoke verification (post-run)

After a full train-split run:
- Print final per-category counts, verify they hit 900.
- Randomly sample 5 AUG images, visualize RGB + mask overlay + depth colormap
  to check visual consistency.
- Load the output via `NuclearWorkpieceDataset(mode='full')` for one batch;
  verify `intrinsics` and `depth` fields are populated and sensible.

Smoke steps live in this design doc as acceptance criteria, not in the
automated test suite.

---

## 12. Risks and open questions

- **Small-class overfitting**: H型钢 receives 11–12 augmentations per source
  image. The augmentation parameter space is large enough that no two samples
  are identical, but the underlying information content is still bounded by
  72 source images. Augmentation is not a substitute for real data — this is
  an acceptable interim measure to get balanced training, not a silver bullet.
- **槽钢 / 坡口**: zero samples. The script logs a warning and skips; filling
  these categories requires new data collection, not augmentation.
- **Pose GT absence**: once pose labels are added to the COCO annotations,
  the geometric consistency established here means pose updates can be bolted
  on by (a) flipping translation.x / rotation for flip, (b) subtracting crop
  origin from translation (already in principal point), (c) leaving rotation
  alone for pure translations. That follow-up work is not part of this spec.

---

## 13. Acceptance criteria

1. Running
   ```
   python scripts/augment_dataset.py \
       --input_dir data/aiws5.2-dataset \
       --output_dir data/aiws5.2-dataset-aug \
       --target_per_category 900 \
       --seed 42
   ```
   produces an output dataset where the train split contains exactly 900
   samples for each of the four populated categories (盖板 / 方管 / 喇叭口 /
   H型钢), with val and test copied verbatim.
2. All unit tests and round-trip tests pass (`pytest tests/scripts/test_augment_dataset.py`).
3. `NuclearWorkpieceDataset(mode='full')` successfully loads one batch from
   the augmented `train.json` with non-null `intrinsics` and `depth`.
4. The design's geometric-consistency invariant holds: for any augmented
   sample, unprojecting foreground pixels via its K and depth yields the same
   3D points (within 5%) as unprojecting the corresponding source pixels.
