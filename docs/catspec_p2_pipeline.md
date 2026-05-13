# CatSpec-Pose P2 Pipeline

This document records the reproducible P2 commands for dataset preprocessing,
training, inference, and evaluation.

## Smoke Commands

Use these on the current category-level assets. They keep outputs under
`results/catspec/p2_*`.

```bash
python scripts/preprocess_catspec_p2.py \
  --dataset-root /home/dq/mnt/localgit/aiws5.2/datasets/obj_share_models \
  --output-dir results/catspec/p2_preprocess \
  --force

python scripts/train_catspec_p2.py \
  --manifest results/catspec/p2_preprocess/catspec_p2_manifest.json \
  --output-dir results/catspec/p2_train \
  --split all \
  --val-split all \
  --epochs 120 \
  --batch-size 4 \
  --seed 7 \
  --device cpu

python scripts/infer_catspec_p2.py \
  --manifest results/catspec/p2_preprocess/catspec_p2_manifest.json \
  --checkpoint results/catspec/p2_train/catspec_p2_best.pt \
  --output-dir results/catspec/p2_infer \
  --split all \
  --device cpu

python scripts/evaluate_catspec_p2.py \
  --predictions results/catspec/p2_infer/catspec_p2_predictions.jsonl \
  --manifest results/catspec/p2_preprocess/catspec_p2_manifest.json \
  --output-dir results/catspec/p2_eval
```

## Full-Scale Entry Points

Use the same commands without category filtering. `preprocess_catspec_p2.py`
supports `--category`, `--max-samples`, and cached reruns. Omit `--force` to
reuse an existing manifest if all expected files exist.
`train_catspec_p2.py` can also omit `--manifest`; in that case it uses
`--dataset-root` and `--preprocess-output-dir` to create or reuse the P2
manifest before training.

```bash
python scripts/preprocess_catspec_p2.py \
  --dataset-root /home/dq/mnt/localgit/aiws5.2/datasets/obj_share_models \
  --output-dir results/catspec/p2_preprocess

python scripts/train_catspec_p2.py \
  --manifest results/catspec/p2_preprocess/catspec_p2_manifest.json \
  --output-dir results/catspec/p2_train \
  --split train \
  --val-split val \
  --epochs 120 \
  --batch-size 32 \
  --lr 0.03 \
  --seed 7 \
  --device auto \
  --num-workers 4

python scripts/infer_catspec_p2.py \
  --manifest results/catspec/p2_preprocess/catspec_p2_manifest.json \
  --checkpoint results/catspec/p2_train/catspec_p2_best.pt \
  --output-dir results/catspec/p2_infer \
  --split test \
  --device auto \
  --num-workers 4

python scripts/evaluate_catspec_p2.py \
  --predictions results/catspec/p2_infer/catspec_p2_predictions.jsonl \
  --manifest results/catspec/p2_preprocess/catspec_p2_manifest.json \
  --output-dir results/catspec/p2_eval
```

## Input Boundary

`_weld.obj` files are used during preprocessing and evaluation as
`reference_weld_path`. P2 dataset `model_inputs` contain only spec token ids,
token mask, and source workpiece geometry summary. They do not include
`reference_weld_path` or weld mesh geometry.
