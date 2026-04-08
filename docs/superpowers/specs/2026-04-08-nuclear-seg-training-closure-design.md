# Nuclear Segmentation Training Closure Design

Date: 2026-04-08
Status: Approved in brainstorming

## Summary

Standardize the long-term training workflow for the nuclear single-agent segmentation/classification head so it can be repeated reliably on `data/aiws5.2_nuclear_workpieces`.

The workflow should:

- train the segmentation/classification head from a pose-init checkpoint
- produce a runtime-compatible single-agent `seg_ckpt`
- use `wandb` as the primary experiment tracking system
- export only `latest` and `best` checkpoints by default
- choose `best` based on validation mask quality, with segmentation quality taking priority over classification
- provide both a stable shell-script entrypoint and an equivalent full CLI template

This design covers only the segmentation/classification training closure. It does not redesign ScoreNet, EnergyNet, or ScaleNet training.

## Goals

- Make segmentation training on the nuclear dataset repeatable and predictable.
- Standardize the dataset contract for `data/aiws5.2_nuclear_workpieces`.
- Replace TensorBoard-first training logs with `wandb`.
- Make `best` checkpoint selection depend on validation mask quality, not only loss.
- Export runtime-ready segmentation checkpoints that can be consumed by the single-agent nuclear inference path.
- Provide a stable daily-use script plus a transparent CLI equivalent.

## Non-Goals

- Redesigning the entire training framework for all agent types.
- Building a general experiment platform for score, energy, and scale retraining.
- Introducing COCO mask AP as the default segmentation ranking metric.
- Adding pose-based validation into segmentation checkpoint selection in the first iteration.
- Supporting segmentation full-checkpoint resume/eval through `pretrained_score_model_path`.

## Current Situation

The repository already supports nuclear segmentation training through:

- `cfg.agent_type == 'segmentation'`
- `cfg.dataset_type == 'nuclear'`
- `cfg.nuclear_data_path`
- pose-init loading through `cfg.pretrained_score_model_path`

The current design is functional, but it is not yet a stable long-term training product because:

- the main recommended entrypoint is not standardized
- the existing `scripts/train_seg.sh` still reflects old assumptions
- training records are written through TensorBoard instead of `wandb`
- checkpoint export is epoch-oriented instead of `best/latest` oriented
- segmentation validation records losses, but lacks a stable epoch-level `mask IoU / Dice` selection contract
- experiment outputs do not yet have a single, explicit naming and artifact policy

## Dataset Contract

### Root Layout

The nuclear segmentation training root is a fixed directory, typically:

`data/aiws5.2_nuclear_workpieces`

Expected layout:

```text
data/aiws5.2_nuclear_workpieces/
  images/
  annotations/
    train.json
    val.json
  depth/                 # optional for segmentation training, recommended for downstream inference
  meta/                  # optional for segmentation training, recommended for downstream inference
  isat_annotations/      # optional source annotations
```

### Mandatory Inputs

Segmentation/classification training requires only:

- `images/`
- `annotations/train.json`
- `annotations/val.json`

### Annotation Semantics

The annotation files must be COCO-style instance segmentation JSON files.

The category names must map cleanly onto the six in-repo nuclear classes:

- `盖板`
- `方管`
- `喇叭口`
- `H型钢`
- `槽钢`
- `坡口`

Extra pose-related fields may remain in the annotations, but they are not required for segmentation training.

## Data Validation Design

Before training, the workflow should provide a dedicated dataset-check step, independent from the trainer itself.

Recommended command shape:

```bash
python scripts/check_nuclear_seg_dataset.py --nuclear_data_path ./data/aiws5.2_nuclear_workpieces
```

The checker should validate:

- `train.json` and `val.json` exist
- every referenced image file exists
- category names map to the six supported classes
- there are no duplicated image ids or duplicated file names across splits
- there are no empty-image surprises that violate training assumptions
- masks and image references are structurally valid

The checker should also print a compact summary:

- train image count
- val image count
- instance count per class
- per-image instance-count distribution

This validation step is a standard preflight step for every new annotation revision.

## Training Surface

### Daily-Use Entrypoint

Create a new stable training script:

`scripts/train_seg_single_agent.sh`

This becomes the recommended segmentation training entrypoint.

### CLI Transparency

The trainer remains the actual execution engine:

`python runners/trainer.py ...`

The shell script only wraps common parameters. Every script run must map cleanly to an equivalent full CLI command.

### Existing Script Handling

The current `scripts/train_seg.sh` should no longer be the primary entrypoint.

Recommended handling:

- keep it as a thin compatibility wrapper, or
- deprecate it clearly and redirect users to `train_seg_single_agent.sh`

The important point is to keep one canonical training surface for future runs.

## Configuration Semantics

### Fixed Training Meaning

For segmentation/classification training, the meaning of the relevant knobs is fixed:

- `agent_type=segmentation`: user-facing training mode
- `dataset_type=nuclear`: use the nuclear dataset loader
- `nuclear_data_path`: root dataset directory
- `pretrained_score_model_path`: pose-init checkpoint only
- `enable_segmentation=True`: build the segmentation-enabled main model

### Pose-Init Rule

`pretrained_score_model_path` is only used to seed the frozen pose path and related score-style model weights before segmentation training begins.

It is not a segmentation full-resume checkpoint path.

### Build Rule

Segmentation training internally builds a score-style architecture with `enable_segmentation=True` so the resulting checkpoint contains the pose path required by the single-agent inference runtime.

### Trainable Parameters

Segmentation training updates only:

- `dino_wrapper.seg_blocks`
- `dino_wrapper.seg_norm`
- `dino_wrapper.query_embed`
- `eomt_head`

The shared DINO stem, frozen pose tail, point encoder, and pose score path remain frozen.

## Experiment Naming And Artifact Layout

Each segmentation experiment uses a single explicit experiment name:

`EXP_NAME=SegNetSingleAgent_<dataset_version>_<img_size>_<tag>`

Example:

`SegNetSingleAgent_aiws5p2_224_v1`

### Local Artifact Layout

Segmentation experiments should write to:

- checkpoints: `results/ckpts/<EXP_NAME>/`
- visualization artifacts: `results/vis_seg/<EXP_NAME>/`

Unlike the previous default, `results/logs/<EXP_NAME>/` is not the primary experiment record surface.

### Required Local Files

By default, the checkpoint directory should contain:

- `latest.pth`
- `best.pth`
- `config.json`
- `summary.json`

The workflow should not keep every epoch checkpoint by default.

## W&B Design

### Role

`wandb` becomes the primary experiment tracking surface for segmentation training.

It records:

- run configuration
- training losses
- validation losses
- validation mask quality metrics
- best/latest checkpoint metadata
- optional qualitative mask visualizations

### Mode Support

Both online and offline `wandb` modes are supported.

The training script exposes a mode switch, but the default is online mode.

### Recommended Script Variables

The stable shell script should expose at least:

- `EXP_NAME`
- `NUCLEAR_DATA_PATH`
- `POSE_INIT_CKPT`
- `DEVICE` or `CUDA_VISIBLE_DEVICES`
- `BATCH_SIZE`
- `N_EPOCHS`
- `NUM_WORKERS`
- `IMG_SIZE`
- `WANDB_MODE`
- `WANDB_PROJECT`
- `WANDB_ENTITY` (optional)

## Metrics And Best Checkpoint Policy

### Training Metrics

Training should continue to log the existing segmentation losses:

- `train/cls_loss`
- `train/mask_loss`
- `train/dice_loss`
- `train/total_loss`
- `train/lr`

### Validation Metrics

Validation should log both losses and real mask-quality metrics:

- `val/cls_loss`
- `val/mask_loss`
- `val/dice_loss`
- `val/total_loss`
- `val/mask_iou`
- `val/mask_dice`

Classification-specific quality metrics may also be logged, but they are auxiliary rather than primary.

### Best Checkpoint Rule

The default `best` selection policy is segmentation-first.

Primary metric:

- `val/mask_iou`

Tie-breaker:

- `val/mask_dice`

Classification quality does not drive `best` selection.

### Checkpoint Export Rule

At each evaluation point:

- overwrite `latest.pth`
- update `summary.json`
- if `val/mask_iou` improves, overwrite `best.pth`

`summary.json` should include at least:

- experiment name
- dataset path
- pose-init checkpoint path
- best epoch
- latest epoch
- best mask IoU
- best mask Dice
- image size
- number of queries
- query injection layer

## Validation Logic

### Metric Choice

The default segmentation quality metrics are epoch-level mask Dice and IoU.

COCO mask AP is intentionally not the default because the goal here is a lightweight, stable training closure rather than a heavy evaluation stack.

### Evaluation Shape

During validation:

- compute the existing segmentation losses
- compute epoch-level `mask_iou` and `mask_dice`
- log them to `wandb`
- use them to decide `best/latest`

The metric computation should match the training objective closely enough that a higher `mask_iou` genuinely means a better segmentation checkpoint for downstream use.

## Inference Handoff

The long-term contract between segmentation training and nuclear inference is:

training output:

- `results/ckpts/<EXP_NAME>/best.pth`

runtime inputs:

- `seg_ckpt=results/ckpts/<EXP_NAME>/best.pth`
- `energy_ckpt=results/ckpts/EnergyNet/energynet.pth`
- `scale_ckpt=results/ckpts/ScaleNet/scalenet.pth`

Recommended inference shape:

```bash
python runners/infer_nuclear_full.py \
  --nuclear_data_path ./data/aiws5.2_nuclear_workpieces \
  --seg_ckpt ./results/ckpts/<EXP_NAME>/best.pth \
  --energy_ckpt ./results/ckpts/EnergyNet/energynet.pth \
  --scale_ckpt ./results/ckpts/ScaleNet/scalenet.pth \
  --split val \
  --output_dir ./results/full_pipeline_<EXP_NAME>
```

## Recommended Full CLI Template

The stable script should expand into a trainer command equivalent to:

```bash
python runners/trainer.py \
  --dataset_type nuclear \
  --nuclear_data_path ./data/aiws5.2_nuclear_workpieces \
  --agent_type segmentation \
  --pretrained_score_model_path ./results/ckpts/ScoreNet/scorenet.pth \
  --dino pointwise \
  --img_size 224 \
  --num_queries 50 \
  --query_inject_layer -4 \
  --num_object_classes 6 \
  --batch_size 8 \
  --n_epochs 100 \
  --eval_freq 1 \
  --lr 1e-4 \
  --lr_decay 0.99 \
  --warmup 50 \
  --num_workers 4 \
  --log_dir SegNetSingleAgent_aiws5p2_224_v1 \
  --is_train
```

## End-To-End Workflow

The standard long-term workflow is:

1. run dataset preflight checks
2. launch segmentation training from `train_seg_single_agent.sh`
3. track progress in `wandb`
4. export `latest.pth` and `best.pth`
5. run a segmentation visualization sanity check on validation data
6. hand `best.pth` to `runners/infer_nuclear_full.py` as the runtime `seg_ckpt`

## Error Handling

- Fail fast if `pretrained_score_model_path` is missing for segmentation training.
- Fail fast if the dataset root is missing `train.json` or `val.json`.
- Fail fast if the category mapping is inconsistent with the six supported classes.
- Fail fast if runtime is asked to treat `pretrained_score_model_path` as a segmentation full-resume checkpoint.
- Fail fast if a produced segmentation checkpoint does not satisfy the single-agent runtime checkpoint contract.

## Acceptance Criteria

This design is considered fulfilled when the repository supports the following stable loop:

1. validate `data/aiws5.2_nuclear_workpieces`
2. train segmentation/classification head from a pose-init checkpoint
3. log training and validation to `wandb`
4. export only `latest` and `best`
5. select `best` by validation `mask_iou`
6. load `best.pth` directly into the single-agent nuclear runtime
