# Segmentation Classification-Balancing Design

## Goal

Improve the single-agent segmentation checkpoint training flow so the
classification head does not collapse to the majority class on imbalanced
nuclear datasets, while keeping segmentation quality as the primary objective.

The first version should:

- generate foreground class weights automatically from the current train split
- apply those weights only to the classification loss
- keep segmentation metrics as the primary `best.pth` selection signal
- use matched classification accuracy only as a late tie-breaker
- record the generated weights and class statistics for later inspection

It should not introduce resampling in the first iteration.

## Problem Statement

Current evidence from `results/ckpts/SegNet/best.pth` shows that inference on
the validation set predicts the foreground class `盖板` for nearly every sample,
including samples whose ground-truth class is `方管`, `喇叭口`, or `H型钢`.

The root cause is consistent with the current training setup:

- training data is heavily imbalanced toward `盖板`
- the classification loss currently does not rebalance foreground classes
- `best.pth` is selected only by `mask_iou` and `mask_dice`
- classification collapse therefore does not prevent a checkpoint from being
  selected as `best`

## Scope

This design only covers the segmentation/classification training closure for
the single-agent nuclear model.

In scope:

- automatic class-weight generation from `train.json`
- passing generated class weights into the EoMT classification loss
- recording class counts, zero-count classes, and effective weights
- extending `best` tie-break rules to consider `cls_acc_matched` only after
  segmentation metrics

Out of scope:

- sampler-level class rebalancing or weighted resampling
- changing inference-time postprocessing rules
- changing the segmentation architecture
- solving classes with zero training samples beyond explicit reporting

## Recommended Approach

Use automatically generated smoothed inverse-frequency class weights, derived
from the train split instance counts at segmentation-training startup.

The recommended rule is:

1. count foreground instances per class from `train.json`
2. for classes with `count > 0`, compute `raw_weight = 1 / sqrt(count)`
3. normalize the foreground weights so their mean is approximately `1.0`
4. clip the normalized weights into a bounded range
5. set any zero-count class weight to `0.0`
6. leave the existing `no-object` weight as a separate parameter

This keeps the mechanism data-adaptive without hard-coding a weight vector for
one dataset version, and it is substantially more stable than a pure `1/count`
rule for the current long-tail distribution.

## Architecture And Boundaries

### Data statistics source

The source of truth is the segmentation training dataset annotation file for
the `train` split. The statistics must be derived from the current training
data, not from a hard-coded table and not from validation data.

### Trainer responsibility

`runners/trainer.py` owns class-count collection and class-weight generation.
This is the correct layer because it already owns segmentation training setup
and has access to the current dataset path and train split semantics.

### Criterion responsibility

`networks/eomt_head.py` should consume an externally provided foreground class
weight vector when computing classification cross-entropy. The criterion should
not read annotation files directly and should not infer dataset statistics on
its own.

### Agent responsibility

`networks/posenet_agent.py` should instantiate `EoMTCriterion` with the
generated class weights from the training config and continue to expose
segmentation evaluation metrics, including `cls_acc_matched`.

### Inference responsibility

Inference remains unchanged. The trained checkpoint contains updated classifier
weights, but the runtime path must not regenerate class weights.

## Weight Generation Details

### Foreground classes

The generated weight vector covers the six foreground classes in
`datasets/datasets_nuclear.py`.

### Zero-count classes

Classes with zero training instances must receive weight `0.0`.

Reason:

- a zero-count class has no supervision signal
- assigning an extreme positive weight would not create signal and would only
  increase instability
- the correct behavior for the first version is explicit reporting, not forced
  extrapolation

### Clipping

The first version should use bounded clipping to prevent unstable weights from
extreme class imbalance.

Recommended defaults:

- `min_class_weight = 0.25`
- `max_class_weight = 4.0`

These values are intentionally conservative. The design goal is to stop the
classifier from collapsing to the dominant class, not to over-correct toward
rare classes.

### Config semantics

The weight-generation mechanism should be automatic by default, with no manual
foreground weight vector required for normal training.

If a manual override is added later, it should be an explicit advanced option,
not the default path.

## Best Checkpoint Selection

The project still prioritizes segmentation quality over classification quality.
Therefore the `best` selection rule should remain segmentation-first.

Recommended ordering:

1. higher `mask_iou`
2. if tied, higher `mask_dice`
3. if still tied, higher `cls_acc_matched`

This keeps the original segmentation-first intent intact while preventing a
classification-collapsed checkpoint from winning a perfect tie on segmentation
metrics.

`cls_acc_matched` should not become a co-primary objective in the first
version.

## Observability

The training run must record enough information to explain future behavior on
new datasets.

Required outputs:

- per-class train instance counts
- list of zero-count classes
- final generated foreground class weights
- validation `cls_acc_matched`
- `best_cls_acc_matched`
- `latest_cls_acc_matched`

These should be surfaced in:

- console startup logs
- `wandb`
- `results/ckpts/<EXP_NAME>/config.json`
- `results/ckpts/<EXP_NAME>/summary.json`

## Validation Requirements

The implementation should be accepted only if all of the following hold:

1. class weights are generated automatically from the train split
2. zero-count classes are reported explicitly and assigned zero foreground
   weight
3. classification cross-entropy consumes the generated foreground weights
4. `best.pth` selection keeps segmentation-first ordering and uses
   `cls_acc_matched` only as the final tie-break
5. training artifacts record class counts, generated weights, and matched
   classification accuracy
6. the trained checkpoint no longer collapses to predicting `盖板` for nearly
   every validation image when the dataset contains multiple supervised classes

## Non-Goals For First Version

The first version deliberately does not include:

- weighted random sampling
- focal loss
- class-balanced loss based on effective-number formulas
- separate classification-only fine-tuning
- joint optimization of checkpoint selection with a weighted mask/class score

If automatic class weighting alone is insufficient, those can be considered in
the next design iteration, but they should not be bundled into the first fix.
