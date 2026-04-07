# Single-Agent Segmentation And Pose Design

Date: 2026-04-07
Status: Approved in brainstorming

## Summary

Replace the current dual-agent nuclear inference path with a single main agent that performs both segmentation and 6D pose inference from one new `seg_ckpt`. The new checkpoint is initialized from an existing `pose_ckpt` during segmentation training, but runtime inference no longer accepts or requires a separate `pose_ckpt`.

The core architectural change is to stop injecting query tokens into the original DINO backbone tail that serves pose estimation. Instead, split the DINO backbone at `inject_layer` into:

- a frozen shared stem
- a frozen pose tail from the original pretrained backbone
- a trainable segmentation tail cloned from the original pose tail

Query tokens are injected only into the cloned segmentation tail. Pose estimation continues to consume features from the untouched frozen pose path.

## Goals

- Remove `pose_ckpt` from nuclear runtime inference.
- Remove the separate pure-pose score agent from `runners/infer_nuclear_full.py`.
- Keep a single main checkpoint, `seg_ckpt`, as the source of both segmentation and pose features.
- Prefer a simple and robust implementation over strict pose parity with the current dual-agent pipeline.

## Non-Goals

- Merging `energy_ckpt` or `scale_ckpt` into the main checkpoint.
- Preserving direct compatibility with old segmentation checkpoints.
- Forcing segmentation and pose to share the same DINO forward cache during inference.
- Introducing additional joint pose losses into segmentation training in the first iteration.

## Current Problem

The current segmentation wrapper modifies the original DINO backbone by injecting query tokens into the last several transformer blocks. At the same time, the pose branch uses patch tokens taken before query injection. This creates two problems:

1. The pose branch no longer uses the original final DINO features.
2. The segmentation checkpoint cannot safely replace the pose checkpoint in runtime inference.

This is why the current nuclear pipeline uses:

- a segmentation agent with `seg_ckpt`
- a separate pure GenPose2 pose agent with `pose_ckpt`

## Considered Approaches

### Approach 1: Shared Stem + Frozen Pose Tail + Trainable Seg Tail

Use the original DINO blocks before `inject_layer` as a frozen shared stem. Keep the original blocks from `inject_layer:` as a frozen pose tail. Clone those same blocks into a trainable segmentation tail. Inject query tokens only into the segmentation tail.

Recommendation: yes.

Why:

- preserves the original pose path semantics as much as possible
- allows a new segmentation checkpoint to carry both segmentation and pose capability
- keeps the first implementation bounded and explicit

### Approach 2: Full Frozen Pose Backbone + Extra Tail After It

Run pose through the full original DINO backbone, then append a new segmentation adaptation tail after the final DINO output.

Rejected for first implementation.

Why:

- the cloned blocks no longer operate at their original depth
- initialization is less aligned with pretrained behavior
- more conceptual mismatch for uncertain benefit

### Approach 3: Single Shared Tail With Query Injection

Keep one tail and let both segmentation and pose depend on the query-injected features.

Rejected.

Why:

- highest risk of repeating the pose degradation that motivated this redesign
- does not cleanly separate segmentation adaptation from pose preservation

## Architecture

### High-Level Structure

When `enable_segmentation=True`, the main `GFObjectPose` model contains:

- a shared frozen DINO stem: `blocks[:inject_layer]`
- a frozen pose tail: original `blocks[inject_layer:]`
- a trainable segmentation tail: deep copy of original `blocks[inject_layer:]`
- learnable query embeddings
- the existing `EoMTHead`
- the existing point-cloud encoder and pose score network

### Path Responsibilities

#### Pose Path

Input `roi_rgb` is tokenized and passed through:

- shared stem
- frozen pose tail

The resulting final patch tokens are used for pointwise RGB feature gathering and then flow into the existing point encoder and pose network. Query tokens are never injected into this path.

#### Segmentation Path

The same shared stem output is reused as the base feature sequence for segmentation. Query embeddings are appended only for the segmentation path, then the token sequence is passed through the trainable segmentation tail. The resulting query tokens and patch tokens are consumed by `EoMTHead`.

### Wrapper Contract

`networks/dino_wrapper.py` should expose an interface that makes the separation explicit. The model should no longer return "pose features before query injection." Instead it should return:

- pose patch tokens after the frozen pose tail
- segmentation query tokens after the trainable segmentation tail
- segmentation patch tokens after the trainable segmentation tail

This removes the main source of feature mismatch in the current design.

## Training Design

### Initialization

New segmentation training requires a pretrained `pose_ckpt` as initialization input. After loading model weights:

1. split the original DINO backbone at `inject_layer`
2. preserve the original tail as the frozen pose tail
3. deep-copy that tail into a trainable segmentation tail

### Trainable Parameters

In the first implementation, segmentation training updates only:

- segmentation tail
- query embeddings
- `EoMTHead`

Everything else remains frozen:

- shared DINO stem
- frozen pose tail
- point encoder
- score network
- any unrelated model components

### Losses

The first implementation keeps the current segmentation objective only:

- classification loss
- mask BCE loss
- mask Dice loss

No additional pose-preservation loss is introduced in this phase because the pose path is already frozen and the project prioritizes simplicity.

### Output Artifact

The resulting new `seg_ckpt` becomes the main checkpoint for runtime inference. It contains both:

- the preserved pose path initialized from `pose_ckpt`
- the newly trained segmentation tail and head

## Inference Design

### Runtime Inputs

The nuclear runtime pipeline keeps these checkpoint inputs:

- `seg_ckpt`
- `energy_ckpt`
- `scale_ckpt`

It removes:

- `pose_ckpt`

### Main Agent

`runners/infer_nuclear_full.py` should construct one main `PoseNet` with `enable_segmentation=True` and load the new `seg_ckpt`.

That main agent handles:

- segmentation prediction
- pose feature extraction
- pose sampling

Separate `energy_agent` and `scale_agent` remain unchanged.

### Forward Flow

Segmentation:

1. call the main agent in segmentation mode
2. obtain class logits and mask logits
3. post-process masks

Pose:

1. build pose input from depth and predicted mask
2. call the same main agent for point feature extraction and pose sampling
3. optionally rank with EnergyNet
4. optionally predict size with ScaleNet

### Cache Policy

The first implementation does not require segmentation and pose to share one DINO forward cache. This is intentional.

Reason:

- segmentation and pose often operate on different crops or different derived inputs
- forcing cache sharing would increase coupling and complexity
- it is not necessary to achieve the single-agent, single-checkpoint objective

Single-agent here means one instantiated main model and one main checkpoint, not one mandatory shared forward pass.

## Checkpoint Semantics

### New Segmentation Checkpoint

The new `seg_ckpt` is a versioned main-model checkpoint whose semantics are:

"A single-agent model initialized from `pose_ckpt`, with a frozen pose path and a trainable segmentation adaptation tail."

### Old Checkpoint Handling

Old segmentation checkpoints should not be silently accepted by the new inference path. The new code should explicitly validate that the checkpoint contains the expected single-agent structure. If it does not, inference should fail with a clear message explaining that a new segmentation checkpoint must be retrained from a pose checkpoint.

This avoids "loads with `strict=False` but behaves incorrectly" failure modes.

## Error Handling

- Reject invalid `inject_layer` values during model construction.
- Reject checkpoints missing required single-agent segmentation or pose-tail keys.
- Keep segmentation mode and pose mode independent so pose does not depend on segmentation caches.
- Keep failure messages specific: old checkpoint format, missing tail weights, unsupported runtime args, or invalid layer split.

## Validation Plan

### Functional Validation

- Train a new segmentation model from a valid `pose_ckpt`.
- Confirm only the intended segmentation parameters update.
- Save a new `seg_ckpt`.
- Run nuclear full-pipeline inference with only `seg_ckpt`, `energy_ckpt`, and `scale_ckpt`.

### Smoke Tests

- checkpoint load succeeds for the new format
- checkpoint load fails clearly for the old format
- single-image end-to-end inference succeeds
- segmentation visualization output remains sane
- pose output remains numerically stable and usable

### Regression Bar

The first implementation is judged by practical usability, not strict parity with the old dual-agent path. Minor pose or size regression is acceptable if:

- the runtime pipeline simplifies to one main agent and one main checkpoint
- outputs remain stable and structurally correct
- segmentation quality remains usable
- EnergyNet and ScaleNet integration are not broken

## Implementation Notes

- Keep the redesign focused on the nuclear segmentation-and-pose path.
- Avoid unrelated refactors in the first change.
- Make the new checkpoint format explicit in code and error messages.
- Prefer correctness and boundary clarity over premature optimization.
