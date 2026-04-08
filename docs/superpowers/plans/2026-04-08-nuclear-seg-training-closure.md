# Nuclear Segmentation Training Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable nuclear segmentation/classification training workflow that uses `wandb`, exports only `best/latest`, and produces runtime-compatible single-agent `seg_ckpt` artifacts.

**Architecture:** Keep `runners/trainer.py` as the training engine, add a dataset-preflight script and a stable shell entrypoint, replace TensorBoard-first tracking with a `wandb`-backed logger abstraction, and make segmentation validation aggregate `mask_iou` / `mask_dice` over the full validation set to drive `best/latest` checkpoint export.

**Tech Stack:** Python, PyTorch, `wandb`, pytest, shell scripts, existing `PoseNet` / `GFObjectPose` / EoMT criterion stack.

---

## File Structure

- Create: `utils/experiment_logger.py`
  Responsibility: provide a `wandb`-backed writer with `add_scalar` / `add_scalars` / `add_image` compatibility plus config/summary helpers for segmentation experiments.

- Modify: `configs/config.py`
  Responsibility: add `wandb`-related CLI flags and keep segmentation checkpoint semantics explicit.

- Modify: `networks/eomt_head.py`
  Responsibility: expose segmentation quality metrics (`mask_iou`, `mask_dice`, optional matched classification accuracy) using the same Hungarian matching used for loss computation.

- Modify: `networks/posenet_agent.py`
  Responsibility: swap TensorBoard-first logging to the new experiment logger abstraction, write `config.json`, support `latest/best` checkpoint exports, and expose segmentation evaluation outputs without per-batch TensorBoard assumptions.

- Modify: `runners/trainer.py`
  Responsibility: aggregate full-validation segmentation metrics, update `best/latest` checkpoint policy, write `summary.json`, and keep segmentation training tied to pose-init score checkpoints only.

- Create: `scripts/check_nuclear_seg_dataset.py`
  Responsibility: validate `data/aiws5.2_nuclear_workpieces` structure and print a compact split/class summary.

- Create: `scripts/train_seg_single_agent.sh`
  Responsibility: provide the stable daily-use segmentation training entrypoint with online/offline `wandb` support.

- Modify: `scripts/train_seg.sh`
  Responsibility: keep compatibility while redirecting users to the new canonical script.

- Create: `tests/networks/test_eomt_head_metrics.py`
  Responsibility: prove segmentation metrics are computed consistently from matched predictions and masks.

- Create: `tests/utils/test_experiment_logger.py`
  Responsibility: prove the logger abstraction writes config/summary locally and maps scalar logging to the expected `wandb` calls without requiring real network access.

- Modify: `tests/runners/test_trainer_segmentation_setup.py`
  Responsibility: prove segmentation training writes through the new segmentation-eval path, rejects full segmentation resume semantics, and selects `best/latest` from validation mask quality.

- Create: `tests/scripts/test_check_nuclear_seg_dataset.py`
  Responsibility: prove the dataset checker validates required files and prints split/class summaries.

## Task 1: Install And Verify `wandb` In `aiws2-genpose`

**Files:**
- Environment only, no repo files

- [ ] **Step 1: Install `wandb` into `aiws2-genpose`**

Run:

```bash
conda run --no-capture-output -n aiws2-genpose python -m pip install wandb
```

Expected: install succeeds inside `aiws2-genpose`

- [ ] **Step 2: Verify `wandb` import**

Run:

```bash
conda run --no-capture-output -n aiws2-genpose python -c "import wandb; print(wandb.__version__)"
```

Expected: prints a version string

- [ ] **Step 3: Verify auth state**

Run:

```bash
conda run --no-capture-output -n aiws2-genpose wandb whoami
conda run --no-capture-output -n aiws2-genpose wandb status
```

Expected: `whoami` resolves a user and `status` shows the configured login state

## Task 2: Add A `wandb`-Backed Experiment Logger

**Files:**
- Create: `utils/experiment_logger.py`
- Modify: `configs/config.py`
- Create: `tests/utils/test_experiment_logger.py`

- [ ] **Step 1: Write the failing logger tests**

Add tests that verify:

```python
def test_build_experiment_logger_initializes_wandb_with_expected_mode(...):
    ...

def test_write_config_snapshot_serializes_namespace_to_config_json(...):
    ...

def test_update_summary_json_merges_best_and_latest_fields(...):
    ...
```

- [ ] **Step 2: Run the logger tests to verify they fail**

Run:

```bash
python -m pytest tests/utils/test_experiment_logger.py -q
```

Expected: FAIL because `utils.experiment_logger` does not exist yet

- [ ] **Step 3: Implement the logger abstraction**

Create `utils/experiment_logger.py` with:

```python
class NoOpLogger:
    def add_scalar(self, *args, **kwargs): ...
    def add_scalars(self, *args, **kwargs): ...
    def add_image(self, *args, **kwargs): ...
    def finish(self): ...

class WandbLogger:
    ...

def build_experiment_logger(cfg, ckpt_dir):
    ...

def write_config_snapshot(cfg, ckpt_dir):
    ...

def update_summary_json(ckpt_dir, summary_update):
    ...
```

- [ ] **Step 4: Add `wandb` config flags**

Extend `configs/config.py` with:

```python
parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
parser.add_argument('--wandb_project', type=str, default='nuclear-seg-single-agent')
parser.add_argument('--wandb_entity', type=str, default='')
parser.add_argument('--wandb_run_name', type=str, default='')
```

- [ ] **Step 5: Run logger tests again**

Run:

```bash
python -m pytest tests/utils/test_experiment_logger.py -q
```

Expected: PASS

## Task 3: Add Segmentation Validation Metrics And Best/Latest Export

**Files:**
- Create: `tests/networks/test_eomt_head_metrics.py`
- Modify: `networks/eomt_head.py`
- Modify: `networks/posenet_agent.py`
- Modify: `runners/trainer.py`
- Modify: `tests/runners/test_trainer_segmentation_setup.py`

- [ ] **Step 1: Write failing metric and trainer tests**

Add tests covering:

```python
def test_eomt_criterion_reports_mask_iou_and_mask_dice_for_matched_masks():
    ...

def test_train_segmentation_updates_latest_and_best_from_val_mask_iou(...):
    ...

def test_train_segmentation_writes_summary_with_best_and_latest(...):
    ...
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
python -m pytest tests/networks/test_eomt_head_metrics.py tests/runners/test_trainer_segmentation_setup.py -q
```

Expected: FAIL because the criterion and trainer do not yet expose those metrics/export rules

- [ ] **Step 3: Extend `EoMTCriterion` with segmentation metrics**

Add a method shaped like:

```python
def evaluate_batch(self, class_logits, mask_logits, gt_classes_list, gt_masks_list):
    return {
        'mask_iou': ...,
        'mask_dice': ...,
        'cls_acc_matched': ...,
    }
```

- [ ] **Step 4: Integrate the experiment logger into `PoseNet`**

Replace the TensorBoard-first setup in `networks/posenet_agent.py` so training uses:

```python
self.writer = build_experiment_logger(self.cfg, self.model_dir)
write_config_snapshot(self.cfg, self.model_dir)
```

and keep the existing `record_losses()` / `record_lr()` call sites working through the compatible writer API.

- [ ] **Step 5: Aggregate full-validation segmentation metrics and export `best/latest`**

Update `runners/trainer.py` so segmentation evaluation:

- iterates the full `val_loader`
- aggregates `cls_loss`, `mask_loss`, `dice_loss`, `total_loss`
- aggregates `mask_iou` and `mask_dice`
- saves `latest.pth` on every eval
- saves `best.pth` when `mask_iou` improves
- writes `summary.json`

- [ ] **Step 6: Run the targeted tests again**

Run:

```bash
python -m pytest tests/networks/test_eomt_head_metrics.py tests/runners/test_trainer_segmentation_setup.py -q
```

Expected: PASS

## Task 4: Add Dataset Preflight Script And Canonical Training Script

**Files:**
- Create: `scripts/check_nuclear_seg_dataset.py`
- Create: `scripts/train_seg_single_agent.sh`
- Modify: `scripts/train_seg.sh`
- Create: `tests/scripts/test_check_nuclear_seg_dataset.py`

- [ ] **Step 1: Write the failing dataset-check tests**

Add tests shaped like:

```python
def test_checker_rejects_missing_train_or_val_json(...):
    ...

def test_checker_prints_split_and_class_summary_for_valid_dataset(...):
    ...
```

- [ ] **Step 2: Run the checker tests to verify they fail**

Run:

```bash
python -m pytest tests/scripts/test_check_nuclear_seg_dataset.py -q
```

Expected: FAIL because the checker script does not exist yet

- [ ] **Step 3: Implement the dataset checker**

Create `scripts/check_nuclear_seg_dataset.py` that:

- validates required paths
- loads `train.json` and `val.json`
- checks category names against the six supported classes
- checks image references exist
- prints split counts and per-class instance counts

- [ ] **Step 4: Add the stable shell training entrypoint**

Create `scripts/train_seg_single_agent.sh` with explicit variables for:

```bash
EXP_NAME=...
NUCLEAR_DATA_PATH=...
POSE_INIT_CKPT=...
WANDB_MODE=online
WANDB_PROJECT=...
WANDB_ENTITY=
```

and run `python runners/trainer.py ...` with the canonical segmentation CLI.

- [ ] **Step 5: Deprecate or redirect the old script**

Modify `scripts/train_seg.sh` so it redirects to `train_seg_single_agent.sh` or clearly warns that it is the legacy wrapper.

- [ ] **Step 6: Run checker tests again**

Run:

```bash
python -m pytest tests/scripts/test_check_nuclear_seg_dataset.py -q
```

Expected: PASS

## Task 5: Final Verification

**Files:**
- Verify changed files only

- [ ] **Step 1: Run focused repo tests**

Run:

```bash
python -m pytest \
  tests/utils/test_experiment_logger.py \
  tests/networks/test_eomt_head_metrics.py \
  tests/runners/test_trainer_segmentation_setup.py \
  tests/scripts/test_check_nuclear_seg_dataset.py \
  tests/runners/test_infer_nuclear_full_lib.py \
  -q
```

Expected: PASS

- [ ] **Step 2: Verify `wandb` CLI in `aiws2-genpose`**

Run:

```bash
conda run --no-capture-output -n aiws2-genpose wandb whoami
conda run --no-capture-output -n aiws2-genpose wandb status
```

Expected: authenticated output

- [ ] **Step 3: Run dataset preflight on the real nuclear dataset**

Run:

```bash
conda run --no-capture-output -n aiws2-genpose python scripts/check_nuclear_seg_dataset.py \
  --nuclear_data_path ./data/aiws5.2_nuclear_workpieces
```

Expected: prints split/class summary without errors

- [ ] **Step 4: Run a 1-epoch segmentation smoke training**

Run:

```bash
env MPLCONFIGDIR=/tmp/matplotlib-aiws2 \
conda run --no-capture-output -n aiws2-genpose bash scripts/train_seg_single_agent.sh
```

Expected: training starts, logs to `wandb`, exports `latest.pth`, and writes `config.json` / `summary.json`

- [ ] **Step 5: Run single-agent inference smoke with the produced `best` or `latest` checkpoint**

Run:

```bash
env MPLCONFIGDIR=/tmp/matplotlib-aiws2 \
conda run --no-capture-output -n aiws2-genpose python runners/infer_nuclear_full.py \
  --nuclear_data_path ./data/aiws5.2_nuclear_workpieces \
  --seg_ckpt ./results/ckpts/<EXP_NAME>/best.pth \
  --energy_ckpt /tmp/energynet_cpu.pth \
  --scale_ckpt /tmp/scalenet_cpu.pth \
  --split val \
  --num_vis 1 \
  --output_dir ./results/full_pipeline_<EXP_NAME> \
  --device cpu
```

Expected: single-agent runtime accepts the produced segmentation checkpoint and writes an output image
