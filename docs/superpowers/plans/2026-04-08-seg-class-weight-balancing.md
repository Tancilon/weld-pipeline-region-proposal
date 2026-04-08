# Segmentation Classification-Balancing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic foreground class weighting for segmentation training, preserve segmentation-first checkpoint selection, and record enough classification-balancing metadata to diagnose future datasets.

**Architecture:** `runners/trainer.py` will derive per-class instance counts from the nuclear train split and generate smoothed inverse-frequency class weights. `networks/posenet_agent.py` will pass those weights into `EoMTCriterion`, and `networks/eomt_head.py` will consume them in classification cross-entropy. Validation and `best` selection remain segmentation-first, with `cls_acc_matched` only used as the final tie-break and recorded for observability.

**Tech Stack:** Python, PyTorch, pytest, COCO-style nuclear dataset annotations, existing `PoseNet` / `EoMTCriterion` / `wandb` training closure.

---

## File Structure

- Modify: `configs/config.py`
  Responsibility: add explicit config knobs for automatic class-weight generation, clipping bounds, and optional enable/disable semantics.

- Modify: `runners/trainer.py`
  Responsibility: read train annotations, count foreground instances, generate smoothed class weights, attach diagnostics to config/summary/wandb, and extend `best` tie-break logic to include `cls_acc_matched` only after segmentation metrics.

- Modify: `networks/posenet_agent.py`
  Responsibility: instantiate `EoMTCriterion` with generated class weights and expose classification-balancing metadata via the existing config snapshot path.

- Modify: `networks/eomt_head.py`
  Responsibility: accept externally provided foreground class weights and combine them with the existing `no-object` weight in classification cross-entropy.

- Modify: `tests/runners/test_trainer_segmentation_setup.py`
  Responsibility: cover automatic class-weight generation, zero-count handling, summary/logging metadata, and segmentation-first best selection with `cls_acc_matched` as the final tie-break. Update the trainer fixture stub for `datasets.datasets_nuclear` to expose `CLASS_NAMES`, because `runners/trainer.py` will import it directly.

- Modify: `tests/networks/test_eomt_head_metrics.py`
  Responsibility: cover criterion behavior when foreground class weights are supplied.

## Task 1: Add Config Surface For Automatic Class Weighting

**Files:**
- Modify: `configs/config.py`
- Test: `tests/runners/test_trainer_segmentation_setup.py`

- [ ] **Step 1: Write the failing config-focused test**

Add a parser-focused test near the trainer setup tests:

```python
def test_get_config_exposes_auto_class_weight_args(monkeypatch):
    from configs.config import get_config

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trainer.py",
            "--class_weight_power",
            "0.75",
            "--class_weight_min",
            "0.5",
            "--class_weight_max",
            "3.0",
            "--no_auto_class_weight",
        ],
    )

    cfg = get_config()

    assert cfg.auto_class_weight is False
    assert cfg.class_weight_power == pytest.approx(0.75)
    assert cfg.class_weight_min == pytest.approx(0.5)
    assert cfg.class_weight_max == pytest.approx(3.0)
```

- [ ] **Step 2: Run the targeted test and verify RED**

Run:

```bash
python -m pytest tests/runners/test_trainer_segmentation_setup.py::test_get_config_exposes_auto_class_weight_args -q
```

Expected: FAIL because the parser does not yet expose the new CLI flags.

- [ ] **Step 3: Add explicit config arguments**

In `configs/config.py`, add these parser arguments in the training section:

```python
    parser.add_argument('--auto_class_weight', dest='auto_class_weight', action='store_true')
    parser.add_argument('--no_auto_class_weight', dest='auto_class_weight', action='store_false')
    parser.set_defaults(auto_class_weight=True)
    parser.add_argument('--class_weight_power', type=float, default=0.5)
    parser.add_argument('--class_weight_min', type=float, default=0.25)
    parser.add_argument('--class_weight_max', type=float, default=4.0)
```

Do not add a manual per-class override path in this task.

- [ ] **Step 4: Run the targeted test and verify GREEN**

Run:

```bash
python -m pytest tests/runners/test_trainer_segmentation_setup.py::test_get_config_exposes_auto_class_weight_args -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/config.py tests/runners/test_trainer_segmentation_setup.py
git commit -m "feat: add auto class weight config surface"
```

## Task 2: Generate Class Counts And Smoothed Foreground Weights In Trainer

**Files:**
- Modify: `runners/trainer.py`
- Modify: `tests/runners/test_trainer_segmentation_setup.py`

- [ ] **Step 1: Write failing trainer tests for count extraction and weight generation**

Add these tests:

```python
def test_collect_nuclear_train_class_counts_reads_foreground_instances(tmp_path, trainer_module):
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir(parents=True)
    payload = {
        "images": [{"id": 1, "file_name": "a.png", "width": 8, "height": 8}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1},
            {"id": 2, "image_id": 1, "category_id": 1},
            {"id": 3, "image_id": 1, "category_id": 3},
        ],
        "categories": [
            {"id": 0, "name": "__background__"},
            {"id": 1, "name": "盖板"},
            {"id": 2, "name": "方管"},
            {"id": 3, "name": "喇叭口"},
            {"id": 4, "name": "H型钢"},
            {"id": 5, "name": "槽钢"},
            {"id": 6, "name": "坡口"},
        ],
    }
    (ann_dir / "train.json").write_text(json.dumps(payload), encoding="utf-8")

    counts = trainer_module.collect_nuclear_train_class_counts(tmp_path.as_posix())

    assert counts == {
        "盖板": 2,
        "方管": 0,
        "喇叭口": 1,
        "H型钢": 0,
        "槽钢": 0,
        "坡口": 0,
    }


def test_generate_smoothed_class_weights_normalizes_and_zeros_missing_classes(trainer_module):
    counts = {
        "盖板": 1045,
        "方管": 100,
        "喇叭口": 200,
        "H型钢": 100,
        "槽钢": 0,
        "坡口": 0,
    }

    weights, zero_count = trainer_module.generate_smoothed_class_weights(
        counts,
        class_weight_power=0.5,
        class_weight_min=0.25,
        class_weight_max=4.0,
    )

    assert len(weights) == 6
    assert zero_count == ["槽钢", "坡口"]
    assert weights[4] == 0.0
    assert weights[5] == 0.0
    assert weights[1] > weights[0]
    assert weights[3] > weights[0]
    positive_weights = [value for value in weights[:4] if value > 0]
    assert pytest.approx(sum(positive_weights) / len(positive_weights), rel=1e-6) == 1.0
```

Also extend the `trainer_module` fixture stub for `datasets.datasets_nuclear` so it exports:

```python
        CLASS_NAMES=["盖板", "方管", "喇叭口", "H型钢", "槽钢", "坡口"],
```

- [ ] **Step 2: Run the targeted tests and verify RED**

Run:

```bash
python -m pytest \
  tests/runners/test_trainer_segmentation_setup.py::test_collect_nuclear_train_class_counts_reads_foreground_instances \
  tests/runners/test_trainer_segmentation_setup.py::test_generate_smoothed_class_weights_normalizes_and_zeros_missing_classes \
  -q
```

Expected: FAIL because the helper functions do not exist yet.

- [ ] **Step 3: Implement the minimal trainer helpers**

Add these helpers near the segmentation training setup in `runners/trainer.py`:

```python
def collect_nuclear_train_class_counts(nuclear_data_path):
    ann_path = os.path.join(nuclear_data_path, "annotations", "train.json")
    with open(ann_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    categories = {
        int(item["id"]): str(item["name"])
        for item in payload.get("categories", [])
    }
    counts = {name: 0 for name in CLASS_NAMES}
    for ann in payload.get("annotations", []):
        cat_name = categories.get(int(ann["category_id"]))
        if cat_name in counts:
            counts[cat_name] += 1
    return counts


def generate_smoothed_class_weights(counts, class_weight_power, class_weight_min, class_weight_max):
    raw_weights = []
    zero_count_classes = []
    for class_name in CLASS_NAMES:
        count = counts[class_name]
        if count <= 0:
            raw_weights.append(0.0)
            zero_count_classes.append(class_name)
        else:
            raw_weights.append(1.0 / (float(count) ** class_weight_power))

    positive = [value for value in raw_weights if value > 0]
    if positive:
        mean_positive = sum(positive) / len(positive)
    else:
        mean_positive = 1.0

    weights = []
    for value in raw_weights:
        if value == 0.0:
            weights.append(0.0)
            continue
        normalized = value / mean_positive
        clipped = min(class_weight_max, max(class_weight_min, normalized))
        weights.append(clipped)
    return weights, zero_count_classes
```

Also import `CLASS_NAMES` in `runners/trainer.py` from `datasets.datasets_nuclear`.

- [ ] **Step 4: Run the targeted tests and verify GREEN**

Run:

```bash
python -m pytest \
  tests/runners/test_trainer_segmentation_setup.py::test_collect_nuclear_train_class_counts_reads_foreground_instances \
  tests/runners/test_trainer_segmentation_setup.py::test_generate_smoothed_class_weights_normalizes_and_zeros_missing_classes \
  -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add runners/trainer.py tests/runners/test_trainer_segmentation_setup.py
git commit -m "feat: derive segmentation class weights from train split"
```

## Task 3: Pass Generated Weights Into The Criterion

**Files:**
- Modify: `networks/eomt_head.py`
- Modify: `networks/posenet_agent.py`
- Modify: `tests/networks/test_eomt_head_metrics.py`

- [ ] **Step 1: Write the failing criterion test**

Add this test:

```python
def test_eomt_criterion_uses_foreground_class_weights_for_cross_entropy():
    criterion = EoMTCriterion(
        num_classes=2,
        class_weights=[0.25, 4.0],
        no_object_weight=0.1,
    )

    class_logits = torch.tensor(
        [[[6.0, -6.0, -6.0], [6.0, -6.0, -6.0]]],
        dtype=torch.float32,
    )
    mask_logits = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    gt_classes = [torch.tensor([1], dtype=torch.long)]
    gt_masks = [torch.zeros((1, 2, 2), dtype=torch.float32)]

    weighted = criterion(class_logits, mask_logits, gt_classes, gt_masks)["cls_loss"]

    baseline = EoMTCriterion(num_classes=2, class_weights=[1.0, 1.0], no_object_weight=0.1)
    unweighted = baseline(class_logits, mask_logits, gt_classes, gt_masks)["cls_loss"]

    assert weighted.item() > unweighted.item()
```

- [ ] **Step 2: Run the targeted test and verify RED**

Run:

```bash
python -m pytest tests/networks/test_eomt_head_metrics.py::test_eomt_criterion_uses_foreground_class_weights_for_cross_entropy -q
```

Expected: FAIL because `EoMTCriterion` does not accept foreground class weights yet.

- [ ] **Step 3: Implement minimal criterion and agent changes**

In `networks/eomt_head.py`, extend the constructor and CE weight construction:

```python
    def __init__(self, num_classes=6, cls_weight=2.0, mask_weight=5.0,
                 dice_weight=5.0, no_object_weight=0.1, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.no_object_weight = no_object_weight
        self.class_weights = class_weights
```

```python
        ce_weight = torch.ones(self.num_classes + 1, device=device)
        if self.class_weights is not None:
            class_weights = torch.as_tensor(
                self.class_weights, dtype=torch.float32, device=device
            )
            ce_weight[: self.num_classes] = class_weights
        ce_weight[self.no_object_class] = self.no_object_weight
```

In `networks/posenet_agent.py`, pass the generated weights:

```python
                class_weights=getattr(self.cfg, 'generated_class_weights', None),
```

- [ ] **Step 4: Run the targeted test and verify GREEN**

Run:

```bash
python -m pytest tests/networks/test_eomt_head_metrics.py::test_eomt_criterion_uses_foreground_class_weights_for_cross_entropy -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add networks/eomt_head.py networks/posenet_agent.py tests/networks/test_eomt_head_metrics.py
git commit -m "feat: apply generated foreground class weights in criterion"
```

## Task 4: Attach Weights To Segmentation Training And Record Diagnostics

**Files:**
- Modify: `runners/trainer.py`
- Modify: `tests/runners/test_trainer_segmentation_setup.py`

- [ ] **Step 1: Write failing tests for config attachment and summary observability**

Add these tests:

```python
def test_build_segmentation_training_agent_attaches_generated_class_weights(
    monkeypatch, trainer_module
):
    monkeypatch.setattr(
        trainer_module,
        "collect_nuclear_train_class_counts",
        lambda path: {
            "盖板": 10,
            "方管": 2,
            "喇叭口": 4,
            "H型钢": 2,
            "槽钢": 0,
            "坡口": 0,
        },
    )
    monkeypatch.setattr(
        trainer_module,
        "generate_smoothed_class_weights",
        lambda counts, class_weight_power, class_weight_min, class_weight_max: (
            [0.5, 1.5, 1.0, 1.5, 0.0, 0.0],
            ["槽钢", "坡口"],
        ),
    )

    created_cfgs = []

    class RecordingAgent:
        def __init__(self, cfg):
            created_cfgs.append(cfg)
            self.cfg = cfg
            self.net = FakeNet()
            self.clock = SimpleNamespace(epoch=0, step=0)

        def load_ckpt(self, **kwargs):
            return None

    monkeypatch.setattr(trainer_module, "PoseNet", RecordingAgent)
    monkeypatch.setattr(trainer_module, "freeze_pose_params", lambda agent: None)

    cfg = SimpleNamespace(
        agent_type="segmentation",
        enable_segmentation=False,
        pretrained_score_model_path="/tmp/pose-init.pth",
        nuclear_data_path="/tmp/nuclear",
        auto_class_weight=True,
        class_weight_power=0.5,
        class_weight_min=0.25,
        class_weight_max=4.0,
    )

    trainer_module.build_segmentation_training_agent(cfg)

    model_cfg = created_cfgs[0]
    assert model_cfg.generated_class_weights == [0.5, 1.5, 1.0, 1.5, 0.0, 0.0]
    assert model_cfg.class_weight_counts["盖板"] == 10
    assert model_cfg.zero_count_classes == ["槽钢", "坡口"]
```

Update the existing `test_build_segmentation_training_agent_uses_score_style_config_and_preserves_training_cfg`
fixture input so it stays valid once automatic weighting is enabled by default. The
simplest version is to set:

```python
        nuclear_data_path="/tmp/nuclear",
        auto_class_weight=False,
```

This keeps the old test focused on pose-init semantics instead of class-weight generation.

Update the summary-writing test to expect:

```python
        "best_cls_acc_matched": 0.5,
        "latest_cls_acc_matched": 0.5,
```

and ensure the fake eval metrics return:

```python
                "cls_acc_matched": torch.tensor(0.5),
```

- [ ] **Step 2: Run the targeted tests and verify RED**

Run:

```bash
python -m pytest \
  tests/runners/test_trainer_segmentation_setup.py::test_build_segmentation_training_agent_attaches_generated_class_weights \
  tests/runners/test_trainer_segmentation_setup.py::test_train_segmentation_writes_summary_with_best_and_latest \
  -q
```

Expected: FAIL because generated weights are not yet attached and summary does not include classification diagnostics.

- [ ] **Step 3: Implement trainer integration**

Update `build_segmentation_training_agent` in `runners/trainer.py`:

```python
    if getattr(cfg, 'auto_class_weight', True):
        class_counts = collect_nuclear_train_class_counts(cfg.nuclear_data_path)
        generated_class_weights, zero_count_classes = generate_smoothed_class_weights(
            class_counts,
            class_weight_power=cfg.class_weight_power,
            class_weight_min=cfg.class_weight_min,
            class_weight_max=cfg.class_weight_max,
        )
        model_cfg.generated_class_weights = generated_class_weights
        model_cfg.class_weight_counts = class_counts
        model_cfg.zero_count_classes = zero_count_classes
        print(f\"[class_weight] counts={class_counts}\")
        print(f\"[class_weight] zero_count_classes={zero_count_classes}\")
        print(f\"[class_weight] generated_class_weights={generated_class_weights}\")
    else:
        model_cfg.generated_class_weights = None
        model_cfg.class_weight_counts = {}
        model_cfg.zero_count_classes = []
```

Update `_build_summary` to include:

```python
                'latest_cls_acc_matched': _to_float(latest_metrics['cls_acc_matched']),
```

and:

```python
                'best_cls_acc_matched': _to_float(current_best['cls_acc_matched']),
```

Also update the `best_metrics` payload to store `cls_acc_matched`.

- [ ] **Step 4: Run the targeted tests and verify GREEN**

Run:

```bash
python -m pytest \
  tests/runners/test_trainer_segmentation_setup.py::test_build_segmentation_training_agent_attaches_generated_class_weights \
  tests/runners/test_trainer_segmentation_setup.py::test_train_segmentation_writes_summary_with_best_and_latest \
  -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add runners/trainer.py tests/runners/test_trainer_segmentation_setup.py
git commit -m "feat: wire automatic class weights into segmentation training"
```

## Task 5: Extend Best-Checkpoint Tie-Break To Use `cls_acc_matched`

**Files:**
- Modify: `runners/trainer.py`
- Modify: `tests/runners/test_trainer_segmentation_setup.py`

- [ ] **Step 1: Write the failing tie-break test**

Add this test:

```python
def test_train_segmentation_uses_cls_acc_as_final_best_tie_break(
    monkeypatch, trainer_module
):
    monkeypatch.setattr(trainer_module, "process_batch_seg", lambda batch, device: batch)

    eval_results = iter(
        [
            {
                "cls_loss": torch.tensor(1.0),
                "mask_loss": torch.tensor(2.0),
                "dice_loss": torch.tensor(3.0),
                "total_loss": torch.tensor(6.0),
                "mask_iou": torch.tensor(0.75),
                "mask_dice": torch.tensor(0.80),
                "cls_acc_matched": torch.tensor(0.30),
                "matched_count": torch.tensor(2.0),
            },
            {
                "cls_loss": torch.tensor(1.0),
                "mask_loss": torch.tensor(2.0),
                "dice_loss": torch.tensor(3.0),
                "total_loss": torch.tensor(6.0),
                "mask_iou": torch.tensor(0.75),
                "mask_dice": torch.tensor(0.80),
                "cls_acc_matched": torch.tensor(0.60),
                "matched_count": torch.tensor(2.0),
            },
        ]
    )

    class FakeClock:
        def __init__(self):
            self.epoch = 1
            self.step = 0

        def tick(self):
            self.step += 1

        def tock(self):
            self.epoch += 1

    class FakeAgent:
        def __init__(self):
            self.clock = FakeClock()
            self.saved = []

        def update_learning_rate(self):
            return None

        def train_func(self, data, gf_mode):
            return {"total_loss": torch.tensor(1.0)}

        def eval_func(self, data, data_mode, gf_mode):
            return next(eval_results)

        def record_losses(self, loss_dict, mode):
            return None

        def save_ckpt(self, name=None):
            self.saved.append(name)

    cfg = SimpleNamespace(
        device="cpu",
        n_epochs=3,
        warmup=0,
        eval_freq=1,
        log_dir="seg-exp",
        nuclear_data_path="/tmp/nuclear",
        pretrained_score_model_path="/tmp/pose-init.pth",
        img_size=224,
        num_queries=50,
        query_injection_layer=6,
    )

    seg_agent = FakeAgent()
    trainer_module.train_segmentation(cfg, [[{"train": 0}]], [{"val": 0}], seg_agent)
    trainer_module.train_segmentation(cfg, [[{"train": 0}]], [{"val": 0}], seg_agent)
```

Replace the last two lines with a single `train_segmentation(...)` call over two
epochs and assert:

```python
    assert seg_agent.saved == ["latest", "best", "latest", "best"]
```

The intent is to prove a second checkpoint with equal segmentation metrics but
better `cls_acc_matched` replaces the first best checkpoint.

- [ ] **Step 2: Run the targeted test and verify RED**

Run:

```bash
python -m pytest tests/runners/test_trainer_segmentation_setup.py::test_train_segmentation_uses_cls_acc_as_final_best_tie_break -q
```

Expected: FAIL because `_is_better()` does not yet consider `cls_acc_matched`.

- [ ] **Step 3: Implement the minimal tie-break change**

Update `_is_better()` in `runners/trainer.py`:

```python
        mask_iou = _to_float(metrics['mask_iou'])
        best_iou = _to_float(current_best['mask_iou'])
        if mask_iou != best_iou:
            return mask_iou > best_iou

        mask_dice = _to_float(metrics['mask_dice'])
        best_dice = _to_float(current_best['mask_dice'])
        if mask_dice != best_dice:
            return mask_dice > best_dice

        return _to_float(metrics['cls_acc_matched']) > _to_float(current_best['cls_acc_matched'])
```

Keep segmentation metrics as the only primary ordering signals.

- [ ] **Step 4: Run the targeted test and verify GREEN**

Run:

```bash
python -m pytest tests/runners/test_trainer_segmentation_setup.py::test_train_segmentation_uses_cls_acc_as_final_best_tie_break -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add runners/trainer.py tests/runners/test_trainer_segmentation_setup.py
git commit -m "feat: use matched cls accuracy as final best tie-break"
```

## Task 6: Run Focused Regression And Manual Diagnostic Checks

**Files:**
- Modify: none
- Test: `tests/networks/test_eomt_head_metrics.py`
- Test: `tests/runners/test_trainer_segmentation_setup.py`
- Test: existing segmentation training outputs under `results/ckpts/<EXP_NAME>/`

- [ ] **Step 1: Run the focused automated regression suite**

Run:

```bash
python -m pytest \
  tests/networks/test_eomt_head_metrics.py \
  tests/runners/test_trainer_segmentation_setup.py \
  tests/utils/test_experiment_logger.py \
  tests/scripts/test_train_seg_single_agent.py \
  tests/scripts/test_check_nuclear_seg_dataset.py \
  -q
```

Expected: all tests PASS

- [ ] **Step 2: Run a short segmentation training smoke on the real dataset**

Run:

```bash
conda run --no-capture-output -n aiws2-genpose env \
  CUDA_VISIBLE_DEVICES=0 \
  MPLCONFIGDIR=/tmp/matplotlib-aiws2-genpose \
  WANDB_MODE=offline \
  EXP_NAME=SegNetClassWeightSmoke \
  N_EPOCHS=2 \
  BATCH_SIZE=4 \
  NUM_WORKERS=0 \
  bash scripts/train_seg_single_agent.sh
```

Expected:

- startup logs include class counts, zero-count classes, and generated weights
- `results/ckpts/SegNetClassWeightSmoke/config.json` contains generated class-weight metadata
- `results/ckpts/SegNetClassWeightSmoke/summary.json` contains `latest_cls_acc_matched` and `best_cls_acc_matched`

- [ ] **Step 3: Verify the produced summary file**

Run:

```bash
cat results/ckpts/SegNetClassWeightSmoke/summary.json
```

Expected: JSON contains at least:

```json
{
  "best_cls_acc_matched": 0.0,
  "latest_cls_acc_matched": 0.0
}
```

The exact values may differ, but both keys must exist.

- [ ] **Step 4: Record the final implementation checkpoint**

```bash
git rev-parse --short HEAD
```

Expected: prints the final commit SHA for the completed implementation.
