# CatSpec-Pose P1 Mini Train/Infer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal reproducible CatSpec-Pose training and inference loop that proves CatSpec prompt embeddings enter a model path and can be compared across spec-correct, spec-shuffle, and no-spec modes.

**Architecture:** Keep this PoC standalone under `catspec/` and do not wire it into EoMT. `SpecEncoder` deterministically converts CatSpec YAML or auto-GT rows into token ids and embeddings; `CatSpecMiniDataset` loads v0.2 auto-GT rows and builds labels; `SpecConditionedLightweightHead` trains three small classification heads for topology, path count, and segment sequence. CLI scripts export auto-GT, train, infer, and evaluate a JSON report.

**Tech Stack:** Python 3.13, PyTorch, pytest, existing CatSpec v0.2 auto-GT/validation code.

---

### Task 1: SpecEncoder

**Files:**
- Create: `catspec/spec_encoder.py`
- Create: `tests/catspec/test_spec_encoder.py`

- [ ] Write failing tests for deterministic tokenization, batch padding/mask, and different embeddings for different specs.
- [ ] Implement stable hash-token encoding from CatSpec dictionaries/YAML paths and auto-GT rows.
- [ ] Implement deterministic frozen embedding table and masked mean pooling.
- [ ] Run `python -m pytest tests/catspec/test_spec_encoder.py -v`.

### Task 2: Dataset Adapter

**Files:**
- Create: `catspec/mini_dataset.py`
- Create: `tests/catspec/test_mini_dataset.py`

- [ ] Write failing tests for loading auto-GT manifest rows with required raw fields.
- [ ] Add target derivation for topology label, path count label, and segment-sequence label.
- [ ] Add modes `spec_correct`, `spec_shuffle`, `no_spec` without adding `_weld.obj` to model inputs.
- [ ] Run `python -m pytest tests/catspec/test_mini_dataset.py -v`.

### Task 3: Lightweight Head

**Files:**
- Create: `catspec/lightweight_head.py`
- Create: `tests/catspec/test_lightweight_head.py`

- [ ] Write failing tests for forward output shapes and finite training loss.
- [ ] Implement a small MLP with topology, path-count, and segment-sequence heads.
- [ ] Run `python -m pytest tests/catspec/test_lightweight_head.py -v`.

### Task 4: Train/Infer/Eval Pipeline

**Files:**
- Create: `catspec/mini_pipeline.py`
- Create: `scripts/train_catspec_mini.py`
- Create: `scripts/infer_catspec_mini.py`
- Create: `scripts/evaluate_catspec_mini.py`
- Create: `tests/catspec/test_mini_pipeline.py`
- Create: `tests/scripts/test_catspec_mini_cli.py`

- [ ] Write failing train smoke, inference smoke, and evaluation report tests.
- [ ] Implement deterministic CPU-friendly training with checkpoint, metrics JSON, and train log.
- [ ] Implement inference JSONL for `spec_correct`, `spec_shuffle`, and `no_spec`.
- [ ] Implement evaluation report with model metrics, reused static validation metrics, and gate pass/fail.
- [ ] Run related pytest and real commands:
  - `python scripts/export_catspec_autogt.py --output-dir results/catspec/autogt`
  - `python scripts/train_catspec_mini.py --manifest results/catspec/autogt/catspec_autogt_manifest.json --output-dir results/catspec/mini_train --epochs 120 --batch-size 4 --seed 7`
  - `python scripts/infer_catspec_mini.py --manifest results/catspec/autogt/catspec_autogt_manifest.json --checkpoint results/catspec/mini_train/catspec_mini_checkpoint.pt --output-dir results/catspec/mini_infer`
  - `python scripts/evaluate_catspec_mini.py --predictions results/catspec/mini_infer/catspec_mini_predictions.jsonl --output-dir results/catspec/mini_eval`

### Task 5: Completion Audit

**Files:**
- Inspect all created files, reports, and command outputs.

- [ ] Verify `_weld.obj` paths appear only in raw/reference fields, not model input tensors.
- [ ] Verify pytest and real smoke commands pass.
- [ ] Verify report compares all three modes and `spec_correct` beats both baselines on at least one model metric.
- [ ] Commit on `aiws5.3` and leave worktree clean.
