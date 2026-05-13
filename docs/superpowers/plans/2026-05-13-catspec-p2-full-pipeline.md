# CatSpec-Pose P2 Full Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the P1 mini loop into a reproducible P2 pipeline with large-scale preprocessing, split-aware training, inference, and evaluation.

**Architecture:** Add P2 modules alongside the P1 implementation instead of replacing it. `p2_preprocess` scans OBJ category folders, validates supported CatSpec samples, writes manifest/index/splits/failure records; `p2_dataset` loads split samples and builds spec-conditioned model inputs; `p2_pipeline` reuses the deterministic `SpecEncoder` and lightweight head for train/infer/eval across `spec_correct`, `spec_shuffle`, and `no_spec`.

**Tech Stack:** Python 3.13, PyTorch, pytest, existing CatSpec v0.2 validation and P1 lightweight head code.

---

### Task 1: P2 Preprocessing

**Files:**
- Create: `catspec/p2_preprocess.py`
- Create: `scripts/preprocess_catspec_p2.py`
- Create: `tests/catspec/test_p2_preprocess.py`

- [ ] Write tests that scan a dataset root, pair `source_obj` with `reference_weld_obj`, write a manifest, sample index, split JSON, stats JSON, and failure JSONL.
- [ ] Implement cached/resumable preprocessing with `--force`, `--max-samples`, `--category`, `--seed`, and default output `results/catspec/p2_preprocess`.
- [ ] Reuse CatSpec validation by writing per-sample resolved spec copies under the preprocess output cache.
- [ ] Record unsupported or invalid samples in failure JSONL.
- [ ] Run `python -m pytest tests/catspec/test_p2_preprocess.py -v`.

### Task 2: P2 Dataset Adapter

**Files:**
- Create: `catspec/p2_dataset.py`
- Create: `tests/catspec/test_p2_dataset.py`

- [ ] Write tests that load P2 manifests and splits, build `spec_correct`, `spec_shuffle`, and `no_spec` samples, and verify `_weld.obj` is excluded from `model_inputs`.
- [ ] Implement `CatSpecP2Dataset`, source geometry summaries, target derivation through P1 target helpers, collate, integrity stats, and optional balanced sampler weights.
- [ ] Run `python -m pytest tests/catspec/test_p2_dataset.py -v`.

### Task 3: P2 Train/Infer/Eval Pipeline

**Files:**
- Create: `catspec/p2_pipeline.py`
- Create: `scripts/train_catspec_p2.py`
- Create: `scripts/infer_catspec_p2.py`
- Create: `scripts/evaluate_catspec_p2.py`
- Create: `tests/catspec/test_p2_pipeline.py`
- Create: `tests/scripts/test_catspec_p2_cli.py`

- [ ] Write smoke tests for train, infer, eval, checkpoint resume, split filtering, and report gate.
- [ ] Implement split-aware train/val loop with CLI options for manifest, split, output-dir, epochs, batch size, lr, seed, device, num workers, and resume checkpoint.
- [ ] Save latest/best checkpoints, metrics JSON, and train log.
- [ ] Implement inference JSONL for `spec_correct`, `spec_shuffle`, and `no_spec`, including category, source mesh, predictions, targets, logits summary, and static validation metrics.
- [ ] Implement evaluation report with aggregate, per-category, failure samples, and gate comparison.
- [ ] Run P2 CLI smoke from preprocessing through evaluation.

### Task 4: Completion Audit

**Files:**
- Inspect P2 modules, scripts, tests, generated reports, and git state.

- [ ] Verify every explicit P2 goal requirement maps to an artifact and command output.
- [ ] Verify `_weld.obj` appears only in raw/reference/evaluation fields, not P2 model inputs.
- [ ] Run related pytest and real asset subset smoke.
- [ ] Commit on `aiws5.3`; keep worktree clean.
