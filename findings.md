# CatSpec-Pose P2 Findings

## Current State

- Branch is `aiws5.3`.
- P1 mini loop exists in `catspec/spec_encoder.py`, `catspec/mini_dataset.py`, `catspec/lightweight_head.py`, and `catspec/mini_pipeline.py`.
- Existing CatSpec YAMLs cover `square_tube`, `channel_steel`, `H_beam`, and `bellmouth`.
- Dataset root `/home/dq/mnt/localgit/aiws5.2/datasets/obj_share_models` currently has OBJ pairs for five categories: the four supported categories plus `cover_plate`.
- `cover_plate` has OBJ and `_weld.obj` assets but no CatSpec v0 support in `catspec/schema.py`; P2 preprocessing should record it as a failure.

## Reuse Points

- `catspec.validation._generate_catspec_loci` can generate loci from a workpiece mesh path.
- `catspec.validation.validate_catspec` currently validates from spec provenance paths; P2 can copy/override spec provenance into a generated cache spec per sample to reuse validation exactly.
- P1 `CatSpecMiniDataset` proves `_weld.obj` exclusion from `model_inputs`; P2 should preserve that boundary and add source geometry summaries.
