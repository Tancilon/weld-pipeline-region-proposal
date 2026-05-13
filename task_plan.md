# CatSpec-Pose P2 Plan

Goal: implement a reproducible P2 pipeline for large-scale preprocessing, split-aware training, inference, and evaluation on `aiws5.3`.

## Phases

- [complete] Phase 1: Capture P2 design and tests
- [complete] Phase 2: Implement preprocessing manifest and split generation
- [complete] Phase 3: Implement P2 dataset adapter
- [complete] Phase 4: Implement P2 train/infer/eval pipeline and CLIs
- [complete] Phase 5: Run pytest, real asset smoke, audit, commit, clean worktree

## Decisions

- Reuse P1 `SpecEncoder` and lightweight classification head for P2; P2 focuses on scalable data plumbing and split-aware commands.
- Treat unsupported scanned categories, currently `cover_plate`, as failed preprocessing records with reasons instead of silently dropping them.
- Keep `_weld.obj` in raw/reference/evaluation fields only; model inputs remain token/mask plus optional source geometry summary.
- Use `results/catspec/p2_*` defaults for real command outputs.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
