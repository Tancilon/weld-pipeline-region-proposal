# Square Tube CatSpec v0 Design

## Purpose

Build the first CatSpec proof of concept on `square_tube`.

The goal is not to create a full CAD reconstruction system. The goal is to prove that a prompt-first structured category specification can describe one existing workpiece category well enough to generate a closed rounded-rectangle weld locus and validate it against the existing `square_tube_weld.obj`.

## Scope

In scope:

- Add a prompt-first CatSpec YAML for `square_tube`.
- Represent the category as one `square_tube` primitive part named `tube_body`.
- Encode size priors from `datasets/workpiece_info.json`.
- Encode one weld rule: a closed rounded-rectangle outer perimeter locus.
- Encode weld metadata needed by the future SeamHead: weld type prior, torch constraints, load-bearing flag, and confidence.
- Add a static validation path that compares the spec-derived locus with the existing weld OBJ through the current `SquareTubeStrategy`.

Out of scope:

- Training `SpecEncoder`.
- Changing Stage 1, Stage 2, Stage 3, or SeamHead model code.
- Automatically inferring a general CatSpec from arbitrary OBJ files.
- Treating the exact `_weld.obj` polyline as model input.
- Full CSG or mesh reconstruction from the YAML.

## Existing Inputs

Source assets:

- Category mesh: `datasets/obj_share_models/square_tube/square_tube.obj`
- Weld mesh: `datasets/obj_share_models/square_tube/square_tube_weld.obj`
- Size priors: `datasets/workpiece_info.json`
- Existing weld fitting strategy: `weld/strategies/square_tube.py`

The existing `SquareTubeStrategy` fits each weld component as a closed rounded rectangle with four line segments and four corner arcs. This becomes the validation reference for CatSpec v0.

## CatSpec YAML Shape

The first YAML should live at:

`specs/categories/square_tube.yaml`

It should be prompt-first, with enough geometry to generate the validation locus:

```yaml
schema_version: catspec.v0
category: square_tube
units: meter

provenance:
  source_mesh: datasets/obj_share_models/square_tube/square_tube.obj
  source_weld_mesh: datasets/obj_share_models/square_tube/square_tube_weld.obj
  size_source: datasets/workpiece_info.json

parts:
  - id: tube_body
    primitive: square_tube
    role: primary_structure
    frame: canonical_bbox
    size_priors:
      bbox_xyz:
        - [0.220000, 0.209000, 0.220000]
        - [0.220000, 0.408000, 0.220000]
    symmetry: z2_or_c4
    prompt_tags:
      - hollow_profile
      - rectilinear_tube
      - rounded_corners

welds:
  - id: outer_perimeter
    parts:
      - tube_body
    locus:
      type: closed_rounded_rect
      source: analytic_from_profile
      frame: weld_local_pca
      params:
        profile_source: part_bbox
        corner_radius_source: estimate_from_workpiece_mesh
    weld_meta:
      weld_type_prior: fillet
      torch_constraints: default_single_pass
      is_load_bearing: true
      confidence: medium
```

The `provenance` block may be used by validators and data tooling, but it should not be encoded as model prompt content. The model-facing content is the structural part, weld, geometry, and metadata blocks.

## Static Validation

Add a validation script, for example:

`scripts/validate_catspec_square_tube.py`

The validator should:

1. Load `specs/categories/square_tube.yaml`.
2. Load `square_tube_weld.obj`.
3. Run the existing `SquareTubeStrategy` on the weld mesh to obtain the reference path.
4. Generate a CatSpec-derived closed rounded-rectangle locus from the YAML and the category mesh, not from the weld mesh.
5. Align the generated and reference loci in the same local PCA plane for metric computation.
6. Report:
   - topology match: expected `line, arc, line, arc, line, arc, line, arc`;
   - closed-path gap;
   - centerline RMSE;
   - Hausdorff distance;
   - segment count and segment types.
7. Write a JSON report and an overlay PNG under a generated output directory.

The first implementation may use the reference path's PCA plane only as an evaluation frame so both paths can be compared. It must not use the reference weld centerline, reference segment endpoints, or reference rounded-rectangle dimensions to generate the CatSpec locus. Shape parameters should come from the YAML and simple measurements on `square_tube.obj`; if those measurements are too crude, the report should show that error instead of hiding it. A later v1 can replace these simple measurements with a more general profile extractor.

## Acceptance Criteria

The v0 is accepted when:

- `square_tube.yaml` parses successfully.
- The generated locus is closed.
- The generated locus topology is four lines plus four arcs.
- The report includes RMSE, Hausdorff, closed-path gap, and topology match.
- The overlay PNG shows the CatSpec locus and weld OBJ reference in the same frame.
- The exact `_weld.obj` centerline is used only as validation reference, not as model prompt content.

No strict numeric threshold is set for the first pass. The first run establishes the baseline error and reveals whether the frame and topology conventions are correct.

## Error Handling

The validator should fail clearly when:

- The YAML is missing required fields.
- The weld mesh cannot be loaded.
- `SquareTubeStrategy` returns no valid component.
- The reference topology is not the expected rounded-rectangle structure.
- Generated and reference paths have incompatible point counts after sampling.

Failures should include the category, source file, and failing validation stage.

## Testing

Focused tests should cover:

- YAML parser accepts the `square_tube` CatSpec.
- Required fields are enforced.
- Generated locus is closed.
- The segment type sequence is exactly four lines and four arcs.
- The validator can run on the existing `square_tube_weld.obj` and produce a report.

These tests should avoid broad training or GPU dependencies.

## Follow-Up Path

After this v0 passes:

1. Generalize the schema parser beyond `square_tube`.
2. Add `channel_steel` and `H_beam` using the existing open-path strategy.
3. Add `bellmouth` as a complex body with a simple line weld.
4. Decide whether `cover_plate` needs richer multi-object handling.
5. Use the accepted schema as the input contract for a future `SpecEncoder` prototype.
