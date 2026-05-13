# CatSpec v0.2 Asset Review

This review records which shared workpiece assets can be represented by the
current static CatSpec locus vocabulary without using the `_weld.obj` mesh as
generation input.

| Category | Asset status | CatSpec v0.2 status | Notes |
| --- | --- | --- | --- |
| `square_tube` | `source_mesh` and `_weld.obj` present | covered | `closed_rounded_rect` locus from dense workpiece end profile. |
| `channel_steel` | `source_mesh` and `_weld.obj` present | covered | `open_line_arc_line_arc_line` locus from dense internal planes. |
| `H_beam` | `source_mesh` and `_weld.obj` present | covered | `open_line_arc_line_arc_line` locus from dense internal planes. |
| `bellmouth` | `source_mesh` and `_weld.obj` present | covered | `parallel_open_lines` locus from the dense profile plane and two dense internal offset values. |
| `cover_plate` | `source_mesh` and `_weld.obj` present | schema gap | The reference weld is a closed freeform curve extracted by `CoverPlateStrategy` as a closed line/arc path. CatSpec v0.2 does not yet have a prompt-first closed freeform or spline locus that can be generated robustly from the workpiece mesh alone. Adding it needs a bounded closed-profile primitive such as `closed_freeform_profile` or a higher-level cover-plate contact rule. |
| `pokou` | no OBJ paths in `workpiece_info.json` | unavailable | No source mesh or weld reference asset is available in the current shared model directory. |

The `_weld.obj` files remain validation or auto-GT comparison references only.
They are not used to generate CatSpec loci.
