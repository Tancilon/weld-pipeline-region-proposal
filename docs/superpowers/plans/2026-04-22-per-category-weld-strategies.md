# Per-Category Weld Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `scripts/extract_weld_seams.py` into a standalone `weld/` Python package with per-category Strategy classes, so each workpiece type can be fine-tuned independently. CLI migrates from `--force-close` to `--category`.

**Architecture:** One `weld/` package containing shared helpers (`core.py`), pipeline orchestration (`pipeline.py`), CLI entry (`__main__.py`), and a `strategies/` subpackage with one file per category. Each strategy implements `Strategy.process(mesh) -> list[paths_data]` however it chooses; the generic pipeline becomes `GenericStrategy`.

**Tech Stack:** Python 3.13, numpy, trimesh, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-04-22-per-category-weld-strategies-design.md`

---

## Final File Structure

```
weld/
├── __init__.py
├── __main__.py                # CLI entry (argparse + --category)
├── pipeline.py                # run_pipeline orchestrator
├── core.py                    # shared helpers (mesh/PCA/fitting/JSON/viz)
└── strategies/
    ├── __init__.py            # registry + get_strategy(name)
    ├── base.py                # Strategy base + GenericStrategy
    ├── bellmouth.py           # BellmouthStrategy (PCA line fit per component)
    ├── channel_steel.py       # ChannelSteelStrategy (stub inheriting Generic)
    ├── h_beam.py              # HBeamStrategy (stub inheriting Generic)
    ├── square_tube.py         # SquareTubeStrategy (4x 90° arc fit)
    └── cover_plate.py         # CoverPlateStrategy (Generic + force_close=True)

scripts/
└── extract_all_weld_seams.sh  # bash only; python entry removed

tests/weld/
├── __init__.py
├── test_core.py               # migrated from tests/scripts/test_extract_weld_seams.py
└── test_strategies.py         # new — registry + per-category tests
```

---

## Task 1: Move code to `weld/` package

**Goal:** Move `scripts/extract_weld_seams.py` → `weld/core.py` and its tests to `tests/weld/test_core.py`. Provide a working `python -m weld` entry via a thin wrapper. Update batch shell script. No behavior changes.

**Files:**
- Move: `scripts/extract_weld_seams.py` → `weld/core.py`
- Move: `tests/scripts/test_extract_weld_seams.py` → `tests/weld/test_core.py`
- Create: `weld/__init__.py`
- Create: `weld/__main__.py`
- Create: `tests/weld/__init__.py`
- Modify: `scripts/extract_all_weld_seams.sh`

- [ ] **Step 1: Move the script and tests with git mv**

```bash
mkdir -p weld tests/weld
git mv scripts/extract_weld_seams.py weld/core.py
git mv tests/scripts/test_extract_weld_seams.py tests/weld/test_core.py
```

- [ ] **Step 2: Create `weld/__init__.py` (empty)**

```bash
: > weld/__init__.py
```

- [ ] **Step 3: Create `tests/weld/__init__.py` (empty)**

```bash
: > tests/weld/__init__.py
```

- [ ] **Step 4: Create `weld/__main__.py` as a thin CLI wrapper**

Write this exact content to `weld/__main__.py`:

```python
from weld.core import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Update test imports in `tests/weld/test_core.py`**

In `tests/weld/test_core.py`, find every occurrence of `from scripts.extract_weld_seams import` and replace the module path with `weld.core`. Use sed:

```bash
sed -i '' 's/from scripts.extract_weld_seams import/from weld.core import/g' tests/weld/test_core.py
```

- [ ] **Step 6: Update shell script to use python -m weld**

Open `scripts/extract_all_weld_seams.sh` and replace every occurrence of `python scripts/extract_weld_seams.py` with `python -m weld`. Use sed:

```bash
sed -i '' 's|python scripts/extract_weld_seams.py|python -m weld|g' scripts/extract_all_weld_seams.sh
```

- [ ] **Step 7: Verify tests still pass**

Run: `python -m pytest tests/weld/test_core.py -v 2>&1 | tail -10`

Expected: 25 passed (same count as before the move).

- [ ] **Step 8: Verify CLI still works**

Run: `python -m weld --help 2>&1 | head -10`

Expected: Argparse help text showing `--workpiece`, `--weld`, `--output`, `--no-viz`, `--force-close` flags.

- [ ] **Step 9: Commit**

```bash
git add weld/ tests/weld/ scripts/extract_all_weld_seams.sh
git commit -m "$(cat <<'EOF'
refactor: move weld seam extraction to weld/ package

git mv scripts/extract_weld_seams.py weld/core.py
git mv tests/scripts/test_extract_weld_seams.py tests/weld/test_core.py
Add weld/__init__.py, weld/__main__.py (thin CLI wrapper),
tests/weld/__init__.py. Update shell script to python -m weld.

No behavior changes yet — strategy refactor in follow-up commits.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add Strategy base class, GenericStrategy, and registry

**Goal:** Introduce `Strategy` abstract base and `GenericStrategy` that wraps the existing pipeline. Add `get_strategy(name)` registry lookup. Does NOT wire into pipeline yet.

**Files:**
- Create: `weld/strategies/__init__.py`
- Create: `weld/strategies/base.py`
- Create: `tests/weld/test_strategies.py`

- [ ] **Step 1: Write failing tests for registry and GenericStrategy**

Create `tests/weld/test_strategies.py` with:

```python
import numpy as np
import trimesh

from weld.strategies import get_strategy
from weld.strategies.base import Strategy, GenericStrategy


def _make_tube_mesh(n_rings=20, n_per_ring=8, radius=2.0, length=100.0):
    verts = []
    faces = []
    for i in range(n_rings):
        x = length * i / (n_rings - 1)
        for j in range(n_per_ring):
            angle = 2 * np.pi * j / n_per_ring
            verts.append([x, radius * np.cos(angle), radius * np.sin(angle)])
    verts = np.array(verts)
    for i in range(n_rings - 1):
        for j in range(n_per_ring):
            jn = (j + 1) % n_per_ring
            v0 = i * n_per_ring + j
            v1 = i * n_per_ring + jn
            v2 = (i + 1) * n_per_ring + j
            v3 = (i + 1) * n_per_ring + jn
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces))


def test_strategy_base_is_abstract():
    s = Strategy()
    try:
        s.process(None)
    except NotImplementedError:
        return
    raise AssertionError("Strategy.process should raise NotImplementedError")


def test_generic_strategy_processes_mesh():
    mesh = _make_tube_mesh()
    result = GenericStrategy().process(mesh)
    assert isinstance(result, list)
    assert len(result) == 1
    path = result[0]
    for key in ("centerline_2d", "plane", "fitted", "closed"):
        assert key in path


def test_generic_strategy_force_close_attribute_default_false():
    assert GenericStrategy.force_close is False


def test_get_strategy_unknown_returns_generic():
    assert isinstance(get_strategy("not_a_category"), GenericStrategy)


def test_get_strategy_none_returns_generic():
    assert isinstance(get_strategy(None), GenericStrategy)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/weld/test_strategies.py -v`

Expected: FAIL — `weld.strategies` module does not exist (ImportError).

- [ ] **Step 3: Create `weld/strategies/base.py`**

Write this content to `weld/strategies/base.py`:

```python
"""Strategy base class and GenericStrategy (default pipeline)."""

from __future__ import annotations

import trimesh

from weld.core import (
    MIN_COMPONENT_VERTICES,
    _process_component,
)


class Strategy:
    """Base class for per-category weld seam extraction strategies.

    Subclasses implement `process(mesh)` returning a list of paths_data
    dicts with keys: centerline_2d, plane, fitted, closed.
    """

    def process(self, mesh: trimesh.Trimesh) -> list[dict]:
        raise NotImplementedError


class GenericStrategy(Strategy):
    """Default pipeline: split → filter → per-component extract+segment+fit.

    Wraps the behavior that existed before per-category strategies were
    introduced. Subclasses can override `force_close` as a class attribute
    to opt into closing-segment insertion.
    """

    force_close: bool = False

    def process(self, mesh: trimesh.Trimesh) -> list[dict]:
        components = mesh.split(only_watertight=False)
        components = [
            c for c in components if len(c.vertices) >= MIN_COMPONENT_VERTICES
        ]
        if not components:
            raise ValueError(
                f"No valid mesh components found (all < {MIN_COMPONENT_VERTICES} vertices)"
            )
        return [_process_component(c, force_close=self.force_close) for c in components]
```

- [ ] **Step 4: Create `weld/strategies/__init__.py`**

Write this content to `weld/strategies/__init__.py`:

```python
"""Per-category weld seam fitting strategies."""

from __future__ import annotations

from weld.strategies.base import Strategy, GenericStrategy


_REGISTRY: dict[str, type[Strategy]] = {}


def get_strategy(name: str | None) -> Strategy:
    """Return a strategy instance by name.

    Falls back to GenericStrategy when name is None or not in the registry.
    """
    if name is None:
        return GenericStrategy()
    cls = _REGISTRY.get(name, GenericStrategy)
    return cls()


__all__ = ["Strategy", "GenericStrategy", "get_strategy"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/weld/test_strategies.py -v`

Expected: 5 passed.

- [ ] **Step 6: Verify core tests still pass**

Run: `python -m pytest tests/weld/ -v 2>&1 | tail -5`

Expected: 30 passed (25 core + 5 new).

- [ ] **Step 7: Commit**

```bash
git add weld/strategies/ tests/weld/test_strategies.py
git commit -m "$(cat <<'EOF'
feat: add Strategy base class and GenericStrategy registry

Introduces Strategy ABC and GenericStrategy that wraps the existing
multi-component pipeline. get_strategy(name) returns an instance from
an internal registry, falling back to GenericStrategy for unknown
or None names. Pipeline wiring comes next.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Extract `pipeline.py` and `__main__.py`; add `--category`

**Goal:** Move `run_pipeline` out of `weld/core.py` into `weld/pipeline.py`, now calling `get_strategy(category).process(mesh)`. Move CLI `main()` into `weld/__main__.py` with a new `--category` arg. Remove `--force-close` (category strategies own that decision).

**Files:**
- Modify: `weld/core.py` (remove `run_pipeline`, `main`, `if __name__ == "__main__"`, and `argparse` import)
- Create: `weld/pipeline.py`
- Modify: `weld/__main__.py` (replace the thin wrapper)
- Modify: `tests/weld/test_core.py` (update `run_pipeline` import location)
- Modify: `tests/weld/test_strategies.py` (add an integration test)

- [ ] **Step 1: Write failing test for new run_pipeline signature**

Append to `tests/weld/test_strategies.py`:

```python
import json
from pathlib import Path


def _make_two_line_obj(tmp_path):
    verts = []
    faces = []
    n_rings_a = 20
    n_per_ring = 8
    tube_r = 2.0
    for i in range(n_rings_a):
        x = 100 * i / (n_rings_a - 1)
        for j in range(n_per_ring):
            ang = 2 * np.pi * j / n_per_ring
            verts.append([x, tube_r * np.cos(ang), tube_r * np.sin(ang)])
    for i in range(n_rings_a - 1):
        for j in range(n_per_ring):
            jn = (j + 1) % n_per_ring
            v0 = i * n_per_ring + j
            v1 = i * n_per_ring + jn
            v2 = (i + 1) * n_per_ring + j
            v3 = (i + 1) * n_per_ring + jn
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    offset_a = n_rings_a * n_per_ring
    for i in range(n_rings_a):
        x = 100 * i / (n_rings_a - 1)
        for j in range(n_per_ring):
            ang = 2 * np.pi * j / n_per_ring
            verts.append([x, 30.0 + tube_r * np.cos(ang), tube_r * np.sin(ang)])
    for i in range(n_rings_a - 1):
        for j in range(n_per_ring):
            jn = (j + 1) % n_per_ring
            v0 = offset_a + i * n_per_ring + j
            v1 = offset_a + i * n_per_ring + jn
            v2 = offset_a + (i + 1) * n_per_ring + j
            v3 = offset_a + (i + 1) * n_per_ring + jn
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    lines = ["o 焊缝"]
    for v in verts:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
    obj_path = tmp_path / "weld.obj"
    obj_path.write_text("\n".join(lines))
    wp_path = tmp_path / "dual.obj"
    wp_path.write_text("o dummy\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    return str(wp_path), str(obj_path)


def test_run_pipeline_uses_generic_when_no_category(tmp_path):
    from weld.pipeline import run_pipeline
    wp, weld = _make_two_line_obj(tmp_path)
    output = tmp_path / "out.json"
    run_pipeline(wp, weld, str(output), no_viz=True, category=None)
    data = json.loads(Path(output).read_text())
    assert "weld_paths" in data
    assert len(data["weld_paths"]) == 2
```

- [ ] **Step 2: Run the new test to verify it fails**

Run: `python -m pytest tests/weld/test_strategies.py::test_run_pipeline_uses_generic_when_no_category -v`

Expected: FAIL — `weld.pipeline` does not exist (ImportError).

- [ ] **Step 3: Create `weld/pipeline.py`**

Write this content to `weld/pipeline.py`:

```python
"""Orchestration entry point for weld seam extraction."""

from __future__ import annotations

import json
import os

from weld.core import (
    build_json_output_multi,
    extract_model_name,
    load_weld_mesh,
    print_summary,
    visualize_multi,
)
from weld.strategies import get_strategy


def run_pipeline(workpiece_path, weld_path, output_path=None,
                 no_viz=False, category=None):
    model_name = extract_model_name(workpiece_path)
    if output_path is None:
        out_dir = os.path.dirname(workpiece_path) or "."
        output_path = os.path.join(out_dir, f"{model_name}_weld_seams.json")
    viz_path = output_path.replace(".json", "_fit.png")

    mesh = load_weld_mesh(weld_path)
    strategy = get_strategy(category)
    paths_data = strategy.process(mesh)

    result = build_json_output_multi(model_name, paths_data)
    print_summary(model_name, paths_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nJSON saved to: {output_path}")

    if not no_viz:
        visualize_multi(paths_data, mesh, viz_path)
        print(f"Visualization saved to: {viz_path}")
```

- [ ] **Step 4: Replace `weld/__main__.py` with full CLI**

Overwrite `weld/__main__.py` with:

```python
"""Command-line entry: python -m weld ..."""

from __future__ import annotations

import argparse

from weld.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Extract weld seam centerlines from OBJ mesh and fit as line/arc segments."
    )
    parser.add_argument("--workpiece", required=True, help="Path to workpiece OBJ file")
    parser.add_argument("--weld", required=True, help="Path to weld seam OBJ file")
    parser.add_argument("--output", default=None, help="JSON output path (default: auto)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--category", default=None,
                        help="Category name (bellmouth, channel_steel, h_beam, "
                             "square_tube, cover_plate). Unknown or omitted = generic.")
    args = parser.parse_args()
    run_pipeline(args.workpiece, args.weld, args.output, args.no_viz, args.category)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Remove `run_pipeline`, `main`, and CLI boilerplate from `weld/core.py`**

In `weld/core.py`:
1. Search for `def run_pipeline(` and delete that entire function.
2. Search for `def main():` and delete that entire function.
3. Delete the `if __name__ == "__main__":` block at the bottom.
4. Delete the `import argparse` statement at the top (no longer needed in core).

Verify core.py no longer contains any of these by running:

```bash
grep -n "def run_pipeline\|def main\|argparse\|__main__" weld/core.py
```

Expected: no output.

- [ ] **Step 6: Update `run_pipeline` import in `tests/weld/test_core.py`**

Currently the file imports `run_pipeline` from `weld.core`. That import must now point to `weld.pipeline`. Use sed:

```bash
sed -i '' 's/from weld.core import run_pipeline/from weld.pipeline import run_pipeline/g' tests/weld/test_core.py
```

- [ ] **Step 7: Run all tests**

Run: `python -m pytest tests/weld/ -v 2>&1 | tail -10`

Expected: 31 passed (30 from before + 1 new).

- [ ] **Step 8: Verify CLI still works end-to-end**

Run: `python -m weld --help 2>&1 | tail -10`

Expected: Argparse help text showing `--category` flag, no `--force-close`.

- [ ] **Step 9: Commit**

```bash
git add weld/ tests/weld/
git commit -m "$(cat <<'EOF'
refactor: split pipeline.py and __main__.py; add --category CLI flag

Moves run_pipeline to weld/pipeline.py where it now resolves a
Strategy via get_strategy(category). Moves argparse+main to
weld/__main__.py. Replaces --force-close with --category
(unknown or omitted category falls back to GenericStrategy).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: BellmouthStrategy

**Goal:** Implement `BellmouthStrategy` that fits each connected component as a single straight line via PCA.

**Files:**
- Create: `weld/strategies/bellmouth.py`
- Modify: `weld/strategies/__init__.py` (register)
- Modify: `tests/weld/test_strategies.py` (add tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/weld/test_strategies.py`:

```python
def test_bellmouth_strategy_fits_single_line_per_component():
    from weld.strategies.bellmouth import BellmouthStrategy
    # _make_two_line_obj produces 2 disconnected straight tubes
    # use _make_tube_mesh twice, stitch to single trimesh with disjoint verts
    m1 = _make_tube_mesh()
    v2 = m1.vertices.copy()
    v2[:, 1] += 30
    mesh = trimesh.Trimesh(
        vertices=np.vstack([m1.vertices, v2]),
        faces=np.vstack([m1.faces, m1.faces + len(m1.vertices)]),
    )
    paths = BellmouthStrategy().process(mesh)
    assert len(paths) == 2
    for p in paths:
        assert p["closed"] is False
        assert len(p["fitted"]) == 1
        assert p["fitted"][0]["type"] == "line"
        # Straight tube of length 100 should fit with near-zero error
        assert p["fitted"][0]["fitting_error_mm"] < 1.0


def test_get_strategy_returns_bellmouth():
    from weld.strategies.bellmouth import BellmouthStrategy
    assert isinstance(get_strategy("bellmouth"), BellmouthStrategy)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/weld/test_strategies.py::test_bellmouth_strategy_fits_single_line_per_component tests/weld/test_strategies.py::test_get_strategy_returns_bellmouth -v`

Expected: FAIL — `BellmouthStrategy` does not exist.

- [ ] **Step 3: Create `weld/strategies/bellmouth.py`**

Write this content to `weld/strategies/bellmouth.py`:

```python
"""BellmouthStrategy: each connected component is a single straight line."""

from __future__ import annotations

import numpy as np
import trimesh

from weld.core import (
    MIN_COMPONENT_VERTICES,
    fit_line_error,
    pca_project,
)
from weld.strategies.base import Strategy


class BellmouthStrategy(Strategy):
    """Fit each connected component as one straight line via PCA."""

    def process(self, mesh: trimesh.Trimesh) -> list[dict]:
        components = mesh.split(only_watertight=False)
        components = [
            c for c in components if len(c.vertices) >= MIN_COMPONENT_VERTICES
        ]
        if not components:
            raise ValueError(
                f"No valid mesh components found (all < {MIN_COMPONENT_VERTICES} vertices)"
            )
        return [self._fit_line(c) for c in components]

    @staticmethod
    def _fit_line(component: trimesh.Trimesh) -> dict:
        pts_2d, plane = pca_project(component.vertices)
        # PC1 is x; project vertices to PC1 and take extreme points
        x_min = float(pts_2d[:, 0].min())
        x_max = float(pts_2d[:, 0].max())
        p0 = np.array([x_min, 0.0])
        p1 = np.array([x_max, 0.0])
        error = fit_line_error(pts_2d, p0, p1)
        fitted = [{
            "type": "line",
            "points_2d": [p0, p1],
            "indices": (0, len(pts_2d)),
            "fitting_error_mm": round(float(error), 4),
        }]
        centerline = np.array([p0, p1])
        return {
            "centerline_2d": centerline,
            "plane": plane,
            "fitted": fitted,
            "closed": False,
        }
```

- [ ] **Step 4: Register BellmouthStrategy in `weld/strategies/__init__.py`**

Modify `weld/strategies/__init__.py`. Replace its content with:

```python
"""Per-category weld seam fitting strategies."""

from __future__ import annotations

from weld.strategies.base import Strategy, GenericStrategy
from weld.strategies.bellmouth import BellmouthStrategy


_REGISTRY: dict[str, type[Strategy]] = {
    "bellmouth": BellmouthStrategy,
}


def get_strategy(name: str | None) -> Strategy:
    """Return a strategy instance by name.

    Falls back to GenericStrategy when name is None or not in the registry.
    """
    if name is None:
        return GenericStrategy()
    cls = _REGISTRY.get(name, GenericStrategy)
    return cls()


__all__ = ["Strategy", "GenericStrategy", "BellmouthStrategy", "get_strategy"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/weld/test_strategies.py -v 2>&1 | tail -10`

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add weld/strategies/bellmouth.py weld/strategies/__init__.py tests/weld/test_strategies.py
git commit -m "$(cat <<'EOF'
feat: add BellmouthStrategy for straight-line weld seams

Each connected component is PCA-projected, then fit as a single
line from min/max along PC1. Registered as category=bellmouth.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: SquareTubeStrategy

**Goal:** Implement `SquareTubeStrategy` that fits a single closed circle as 4 quarter-arcs.

**Files:**
- Create: `weld/strategies/square_tube.py`
- Modify: `weld/strategies/__init__.py` (register)
- Modify: `tests/weld/test_strategies.py` (add tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/weld/test_strategies.py`:

```python
def _make_circle_tube_mesh(radius=50.0, tube_r=2.0, n_along=40, n_per_ring=8):
    """Make a mesh approximating a torus (circle sweep)."""
    verts = []
    faces = []
    for i in range(n_along):
        theta = 2 * np.pi * i / n_along
        cx = radius * np.cos(theta)
        cy = radius * np.sin(theta)
        for j in range(n_per_ring):
            ang = 2 * np.pi * j / n_per_ring
            normal = np.array([np.cos(theta), np.sin(theta)])
            x = cx + tube_r * np.cos(ang) * normal[0]
            y = cy + tube_r * np.cos(ang) * normal[1]
            z = tube_r * np.sin(ang)
            verts.append([x, y, z])
    for i in range(n_along):
        i_next = (i + 1) % n_along
        for j in range(n_per_ring):
            jn = (j + 1) % n_per_ring
            v0 = i * n_per_ring + j
            v1 = i * n_per_ring + jn
            v2 = i_next * n_per_ring + j
            v3 = i_next * n_per_ring + jn
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces))


def test_square_tube_strategy_fits_four_arcs():
    from weld.strategies.square_tube import SquareTubeStrategy
    mesh = _make_circle_tube_mesh(radius=50.0, tube_r=2.0)
    paths = SquareTubeStrategy().process(mesh)
    assert len(paths) == 1
    path = paths[0]
    assert path["closed"] is True
    assert len(path["fitted"]) == 4
    for seg in path["fitted"]:
        assert seg["type"] == "arc"
        # Circle fit on dense ring should have small error
        assert seg["fitting_error_mm"] < 3.0


def test_get_strategy_returns_square_tube():
    from weld.strategies.square_tube import SquareTubeStrategy
    assert isinstance(get_strategy("square_tube"), SquareTubeStrategy)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/weld/test_strategies.py::test_square_tube_strategy_fits_four_arcs tests/weld/test_strategies.py::test_get_strategy_returns_square_tube -v`

Expected: FAIL — `SquareTubeStrategy` does not exist.

- [ ] **Step 3: Create `weld/strategies/square_tube.py`**

Write this content to `weld/strategies/square_tube.py`:

```python
"""SquareTubeStrategy: single closed circle fit as 4 quarter-arcs."""

from __future__ import annotations

import numpy as np
import trimesh

from weld.core import (
    MIN_COMPONENT_VERTICES,
    _fit_circle_center,
    pca_project,
)
from weld.strategies.base import Strategy


class SquareTubeStrategy(Strategy):
    """Fit a single connected component as a closed circle (4 quarter-arcs)."""

    def process(self, mesh: trimesh.Trimesh) -> list[dict]:
        components = mesh.split(only_watertight=False)
        components = [
            c for c in components if len(c.vertices) >= MIN_COMPONENT_VERTICES
        ]
        if not components:
            raise ValueError(
                f"No valid mesh components found (all < {MIN_COMPONENT_VERTICES} vertices)"
            )
        return [self._fit_closed_circle(c) for c in components]

    @staticmethod
    def _fit_closed_circle(component: trimesh.Trimesh) -> dict:
        pts_2d, plane = pca_project(component.vertices)
        center, radius = _fit_circle_center(pts_2d)
        # 5 endpoints at θ = 0, π/2, π, 3π/2, 2π — 4 arcs between consecutive pairs
        thetas = np.linspace(0.0, 2.0 * np.pi, 5)
        end_pts = np.array([
            [center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)]
            for t in thetas
        ])
        dists = np.linalg.norm(pts_2d - center, axis=1)
        error = float(np.max(np.abs(dists - radius)))
        fitted = []
        for i in range(4):
            p0 = end_pts[i]
            p_end = end_pts[i + 1]
            t_mid = 0.5 * (thetas[i] + thetas[i + 1])
            pm = np.array([
                center[0] + radius * np.cos(t_mid),
                center[1] + radius * np.sin(t_mid),
            ])
            fitted.append({
                "type": "arc",
                "points_2d": [p0, pm, p_end],
                "indices": (0, 0),
                "fitting_error_mm": round(error, 4),
            })
        return {
            "centerline_2d": end_pts,
            "plane": plane,
            "fitted": fitted,
            "closed": True,
        }
```

- [ ] **Step 4: Register SquareTubeStrategy**

Modify `weld/strategies/__init__.py`. Replace its content with:

```python
"""Per-category weld seam fitting strategies."""

from __future__ import annotations

from weld.strategies.base import Strategy, GenericStrategy
from weld.strategies.bellmouth import BellmouthStrategy
from weld.strategies.square_tube import SquareTubeStrategy


_REGISTRY: dict[str, type[Strategy]] = {
    "bellmouth": BellmouthStrategy,
    "square_tube": SquareTubeStrategy,
}


def get_strategy(name: str | None) -> Strategy:
    """Return a strategy instance by name.

    Falls back to GenericStrategy when name is None or not in the registry.
    """
    if name is None:
        return GenericStrategy()
    cls = _REGISTRY.get(name, GenericStrategy)
    return cls()


__all__ = [
    "Strategy", "GenericStrategy",
    "BellmouthStrategy", "SquareTubeStrategy",
    "get_strategy",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/weld/test_strategies.py -v 2>&1 | tail -10`

Expected: 10 passed.

- [ ] **Step 6: Commit**

```bash
git add weld/strategies/square_tube.py weld/strategies/__init__.py tests/weld/test_strategies.py
git commit -m "$(cat <<'EOF'
feat: add SquareTubeStrategy for closed-circle weld seams

Single connected component is fit as a circle via Kasa algorithm,
then emitted as 4 quarter-arcs covering 0→2π. Closed path.
Registered as category=square_tube.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: CoverPlateStrategy, ChannelSteelStrategy, HBeamStrategy

**Goal:** Add the remaining 3 strategies and register all of them. CoverPlateStrategy enables `force_close=True`; the other two are stubs that inherit GenericStrategy verbatim.

**Files:**
- Create: `weld/strategies/cover_plate.py`
- Create: `weld/strategies/channel_steel.py`
- Create: `weld/strategies/h_beam.py`
- Modify: `weld/strategies/__init__.py` (register all)
- Modify: `tests/weld/test_strategies.py` (add tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/weld/test_strategies.py`:

```python
def test_cover_plate_strategy_has_force_close_true():
    from weld.strategies.cover_plate import CoverPlateStrategy
    assert CoverPlateStrategy.force_close is True
    # Confirm it is still a Generic-like (so process() works)
    from weld.strategies.base import GenericStrategy
    assert issubclass(CoverPlateStrategy, GenericStrategy)


def test_channel_steel_strategy_is_generic_subclass():
    from weld.strategies.channel_steel import ChannelSteelStrategy
    from weld.strategies.base import GenericStrategy
    assert issubclass(ChannelSteelStrategy, GenericStrategy)


def test_h_beam_strategy_is_generic_subclass():
    from weld.strategies.h_beam import HBeamStrategy
    from weld.strategies.base import GenericStrategy
    assert issubclass(HBeamStrategy, GenericStrategy)


def test_registry_has_all_five_categories():
    from weld.strategies.bellmouth import BellmouthStrategy
    from weld.strategies.channel_steel import ChannelSteelStrategy
    from weld.strategies.h_beam import HBeamStrategy
    from weld.strategies.square_tube import SquareTubeStrategy
    from weld.strategies.cover_plate import CoverPlateStrategy

    assert isinstance(get_strategy("bellmouth"), BellmouthStrategy)
    assert isinstance(get_strategy("channel_steel"), ChannelSteelStrategy)
    assert isinstance(get_strategy("h_beam"), HBeamStrategy)
    assert isinstance(get_strategy("H_beam"), HBeamStrategy)  # legacy alias
    assert isinstance(get_strategy("square_tube"), SquareTubeStrategy)
    assert isinstance(get_strategy("cover_plate"), CoverPlateStrategy)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/weld/test_strategies.py::test_cover_plate_strategy_has_force_close_true tests/weld/test_strategies.py::test_channel_steel_strategy_is_generic_subclass tests/weld/test_strategies.py::test_h_beam_strategy_is_generic_subclass tests/weld/test_strategies.py::test_registry_has_all_five_categories -v`

Expected: FAIL — strategies don't exist.

- [ ] **Step 3: Create `weld/strategies/cover_plate.py`**

Write this content:

```python
"""CoverPlateStrategy: racetrack shape, works with generic + force_close."""

from __future__ import annotations

from weld.strategies.base import GenericStrategy


class CoverPlateStrategy(GenericStrategy):
    force_close: bool = True
```

- [ ] **Step 4: Create `weld/strategies/channel_steel.py`**

Write this content:

```python
"""ChannelSteelStrategy: placeholder for per-category tuning."""

from __future__ import annotations

from weld.strategies.base import GenericStrategy


class ChannelSteelStrategy(GenericStrategy):
    """Uses generic pipeline; refine per-category as needed."""
    pass
```

- [ ] **Step 5: Create `weld/strategies/h_beam.py`**

Write this content:

```python
"""HBeamStrategy: placeholder for per-category tuning."""

from __future__ import annotations

from weld.strategies.base import GenericStrategy


class HBeamStrategy(GenericStrategy):
    """Uses generic pipeline; refine per-category as needed."""
    pass
```

- [ ] **Step 6: Update `weld/strategies/__init__.py` to register all**

Replace `weld/strategies/__init__.py` content with:

```python
"""Per-category weld seam fitting strategies."""

from __future__ import annotations

from weld.strategies.base import Strategy, GenericStrategy
from weld.strategies.bellmouth import BellmouthStrategy
from weld.strategies.channel_steel import ChannelSteelStrategy
from weld.strategies.cover_plate import CoverPlateStrategy
from weld.strategies.h_beam import HBeamStrategy
from weld.strategies.square_tube import SquareTubeStrategy


_REGISTRY: dict[str, type[Strategy]] = {
    "bellmouth": BellmouthStrategy,
    "channel_steel": ChannelSteelStrategy,
    "cover_plate": CoverPlateStrategy,
    "h_beam": HBeamStrategy,
    "H_beam": HBeamStrategy,  # legacy asset directory used hyphen
    "square_tube": SquareTubeStrategy,
}


def get_strategy(name: str | None) -> Strategy:
    """Return a strategy instance by name.

    Falls back to GenericStrategy when name is None or not in the registry.
    """
    if name is None:
        return GenericStrategy()
    cls = _REGISTRY.get(name, GenericStrategy)
    return cls()


__all__ = [
    "Strategy", "GenericStrategy",
    "BellmouthStrategy", "ChannelSteelStrategy", "CoverPlateStrategy",
    "HBeamStrategy", "SquareTubeStrategy",
    "get_strategy",
]
```

- [ ] **Step 7: Run all weld tests to verify everything passes**

Run: `python -m pytest tests/weld/ -v 2>&1 | tail -10`

Expected: 39 passed (25 core + 10 strategies before this task + 4 new in this task).

- [ ] **Step 8: Commit**

```bash
git add weld/strategies/cover_plate.py weld/strategies/channel_steel.py weld/strategies/h_beam.py weld/strategies/__init__.py tests/weld/test_strategies.py
git commit -m "$(cat <<'EOF'
feat: add cover_plate, channel_steel, h_beam strategies

CoverPlateStrategy inherits GenericStrategy with force_close=True
(restoring the prior cover-plate behavior). ChannelSteelStrategy
and HBeamStrategy are stubs that inherit GenericStrategy verbatim
— placeholders for later category-specific tuning. All five
strategies now registered.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Regenerate outputs and verify

**Goal:** Rerun the batch script with the new per-category strategies and compare against expectations.

- [ ] **Step 1: Update batch script to pass --category**

Overwrite `scripts/extract_all_weld_seams.sh` with:

```bash
#!/usr/bin/env bash
set -e

OUT_DIR="assets/weld_seams_output"
mkdir -p "$OUT_DIR"

CATEGORIES=(bellmouth channel_steel H_beam square_tube cover_plate)

for cat in "${CATEGORIES[@]}"; do
    echo "=== Processing $cat ==="
    python -m weld \
        --workpiece "assets/$cat/$cat.obj" \
        --weld "assets/$cat/${cat}_weld.obj" \
        --output "$OUT_DIR/${cat}_weld_seams.json" \
        --category "$cat"
    echo
done

echo "All results saved to: $OUT_DIR"
```

- [ ] **Step 2: Run batch script**

Run: `bash scripts/extract_all_weld_seams.sh 2>&1 | tail -40`

Expected: 5 categories processed, each summary shows `Paths: ...`, JSON + PNG saved.

- [ ] **Step 3: Verify per-category improvements**

Run this diagnostic one-liner to summarize results:

```bash
python -c "
import json
for cat in ['bellmouth', 'channel_steel', 'H_beam', 'square_tube', 'cover_plate']:
    d = json.load(open(f'assets/weld_seams_output/{cat}_weld_seams.json'))
    for i, p in enumerate(d['weld_paths']):
        types = [s['type'] for s in p['segments']]
        errs = [s['fitting_error_mm'] for s in p['segments']]
        print(f'{cat}/P{i}: {len(types)} segs {types}  closed={p[\"closed\"]}  max_err={max(errs):.2f}mm')
"
```

Expected:
- `bellmouth/P0`, `bellmouth/P1`: 1 segment `['line']` each, closed=False, err < 1.0 mm
- `square_tube/P0`: 4 segments `['arc', 'arc', 'arc', 'arc']`, closed=True
- `cover_plate/P0`: closed=True (multiple segments, similar to prior)
- `channel_steel`, `H_beam`: whatever GenericStrategy produces (unchanged from before)

If any of the three expected categories (bellmouth, square_tube, cover_plate) are not as described, stop and report.

- [ ] **Step 4: Visually spot-check bellmouth PNG**

Open `assets/weld_seams_output/bellmouth_weld_seams_fit.png` and confirm:
- Two clean straight line segments in the 2D panel
- Corresponding lines in the 3D panel

No commit required for regenerated outputs — `assets/weld_seams_output/` is local artifacts only.

- [ ] **Step 5: Commit shell script update**

```bash
git add scripts/extract_all_weld_seams.sh
git commit -m "$(cat <<'EOF'
chore: update batch script to pass --category per workpiece

Each of the 5 workpiece categories now dispatches to its matching
Strategy class. Uniform command structure across all categories;
--force-close is no longer needed at the shell level.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```
