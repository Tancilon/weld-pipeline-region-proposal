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
