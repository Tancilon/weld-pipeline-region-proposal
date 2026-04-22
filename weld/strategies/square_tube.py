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
