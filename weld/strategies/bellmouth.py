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
        # Compute per-slice centroids along PC1 for a noise-free error estimate.
        # Raw pts_2d include cross-section spread (≈±radius), so fitting all
        # vertices against the axis line would report the tube radius as error.
        # Instead, bin vertices by position along PC1 and use bin centroids.
        n_bins = max(2, min(50, len(pts_2d) // 4))
        bin_edges = np.linspace(x_min, x_max, n_bins + 1)
        centroids = []
        for k in range(n_bins):
            mask = (pts_2d[:, 0] >= bin_edges[k]) & (pts_2d[:, 0] < bin_edges[k + 1])
            if mask.sum() > 0:
                centroids.append(pts_2d[mask].mean(axis=0))
        if len(centroids) < 2:
            centroids = [p0, p1]
        centroids = np.array(centroids)
        error = fit_line_error(centroids, p0, p1)
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
