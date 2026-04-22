"""SquareTubeStrategy: rounded-rectangle fit (4 straight edges + 4 corner arcs)."""

from __future__ import annotations

import numpy as np
import trimesh

from weld.core import (
    MIN_COMPONENT_VERTICES,
    pca_project,
)
from weld.strategies.base import Strategy


class SquareTubeStrategy(Strategy):
    """Fit each connected component as a rounded rectangle (4 lines + 4 arcs)."""

    def process(self, mesh: trimesh.Trimesh) -> list[dict]:
        components = mesh.split(only_watertight=False)
        components = [
            c for c in components if len(c.vertices) >= MIN_COMPONENT_VERTICES
        ]
        if not components:
            raise ValueError(
                f"No valid mesh components found (all < {MIN_COMPONENT_VERTICES} vertices)"
            )
        return [self._fit_rounded_rect(c) for c in components]

    @staticmethod
    def _fit_rounded_rect(component: trimesh.Trimesh) -> dict:
        pts_2d, plane = pca_project(component.vertices)
        x_min = float(pts_2d[:, 0].min())
        x_max = float(pts_2d[:, 0].max())
        y_min = float(pts_2d[:, 1].min())
        y_max = float(pts_2d[:, 1].max())
        W, H = x_max - x_min, y_max - y_min

        # Estimate corner radius geometrically:
        # bbox corner sits at distance r*(sqrt(2)-1) from the nearest arc point
        # on the outer boundary. Average across the 4 bbox corners.
        r_estimates = []
        for bx, by in [(x_max, y_max), (x_min, y_max),
                        (x_min, y_min), (x_max, y_min)]:
            d = float(np.min(np.linalg.norm(
                pts_2d - np.array([bx, by]), axis=1)))
            r_estimates.append(d / (np.sqrt(2.0) - 1.0))
        r = float(np.median(r_estimates))
        # Guard against degenerate r
        r = min(max(r, 1e-3), 0.5 * min(W, H) - 1e-6)

        # Tangent points where edges meet arcs
        top_l = np.array([x_min + r, y_max])
        top_r = np.array([x_max - r, y_max])
        right_t = np.array([x_max, y_max - r])
        right_b = np.array([x_max, y_min + r])
        bot_r = np.array([x_max - r, y_min])
        bot_l = np.array([x_min + r, y_min])
        left_b = np.array([x_min, y_min + r])
        left_t = np.array([x_min, y_max - r])

        # Arc centers (inward by r from each bbox corner)
        tr_c = np.array([x_max - r, y_max - r])
        br_c = np.array([x_max - r, y_min + r])
        bl_c = np.array([x_min + r, y_min + r])
        tl_c = np.array([x_min + r, y_max - r])

        def arc_mid(center, angle_deg):
            a = np.deg2rad(angle_deg)
            return center + r * np.array([np.cos(a), np.sin(a)])

        error = SquareTubeStrategy._fit_error(
            pts_2d, x_min, x_max, y_min, y_max, r,
            tr_c, br_c, bl_c, tl_c,
        )
        err_round = round(error, 4)

        fitted = [
            # Clockwise from top edge starting at top-left tangent
            {"type": "line", "points_2d": [top_l, top_r],
             "indices": (0, 0), "fitting_error_mm": err_round},
            {"type": "arc",
             "points_2d": [top_r, arc_mid(tr_c, 45.0), right_t],
             "indices": (0, 0), "fitting_error_mm": err_round},
            {"type": "line", "points_2d": [right_t, right_b],
             "indices": (0, 0), "fitting_error_mm": err_round},
            {"type": "arc",
             "points_2d": [right_b, arc_mid(br_c, -45.0), bot_r],
             "indices": (0, 0), "fitting_error_mm": err_round},
            {"type": "line", "points_2d": [bot_r, bot_l],
             "indices": (0, 0), "fitting_error_mm": err_round},
            {"type": "arc",
             "points_2d": [bot_l, arc_mid(bl_c, -135.0), left_b],
             "indices": (0, 0), "fitting_error_mm": err_round},
            {"type": "line", "points_2d": [left_b, left_t],
             "indices": (0, 0), "fitting_error_mm": err_round},
            {"type": "arc",
             "points_2d": [left_t, arc_mid(tl_c, 135.0), top_l],
             "indices": (0, 0), "fitting_error_mm": err_round},
        ]

        centerline = np.array([
            top_l, top_r, right_t, right_b, bot_r, bot_l, left_b, left_t,
        ])
        return {
            "centerline_2d": centerline,
            "plane": plane,
            "fitted": fitted,
            "closed": True,
        }

    @staticmethod
    def _fit_error(pts_2d, x_min, x_max, y_min, y_max, r,
                   tr_c, br_c, bl_c, tl_c) -> float:
        """Max distance from mesh points to the rounded-rectangle boundary."""
        x = pts_2d[:, 0]
        y = pts_2d[:, 1]
        # Vectorized region classification
        in_tr = (x > x_max - r) & (y > y_max - r)
        in_br = (x > x_max - r) & (y < y_min + r)
        in_bl = (x < x_min + r) & (y < y_min + r)
        in_tl = (x < x_min + r) & (y > y_max - r)
        d = np.empty(len(pts_2d))
        d[in_tr] = np.abs(np.linalg.norm(pts_2d[in_tr] - tr_c, axis=1) - r)
        d[in_br] = np.abs(np.linalg.norm(pts_2d[in_br] - br_c, axis=1) - r)
        d[in_bl] = np.abs(np.linalg.norm(pts_2d[in_bl] - bl_c, axis=1) - r)
        d[in_tl] = np.abs(np.linalg.norm(pts_2d[in_tl] - tl_c, axis=1) - r)
        # Edge regions (not in any corner)
        in_any_corner = in_tr | in_br | in_bl | in_tl
        top_edge = (y >= y_max - r) & ~in_any_corner
        bot_edge = (y <= y_min + r) & ~in_any_corner
        right_edge = (x >= x_max - r) & ~in_any_corner & ~top_edge & ~bot_edge
        left_edge = (x <= x_min + r) & ~in_any_corner & ~top_edge & ~bot_edge
        d[top_edge] = np.abs(y[top_edge] - y_max)
        d[bot_edge] = np.abs(y[bot_edge] - y_min)
        d[right_edge] = np.abs(x[right_edge] - x_max)
        d[left_edge] = np.abs(x[left_edge] - x_min)
        return float(np.max(d))
