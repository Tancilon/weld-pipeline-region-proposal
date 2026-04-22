"""ChannelSteelStrategy: open ∩/⊃-shaped weld seam (3 lines + 2 arcs per path)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import trimesh

from weld.core import (
    MIN_COMPONENT_VERTICES,
    compute_curvature,
    fit_line_error,
    fit_arc_error,
    pca_project,
)
from weld.strategies.base import Strategy


class ChannelSteelStrategy(Strategy):
    """Fit each connected component as an open path: line - arc - line - arc - line."""

    def process(self, mesh: trimesh.Trimesh) -> list[dict]:
        components = mesh.split(only_watertight=False)
        components = [
            c for c in components if len(c.vertices) >= MIN_COMPONENT_VERTICES
        ]
        if not components:
            raise ValueError(
                f"No valid mesh components found (all < {MIN_COMPONENT_VERTICES} vertices)"
            )
        return [self._fit_open_path(c) for c in components]

    @staticmethod
    def _fit_open_path(component: trimesh.Trimesh) -> dict:
        centerline_3d = _centerline_by_graph_bfs(component, group_size=20)
        centerline_3d = _moving_average(centerline_3d, window=5)

        pts_2d, plane = pca_project(component.vertices)
        cl_diffs = centerline_3d - plane["origin"]
        centerline_2d = np.column_stack([
            cl_diffs @ plane["u"],
            cl_diffs @ plane["v"],
        ])

        kappa = compute_curvature(centerline_2d)
        kappa_smooth = _moving_average_1d(kappa, window=5)
        a0, a1, b0, b1 = _find_two_arc_regions(kappa_smooth)

        # Overlap adjacent pieces by one centerline point so line/arc
        # boundaries share a common endpoint. Without this, sparse
        # sampling between bends leaves a visible gap between segments.
        n = len(centerline_2d)
        pieces = [
            ("line", 0, a0 + 1),
            ("arc", a0, a1),
            ("line", a1 - 1, b0 + 1),
            ("arc", b0, b1),
            ("line", b1 - 1, n),
        ]
        fitted = []
        for kind, start, end in pieces:
            pts = centerline_2d[start:end]
            if len(pts) < 2:
                continue
            if kind == "line":
                p0, p1 = pts[0], pts[-1]
                err = fit_line_error(pts, p0, p1)
                fitted.append({
                    "type": "line",
                    "points_2d": [p0, p1],
                    "indices": (start, end),
                    "fitting_error_mm": round(float(err), 4),
                })
            else:
                if len(pts) < 3:
                    continue
                p0 = pts[0]
                pm = pts[len(pts) // 2]
                p1 = pts[-1]
                err = fit_arc_error(pts, p0, pm, p1)
                fitted.append({
                    "type": "arc",
                    "points_2d": [p0, pm, p1],
                    "indices": (start, end),
                    "fitting_error_mm": round(float(err), 4),
                })

        return {
            "centerline_2d": centerline_2d,
            "plane": plane,
            "fitted": fitted,
            "closed": False,
        }


def _centerline_by_graph_bfs(component: trimesh.Trimesh,
                              group_size: int = 20) -> np.ndarray:
    """Extract a centerline from an open-path tubular mesh using mesh graph BFS.

    Finds the graph-diameter endpoints (geodesically farthest pair), then
    sorts all vertices by distance from one endpoint and groups them into
    fixed-size bins. Each bin's centroid becomes a centerline point.
    """
    verts = component.vertices
    g = nx.Graph()
    for face in component.faces:
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            w = float(np.linalg.norm(verts[a] - verts[b]))
            g.add_edge(a, b, weight=w)

    dists0 = nx.single_source_dijkstra_path_length(g, 0)
    end_a = max(dists0, key=dists0.get)
    dists_a = nx.single_source_dijkstra_path_length(g, end_a)

    sorted_verts = sorted(dists_a.keys(), key=lambda v: dists_a[v])
    n_groups = max(2, len(sorted_verts) // group_size)
    centerline = []
    for i in range(n_groups):
        start = i * group_size
        end = start + group_size if i < n_groups - 1 else len(sorted_verts)
        grp = sorted_verts[start:end]
        centerline.append(verts[grp].mean(axis=0))
    return np.array(centerline)


def _moving_average(pts: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(pts) <= window:
        return pts.copy()
    out = pts.copy()
    half = window // 2
    for i in range(half, len(pts) - half):
        out[i] = pts[i - half:i + half + 1].mean(axis=0)
    return out


def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) <= window:
        return x.copy()
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def _find_two_arc_regions(kappa_smooth: np.ndarray) -> tuple:
    """Identify two high-curvature regions in a smoothed curvature signal.

    Returns (arc1_start, arc1_end, arc2_start, arc2_end) — indices of the
    contiguous high-curvature runs, ordered by position.
    """
    n = len(kappa_smooth)
    if n < 10:
        third = max(1, n // 3)
        return (third, 2 * third, 2 * third, n - 1)
    nonzero = kappa_smooth[kappa_smooth > 1e-6]
    thresh = np.median(nonzero) if len(nonzero) else 1e-6
    high = kappa_smooth >= thresh

    # Merge high-curvature runs separated by small low-curvature gaps.
    # Without this, a single noisy dip inside an arc splits it into two,
    # and the algorithm picks both halves as "the two arcs" while missing
    # the actual second arc elsewhere.
    max_gap = max(3, n // 20)
    merged = high.copy()
    i = 0
    while i < n:
        if not merged[i]:
            j = i
            while j < n and not merged[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                merged[i:j] = True
            i = j
        else:
            i += 1

    runs = []
    i = 0
    while i < n:
        if merged[i]:
            j = i
            while j < n and merged[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1

    if len(runs) < 2:
        mid = n // 2
        return (n // 4, mid, mid, 3 * n // 4)

    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    top2 = sorted(runs[:2], key=lambda r: r[0])
    return (top2[0][0], top2[0][1], top2[1][0], top2[1][1])
