"""Extract weld seam centerlines from OBJ meshes and fit as line/arc sequences."""

import warnings
from pathlib import Path

import numpy as np
import trimesh


def _parse_obj_objects(obj_path: str) -> dict[str, dict]:
    """Parse an OBJ file and return per-object vertex and face data.

    Returns a dict mapping object name -> {'vertices': list, 'faces': list}.
    Faces are stored as 0-indexed vertex indices into the per-object vertex list.
    """
    objects: dict[str, dict] = {}
    all_vertices: list[list[float]] = []  # global vertex list (1-indexed in OBJ)
    current_name: str | None = None

    with open(obj_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            token = parts[0]

            if token == "o":
                current_name = " ".join(parts[1:])
                if current_name not in objects:
                    objects[current_name] = {"vertices": [], "faces": []}

            elif token == "v":
                coords = [float(x) for x in parts[1:4]]
                all_vertices.append(coords)
                if current_name is not None:
                    objects[current_name]["vertices"].append(len(all_vertices) - 1)  # global idx

            elif token == "f":
                if current_name is None:
                    continue
                # Face indices are 1-based; strip optional texture/normal (/...)
                raw = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                # Triangulate fan from first vertex
                for i in range(1, len(raw) - 1):
                    objects[current_name]["faces"].append((raw[0], raw[i], raw[i + 1]))

    # Build per-object vertex arrays using global vertex list
    all_v = np.array(all_vertices, dtype=float) if all_vertices else np.empty((0, 3))
    result: dict[str, dict] = {}
    for name, data in objects.items():
        global_indices = data["vertices"]
        if not global_indices:
            continue
        # Create a mapping from global index -> local index
        global_to_local = {g: l for l, g in enumerate(global_indices)}
        verts = all_v[global_indices]
        faces_local = []
        for face in data["faces"]:
            try:
                local_face = [global_to_local[fi] for fi in face]
                faces_local.append(local_face)
            except KeyError:
                pass  # face references vertex from another object; skip
        result[name] = {
            "vertices": verts,
            "faces": np.array(faces_local, dtype=int) if faces_local else np.empty((0, 3), dtype=int),
        }
    return result


def load_weld_mesh(obj_path: str) -> trimesh.Trimesh:
    """Load OBJ and return the sub-mesh with the most vertices."""
    objects = _parse_obj_objects(obj_path)

    if not objects:
        raise ValueError(f"No valid mesh found in {obj_path}")

    if len(objects) == 1:
        name, data = next(iter(objects.items()))
        return trimesh.Trimesh(vertices=data["vertices"], faces=data["faces"], process=False)

    best_name = max(objects, key=lambda n: len(objects[n]["vertices"]))
    data = objects[best_name]
    return trimesh.Trimesh(vertices=data["vertices"], faces=data["faces"], process=False)


def extract_model_name(workpiece_path: str) -> str:
    """Extract model name from workpiece OBJ filename (without extension)."""
    return Path(workpiece_path).stem


def pca_project(vertices: np.ndarray) -> tuple[np.ndarray, dict]:
    """Project 3D vertices onto their best-fit plane via PCA.

    Returns:
        pts_2d: (N, 2) projected coordinates
        plane: dict with keys origin, u, v, n, planarity
    """
    origin = vertices.mean(axis=0)
    centered = vertices - origin
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    u, v, n = Vt[0], Vt[1], Vt[2]
    variance = S ** 2
    planarity = 1.0 - variance[2] / variance.sum()
    if planarity < 0.95:
        warnings.warn(
            f"Weld seam planarity is {planarity:.1%} (< 95%). "
            "Results may be inaccurate for non-planar weld seams."
        )
    pts_2d = centered @ np.column_stack([u, v])
    plane = {"origin": origin, "u": u, "v": v, "n": n, "planarity": planarity}
    return pts_2d, plane


def back_project(pts_2d: np.ndarray, plane: dict) -> np.ndarray:
    """Map 2D plane coordinates back to 3D."""
    return plane["origin"] + pts_2d[:, 0:1] * plane["u"] + pts_2d[:, 1:2] * plane["v"]


def _centerline_from_topology(mesh: trimesh.Trimesh, pts_2d: np.ndarray) -> np.ndarray:
    """Extract centerline by grouping vertices into cross-sectional rings.

    Projects vertices onto the first principal axis (pts_2d[:, 0]), finds gaps
    in the sorted projection values to detect ring boundaries, takes ring
    centroids (projected from 3D without mean-centering), orders them by
    nearest-neighbour chain, and applies a light moving-average smoothing
    (window 3).

    Returns:
        (M, 2) ordered centerline points, or raises ValueError if extraction
        yields fewer than 3 rings.
    """
    proj = pts_2d[:, 0]
    order = np.argsort(proj)
    sorted_proj = proj[order]

    # Detect gaps between consecutive sorted projection values
    steps = np.diff(sorted_proj)
    median_step = np.median(steps[steps > 0]) if np.any(steps > 0) else 0.0
    if median_step == 0.0:
        raise ValueError("Cannot detect ring boundaries: zero median step")

    gap_threshold = 3.0 * median_step
    boundaries = np.where(steps > gap_threshold)[0] + 1  # indices where new ring starts

    # Build ring slices (indices into the original vertex array via `order`)
    ring_slices: list[np.ndarray] = []
    prev = 0
    for b in boundaries:
        ring_slices.append(order[prev:b])
        prev = b
    ring_slices.append(order[prev:])

    if len(ring_slices) < 3:
        raise ValueError(f"Only {len(ring_slices)} ring(s) detected; need >= 3")

    # Recompute PCA axes (u, v) so we can project 3D ring centroids without
    # mean-centering, preserving absolute distances from the 3D origin.
    verts = mesh.vertices
    _, _, Vt = np.linalg.svd(verts - verts.mean(axis=0), full_matrices=False)
    u, v = Vt[0], Vt[1]
    proj_matrix = np.column_stack([u, v])  # (3, 2)

    # Compute 3D centroid of each ring, then project onto (u, v) without centering
    centroids = np.array(
        [verts[idx].mean(axis=0) @ proj_matrix for idx in ring_slices]
    )

    # Order centroids by nearest-neighbour chain starting from one end
    n = len(centroids)
    visited = np.zeros(n, dtype=bool)
    # Start from the centroid with the smallest first-axis value
    start = int(np.argmin(centroids[:, 0]))
    chain = [start]
    visited[start] = True
    for _ in range(n - 1):
        cur = chain[-1]
        dists = np.linalg.norm(centroids - centroids[cur], axis=1)
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        chain.append(nxt)
        visited[nxt] = True
    ordered = centroids[chain]

    # Light moving-average smoothing with window 3
    if len(ordered) >= 3:
        smoothed = ordered.copy()
        smoothed[1:-1] = (ordered[:-2] + ordered[1:-1] + ordered[2:]) / 3.0
        ordered = smoothed

    return ordered


def _centerline_fallback(pts_2d: np.ndarray, n_bins: int = 36) -> np.ndarray:
    """Fallback centerline extraction: bin by angle from centroid, average each bin.

    Returns:
        (M, 2) centerline points (unordered, one per non-empty angular bin).
    """
    centroid = pts_2d.mean(axis=0)
    centered = pts_2d - centroid
    angles = np.arctan2(centered[:, 1], centered[:, 0])  # [-pi, pi]
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.digitize(angles, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    centerline_pts = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            centerline_pts.append(pts_2d[mask].mean(axis=0))

    return np.array(centerline_pts)


def extract_centerline(mesh: trimesh.Trimesh, pts_2d: np.ndarray) -> np.ndarray:
    """Extract ordered centerline points from a structured sweep mesh.

    Uses mesh topology to identify cross-sectional rings, then takes
    ring centroids as centerline points.

    Falls back to boundary-averaging if topology analysis fails.

    Args:
        mesh: the weld seam trimesh
        pts_2d: (N, 2) PCA-projected vertex coordinates

    Returns:
        centerline: (M, 2) ordered 2D centerline points
    """
    try:
        return _centerline_from_topology(mesh, pts_2d)
    except Exception:
        return _centerline_fallback(pts_2d)
