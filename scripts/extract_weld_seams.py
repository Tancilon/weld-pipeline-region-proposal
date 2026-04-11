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
