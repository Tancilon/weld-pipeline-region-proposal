"""Extract weld seam centerlines from OBJ meshes and fit as line/arc sequences."""

import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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


def _fit_circle_center(pts_2d: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a circle to 2D points, return (center, radius).

    Uses algebraic circle fit (Kasa method) which is fast and sufficient
    for approximately circular point clouds.
    """
    x, y = pts_2d[:, 0], pts_2d[:, 1]
    # Solve: x^2 + y^2 + Dx + Ey + F = 0 => [x y 1] @ [D E F]^T = -(x^2+y^2)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = result
    cx, cy = -D / 2, -E / 2
    val = cx ** 2 + cy ** 2 - F
    R = np.sqrt(max(val, 0.0))
    if R < 1e-12:
        # Degenerate fit (e.g. collinear points): fall back to centroid
        center = pts_2d.mean(axis=0)
        R = np.max(np.linalg.norm(pts_2d - center, axis=1))
        return center, R
    return np.array([cx, cy]), R


def _resample_by_arclength(pts: np.ndarray, target_points: int) -> np.ndarray:
    """Resample an ordered point sequence to uniform arc-length spacing."""
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]
    if total_length < 1e-12:
        return pts

    target_dists = np.linspace(0, total_length, target_points)
    resampled = np.empty((target_points, 2))
    for i, d in enumerate(target_dists):
        idx = np.searchsorted(cum_length, d, side="right") - 1
        idx = np.clip(idx, 0, len(pts) - 2)
        seg_len = seg_lengths[idx]
        if seg_len < 1e-12:
            resampled[i] = pts[idx]
        else:
            t = (d - cum_length[idx]) / seg_len
            resampled[i] = pts[idx] * (1 - t) + pts[idx + 1] * t
    return resampled


def _extract_centerline_by_angle(pts_2d: np.ndarray,
                                  target_points: int = 120) -> np.ndarray:
    """Extract centerline by sorting vertices by angle and group-averaging.

    Fits a circle to the 2D points to find the best reference center,
    sorts by angle from that center, groups consecutive vertices into
    windows, takes window centroids, removes outliers, and resamples
    to uniform arc-length spacing.

    Works for any tube mesh regardless of ring size.

    Returns:
        (M, 2) ordered centerline points
    """
    n_verts = len(pts_2d)

    # Use fitted circle center for sorting — this gives much cleaner
    # ring separation than the centroid for curved paths
    center, radius = _fit_circle_center(pts_2d)

    angles = np.arctan2(pts_2d[:, 1] - center[1], pts_2d[:, 0] - center[0])
    order = np.argsort(angles)
    sorted_pts = pts_2d[order]

    # Window size: average enough vertices per group to smooth out
    # cross-section variation while producing ~target_points groups
    window = max(2, n_verts // target_points)
    n_groups = n_verts // window
    if n_groups < 3:
        raise ValueError(f"Too few groups ({n_groups}) for centerline extraction")

    centerline = np.array([
        sorted_pts[i * window:(i + 1) * window].mean(axis=0)
        for i in range(n_groups)
    ])

    # Remove outlier points: points whose distance from the fitted circle
    # center deviates significantly from the fitted radius
    if len(centerline) > 5 and np.isfinite(radius) and radius > 1e-12:
        dists_from_center = np.linalg.norm(centerline - center, axis=1)
        keep = np.abs(dists_from_center - radius) < radius * 0.3
        if keep.sum() >= 5:
            centerline = centerline[keep]

    # Also remove step-distance jumps iteratively
    for _ in range(3):
        if len(centerline) < 5:
            break
        step_dists = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
        median_step = np.median(step_dists)
        if median_step <= 0:
            break
        max_neighbor_step = np.zeros(len(centerline))
        max_neighbor_step[0] = step_dists[0]
        max_neighbor_step[-1] = step_dists[-1]
        for i in range(1, len(centerline) - 1):
            max_neighbor_step[i] = max(step_dists[i - 1], step_dists[i])
        keep = max_neighbor_step < 5.0 * median_step
        if keep.all():
            break
        centerline = centerline[keep]

    if len(centerline) < 3:
        raise ValueError("Too few centerline points after outlier removal")

    # Light moving-average smoothing with window 3
    if len(centerline) >= 5:
        smoothed = centerline.copy()
        smoothed[1:-1] = (centerline[:-2] + centerline[1:-1] + centerline[2:]) / 3.0
        centerline = smoothed

    # Resample to uniform arc-length spacing via interpolation.
    # Angle-based grouping produces uneven spacing: dense on arcs, sparse
    # on straights. Uniform spacing ensures curvature analysis can detect
    # line/arc transitions reliably.
    # Check if the bulk spacing is uneven (p75/p25 ratio of step sizes).
    # Ignore endpoint outliers by using percentiles instead of max.
    step_sizes = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    if len(step_sizes) > 10:
        p25 = np.percentile(step_sizes, 25)
        p75 = np.percentile(step_sizes, 75)
        iqr_ratio = p75 / max(p25, 1e-12)
        if iqr_ratio > 3.0:
            resample_n = max(target_points * 2, len(centerline) * 2)
            centerline = _resample_by_arclength(centerline, resample_n)

    return centerline


def compute_curvature(pts: np.ndarray) -> np.ndarray:
    """Compute discrete curvature at each point using circumscribed circle.

    For three consecutive points, curvature = 1/R where R is the circumradius.
    Endpoints get the same curvature as their nearest interior neighbor.
    """
    n = len(pts)
    kappa = np.zeros(n)
    for i in range(1, n - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ac = np.linalg.norm(c - a)
        cross = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        denom = ab * bc * ac
        if denom < 1e-12:
            kappa[i] = 0.0
        else:
            kappa[i] = 2.0 * cross / denom
    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]
    return kappa


def segment_by_curvature(centerline: np.ndarray) -> list[dict]:
    """Segment centerline into line and arc segments based on curvature.

    Returns list of dicts with: type, indices (start, end), points_2d
    """
    kappa = compute_curvature(centerline)

    # Smoothing kernel: small enough to preserve line/arc transitions
    kernel_size = max(3, min(len(kappa) // 10, 7))
    kernel = np.ones(kernel_size) / kernel_size
    kappa_smooth = np.convolve(kappa, kernel, mode="same")

    # Threshold: find natural gap between line (low κ) and arc (high κ).
    # Sort unique curvature values and look for the largest relative jump.
    # If curvature is uniform (all line or all arc), fall back to median * 0.5.
    nonzero_kappa = kappa_smooth[kappa_smooth > 1e-8]
    if len(nonzero_kappa) == 0:
        threshold = 1e-6
    else:
        sorted_k = np.sort(nonzero_kappa)
        if len(sorted_k) >= 4:
            # Look for largest relative gap in curvature values
            ratios = sorted_k[1:] / np.maximum(sorted_k[:-1], 1e-12)
            best_gap_idx = np.argmax(ratios)
            best_ratio = ratios[best_gap_idx]
            # Only use gap threshold if the jump is significant (> 3x)
            if best_ratio > 3.0:
                threshold = (sorted_k[best_gap_idx] + sorted_k[best_gap_idx + 1]) / 2
            else:
                # Uniform curvature: use median * 0.5
                threshold = np.median(nonzero_kappa) * 0.5
        else:
            threshold = np.median(nonzero_kappa) * 0.5

    labels = np.where(kappa_smooth < threshold, 0, 1)  # 0=line, 1=arc

    # Merge contiguous same-label runs
    segments = []
    i = 0
    while i < len(labels):
        label = labels[i]
        j = i
        while j < len(labels) and labels[j] == label:
            j += 1
        segments.append({"label": label, "start": i, "end": j})
        i = j

    # Merge short segments into the neighbor with the same label.
    # If both neighbors differ, merge into the one whose median curvature
    # is closer to this segment's median curvature.
    min_seg_len = max(3, len(centerline) // 40)
    changed = True
    while changed:
        changed = False
        new_segments = []
        for seg in segments:
            length = seg["end"] - seg["start"]
            if length < min_seg_len and new_segments:
                prev = new_segments[-1]
                if prev["label"] == seg["label"]:
                    # Same type: just extend
                    prev["end"] = seg["end"]
                    changed = True
                else:
                    # Different type: merge into previous (will re-evaluate)
                    prev["end"] = seg["end"]
                    # Re-classify merged segment by median curvature
                    merged_kappa = kappa_smooth[prev["start"]:prev["end"]]
                    prev["label"] = 1 if np.median(merged_kappa) >= threshold else 0
                    changed = True
            else:
                new_segments.append(seg)
        segments = new_segments

    # Second pass: merge adjacent segments of the same type
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg["label"] == merged[-1]["label"]:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)

    result = []
    for seg in merged:
        seg_type = "arc" if seg["label"] == 1 else "line"
        result.append({
            "type": seg_type,
            "indices": (seg["start"], seg["end"]),
            "points_2d": centerline[seg["start"]:seg["end"]],
        })
    return result


def fit_line_error(pts: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    """Max distance from points to the line defined by p0-p1."""
    d = p1 - p0
    length = np.linalg.norm(d)
    if length < 1e-12:
        return float(np.max(np.linalg.norm(pts - p0, axis=1)))
    diffs = pts - p0
    cross = np.abs(diffs[:, 0] * d[1] - diffs[:, 1] * d[0])
    return float(np.max(cross / length))


def fit_arc_error(pts: np.ndarray, p0: np.ndarray, pm: np.ndarray, p1: np.ndarray) -> float:
    """Max distance from points to the circular arc defined by 3 points."""
    ax, ay = p0
    bx, by = pm
    cx, cy = p1
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-12:
        return fit_line_error(pts, p0, p1)
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / D
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / D
    center = np.array([ux, uy])
    R = np.linalg.norm(p0 - center)
    dists = np.linalg.norm(pts - center, axis=1)
    return float(np.max(np.abs(dists - R)))


def fit_segment(segment: dict) -> dict:
    """Fit a segment and compute fitting error."""
    pts = segment["points_2d"]
    seg_type = segment["type"]
    if seg_type == "line":
        key_points = [pts[0], pts[-1]]
        error = fit_line_error(pts, pts[0], pts[-1])
    else:
        mid_idx = len(pts) // 2
        key_points = [pts[0], pts[mid_idx], pts[-1]]
        error = fit_arc_error(pts, pts[0], pts[mid_idx], pts[-1])
    return {
        "type": seg_type,
        "points_2d": key_points,
        "indices": segment["indices"],
        "fitting_error_mm": round(error, 4),
    }


def extract_centerline(mesh: trimesh.Trimesh, pts_2d: np.ndarray) -> np.ndarray:
    """Extract ordered centerline from PCA-projected weld seam vertices.

    Fits a circle to find a reference center, sorts vertices by angle,
    groups into windows, and smooths to produce an ordered centerline.

    Args:
        mesh: the weld seam trimesh (reserved for future topology-based methods)
        pts_2d: (N, 2) PCA-projected vertex coordinates

    Returns:
        centerline: (M, 2) ordered 2D centerline points
    """
    return _extract_centerline_by_angle(pts_2d)


def detect_closed(centerline: np.ndarray) -> bool:
    """Check if centerline forms a closed path. Closed if gap < 2% of total length."""
    diffs = np.diff(centerline, axis=0)
    total_length = np.sum(np.linalg.norm(diffs, axis=1))
    gap = np.linalg.norm(centerline[-1] - centerline[0])
    return bool(gap < total_length * 0.02)


def _make_closing_segment(fitted_segments, centerline):
    """Create a closing segment (line) from the last endpoint to the first."""
    last_seg = fitted_segments[-1]
    first_seg = fitted_segments[0]
    p_end = np.array(last_seg["points_2d"][-1])
    p_start = np.array(first_seg["points_2d"][0])
    return {
        "type": "line",
        "points_2d": [p_end, p_start],
        "indices": (len(centerline) - 1, 0),
        "fitting_error_mm": 0.0,
    }


def build_json_output_multi(model_name, paths_data):
    """Assemble final JSON with multiple weld paths.

    Args:
        model_name: model name string
        paths_data: list of dicts, each containing:
            - "fitted": list of fitted segment dicts (with points_2d)
            - "plane": PCA plane dict for this path
            - "closed": bool, whether this path is closed

    Returns:
        JSON-serializable dict with model, coord_system, and weld_paths.
    """
    weld_paths = []
    for path in paths_data:
        segments_json = []
        for seg in path["fitted"]:
            pts_2d = np.array(seg["points_2d"])
            pts_3d = back_project(pts_2d, path["plane"])
            segments_json.append({
                "type": seg["type"],
                "points": [[round(c, 6) for c in pt] for pt in pts_3d.tolist()],
                "fitting_error_mm": seg["fitting_error_mm"],
            })
        weld_paths.append({
            "closed": bool(path["closed"]),
            "segments": segments_json,
        })
    return {
        "model": model_name,
        "coord_system": "raw",
        "weld_paths": weld_paths,
    }


def _interpolate_arc_2d(p0, pm, p1, n=50):
    """Interpolate n points along circular arc through p0, pm, p1."""
    ax, ay = p0
    bx, by = pm
    cx, cy = p1
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-12:
        return np.column_stack([np.linspace(p0[0], p1[0], n), np.linspace(p0[1], p1[1], n)])
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / D
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / D
    center = np.array([ux, uy])
    a0 = np.arctan2(p0[1] - uy, p0[0] - ux)
    am = np.arctan2(pm[1] - uy, pm[0] - ux)
    a1 = np.arctan2(p1[1] - uy, p1[0] - ux)

    def _unwrap(start, mid, end):
        # Pick the direction (CCW or CW) that has mid between start and end.
        mid_ccw = (mid - start) % (2 * np.pi)
        end_ccw = (end - start) % (2 * np.pi)
        if mid_ccw < end_ccw:
            # CCW: positive offsets, mid between start and end
            return start, start + mid_ccw, start + end_ccw
        # CW: negate to traverse in the opposite direction
        return start, start + mid_ccw - 2 * np.pi, start + end_ccw - 2 * np.pi

    a0, am, a1 = _unwrap(a0, am, a1)
    R = np.linalg.norm(p0 - center)
    angles = np.linspace(a0, a1, n)
    return np.column_stack([ux + R * np.cos(angles), uy + R * np.sin(angles)])


# Path color cycle: (line_color, arc_color) per path index
_PATH_COLORS = [
    ("tab:blue", "tab:red"),
    ("tab:purple", "tab:orange"),
    ("tab:green", "tab:pink"),
    ("tab:cyan", "tab:brown"),
    ("tab:olive", "tab:gray"),
]

MIN_COMPONENT_VERTICES = 10


def _global_pca_plane(paths_data):
    """Compute a unified PCA plane across all centerlines' 3D points."""
    all_pts_3d = []
    for path in paths_data:
        pts_3d = back_project(path["centerline_2d"], path["plane"])
        all_pts_3d.append(pts_3d)
    all_pts_3d = np.vstack(all_pts_3d)
    _, global_plane = pca_project(all_pts_3d)
    return global_plane


def _project_to_plane(pts_3d, plane):
    """Project 3D points onto the given PCA plane returning 2D coords."""
    diffs = pts_3d - plane["origin"]
    u = np.dot(diffs, plane["u"])
    v = np.dot(diffs, plane["v"])
    return np.column_stack([u, v])


def _feature_edge_segments(mesh, angle_threshold_deg: float = 30.0,
                           max_segments: int = 800) -> list:
    """Return 3D segments for a sparse wireframe view of the mesh.

    Prefers feature edges — boundaries and creases with dihedral angles
    above the threshold — to match MeshLab's crease-edge display. If
    the mesh is smooth and has no crease edges (e.g., a straight tube),
    falls back to a stride-sampled subset of all edges so the 3D panel
    still shows some geometry.
    """
    verts = mesh.vertices
    faces = mesh.faces
    tri0 = verts[faces[:, 0]]
    tri1 = verts[faces[:, 1]]
    tri2 = verts[faces[:, 2]]
    face_normals = np.cross(tri1 - tri0, tri2 - tri0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    face_normals = face_normals / norms

    cos_thresh = np.cos(np.deg2rad(angle_threshold_deg))
    edge_faces: dict[tuple, list] = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            key = (a, b) if a < b else (b, a)
            edge_faces.setdefault(key, []).append(fi)

    feature = []
    all_edges = []
    for (a, b), face_ids in edge_faces.items():
        all_edges.append((a, b))
        if len(face_ids) == 1:
            feature.append([verts[a], verts[b]])
        elif len(face_ids) == 2:
            n1 = face_normals[face_ids[0]]
            n2 = face_normals[face_ids[1]]
            if abs(float(np.dot(n1, n2))) < cos_thresh:
                feature.append([verts[a], verts[b]])

    # Combine feature edges with a stride-sampled subset of ALL edges so
    # smooth meshes (e.g. cylindrical tubes whose only feature edges are
    # tiny end-cap circles) still show their surface along the length.
    target_sample = max(0, max_segments - len(feature))
    stride = max(1, len(all_edges) // max(target_sample, 1))
    sampled = [[verts[a], verts[b]] for (a, b) in all_edges[::stride]]
    return (feature + sampled)[:max_segments]


def visualize_multi(paths_data, mesh, output_path):
    """Generate multi-path overlay plot: all paths on a single 2D+3D panel.

    Each path is drawn in its own color (line / arc pair from _PATH_COLORS).
    The 2D panel uses a global PCA plane so all paths share coordinates.
    """
    if not paths_data:
        return
    global_plane = _global_pca_plane(paths_data)

    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    # 3D background: render the mesh as a sparse wireframe (feature edges
    # only). Filled trisurf obscures thin ring geometry. We emit only edges
    # with a large dihedral angle — these trace silhouettes of the sweep
    # cross-section, keeping the weld seam shape visible without flooding
    # the plot with interior triangulation.
    verts = mesh.vertices
    segments = _feature_edge_segments(mesh)
    if segments:
        wire = Line3DCollection(segments, colors="lightgray",
                                linewidths=0.4, alpha=0.4)
        ax2.add_collection3d(wire)
    # Equal data limits on all 3 axes so 1mm looks the same in X, Y, and Z.
    # Pad smaller dimensions symmetrically around their midpoint so the
    # mesh stays centered without the 3D panel being visually stretched.
    ax2.set_box_aspect((1.0, 1.0, 1.0))
    x_mid = 0.5 * float(verts[:, 0].max() + verts[:, 0].min())
    y_mid = 0.5 * float(verts[:, 1].max() + verts[:, 1].min())
    z_mid = 0.5 * float(verts[:, 2].max() + verts[:, 2].min())
    half_span = 0.5 * max(
        float(verts[:, 0].max() - verts[:, 0].min()),
        float(verts[:, 1].max() - verts[:, 1].min()),
        float(verts[:, 2].max() - verts[:, 2].min()),
        1e-6,
    )
    ax2.set_xlim(x_mid - half_span, x_mid + half_span)
    ax2.set_ylim(y_mid - half_span, y_mid + half_span)
    ax2.set_zlim(z_mid - half_span, z_mid + half_span)

    for idx, path in enumerate(paths_data):
        line_color, arc_color = _PATH_COLORS[idx % len(_PATH_COLORS)]
        local_plane = path["plane"]

        # 2D centerline (via global plane for unified coords)
        cl_3d = back_project(path["centerline_2d"], local_plane)
        cl_2d_global = _project_to_plane(cl_3d, global_plane)
        ax1.plot(cl_2d_global[:, 0], cl_2d_global[:, 1], '-',
                 color='gray', linewidth=1, alpha=0.5)

        for seg in path["fitted"]:
            pts_local_2d = np.array(seg["points_2d"])
            pts_3d = back_project(pts_local_2d, local_plane)
            pts_global_2d = _project_to_plane(pts_3d, global_plane)
            if seg["type"] == "line":
                ax1.plot(pts_global_2d[:, 0], pts_global_2d[:, 1],
                         '-', color=line_color, linewidth=2.5)
                ax1.plot(pts_global_2d[:, 0], pts_global_2d[:, 1],
                         'o', color=line_color, markersize=5)
                ax2.plot(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
                         '-', color=line_color, linewidth=3.5)
            else:
                arc_local_2d = _interpolate_arc_2d(
                    pts_local_2d[0], pts_local_2d[1], pts_local_2d[2], n=50)
                arc_3d = back_project(arc_local_2d, local_plane)
                arc_global_2d = _project_to_plane(arc_3d, global_plane)
                ax1.plot(arc_global_2d[:, 0], arc_global_2d[:, 1],
                         '-', color=arc_color, linewidth=2.5)
                ax1.plot(pts_global_2d[:, 0], pts_global_2d[:, 1],
                         'o', color=arc_color, markersize=5)
                ax2.plot(arc_3d[:, 0], arc_3d[:, 1], arc_3d[:, 2],
                         '-', color=arc_color, linewidth=3.5)
            # Error annotation at mid-point of segment
            mid_idx = len(pts_global_2d) // 2
            mid = pts_global_2d[mid_idx]
            ax1.annotate(f'{seg["fitting_error_mm"]:.2f}mm',
                         xy=mid, fontsize=7, color='darkred',
                         textcoords="offset points", xytext=(4, 4))

    ax1.set_xlabel('u (mm)')
    ax1.set_ylabel('v (mm)')
    ax1.set_title('2D: all paths (global PCA plane)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    ax2.set_title('3D Mesh + Fitted Paths')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_summary(model_name, paths_data):
    """Print analysis summary for a multi-path weld seam."""
    print(f"Weld seam analysis: {model_name}")
    print(f"  Paths: {len(paths_data)}")
    all_errors = []
    for idx, path in enumerate(paths_data):
        planarity = path["plane"]["planarity"]
        planarity_ok = "OK" if planarity >= 0.95 else "WARNING"
        n_pts = len(path["centerline_2d"])
        n_line = sum(1 for s in path["fitted"] if s["type"] == "line")
        n_arc = sum(1 for s in path["fitted"] if s["type"] == "arc")
        errors = [s["fitting_error_mm"] for s in path["fitted"]]
        max_err = max(errors) if errors else 0.0
        all_errors.extend(errors)
        print(f"  Path {idx}: Planarity {planarity:.1%} ({planarity_ok}) | "
              f"{n_pts} centerline pts | "
              f"{len(path['fitted'])} segments ({n_line} line, {n_arc} arc) | "
              f"closed: {path['closed']} | max err: {max_err:.2f} mm")
    overall_max = max(all_errors) if all_errors else 0.0
    print(f"  Overall max fitting error: {overall_max:.2f} mm")


def _process_component(component_mesh: trimesh.Trimesh,
                       force_close: bool = False) -> dict:
    """Run the single-path pipeline (PCA → centerline → segment → fit) on one mesh component.

    Args:
        component_mesh: a trimesh.Trimesh representing one connected weld component
        force_close: if True and path not naturally closed, append a closing line

    Returns:
        dict with keys: centerline_2d, plane, fitted, closed
    """
    pts_2d, plane = pca_project(component_mesh.vertices)
    centerline = extract_centerline(component_mesh, pts_2d)
    segments = segment_by_curvature(centerline)
    fitted = [fit_segment(seg) for seg in segments]
    is_closed = detect_closed(centerline)
    if force_close and not is_closed:
        fitted = fitted + [_make_closing_segment(fitted, centerline)]
        is_closed = True
    return {
        "centerline_2d": centerline,
        "plane": plane,
        "fitted": fitted,
        "closed": is_closed,
    }


