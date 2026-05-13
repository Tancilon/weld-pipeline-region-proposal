"""Analytic weld loci derived from CatSpec workpiece meshes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class SquareTubeProfile:
    """Quantile-clipped rounded square tube profile on one mesh end plane."""

    plane_axis: str
    plane_value: float
    profile_axes: tuple[str, str]
    min_uv: tuple[float, float]
    max_uv: tuple[float, float]
    corner_radius: float


def _axis_index(axis: str) -> int:
    try:
        return AXIS_TO_INDEX[axis]
    except KeyError as exc:
        raise ValueError(f"unsupported axis: {axis!r}") from exc


def _as_vertices(mesh: Any) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("mesh must provide a non-empty vertices array with shape (N, 3)")
    return vertices


def _dense_plane_value(values: np.ndarray, plane_side: str) -> float:
    rounded = np.round(values.astype(float), decimals=6)
    unique, counts = np.unique(rounded, return_counts=True)
    if len(unique) == 0:
        raise ValueError("cannot estimate a dense plane from empty coordinates")

    dense_cutoff = max(2, int(np.ceil(float(counts.max()) * 0.5)))
    dense_values = unique[counts >= dense_cutoff]
    if len(dense_values) == 0:
        dense_values = unique[[int(np.argmax(counts))]]

    if plane_side == "min_dense":
        return float(dense_values.min())
    if plane_side == "max_dense":
        return float(dense_values.max())
    raise ValueError("plane_side must be 'min_dense' or 'max_dense'")


def _estimate_corner_radius(min_uv: np.ndarray, max_uv: np.ndarray) -> float:
    width, height = max_uv - min_uv
    if width <= 0.0 or height <= 0.0:
        raise ValueError("profile extents must have positive width and height")
    return float(0.1 * min(width, height))


def estimate_square_tube_profile(
    mesh: Any,
    plane_axis: str,
    plane_side: Literal["min_dense", "max_dense"],
    profile_axes: tuple[str, str],
    profile_quantile: float,
) -> SquareTubeProfile:
    """Estimate a square tube end profile from the workpiece mesh vertices."""

    vertices = _as_vertices(mesh)
    plane_idx = _axis_index(plane_axis)
    profile_indices = tuple(_axis_index(axis) for axis in profile_axes)
    if len(set((plane_idx, *profile_indices))) != 3:
        raise ValueError("plane_axis and profile_axes must identify distinct axes")
    if not 0.0 <= profile_quantile < 50.0:
        raise ValueError("profile_quantile must be in [0, 50)")

    plane_value = _dense_plane_value(vertices[:, plane_idx], plane_side)
    plane_mask = np.isclose(vertices[:, plane_idx], plane_value, atol=1e-6, rtol=0.0)
    plane_vertices = vertices[plane_mask]
    if len(plane_vertices) == 0:
        raise ValueError("dense plane selection produced no vertices")

    uv = plane_vertices[:, profile_indices]
    min_uv = np.percentile(uv, profile_quantile, axis=0)
    max_uv = np.percentile(uv, 100.0 - profile_quantile, axis=0)
    radius = _estimate_corner_radius(min_uv, max_uv)

    return SquareTubeProfile(
        plane_axis=plane_axis,
        plane_value=plane_value,
        profile_axes=(profile_axes[0], profile_axes[1]),
        min_uv=(float(min_uv[0]), float(min_uv[1])),
        max_uv=(float(max_uv[0]), float(max_uv[1])),
        corner_radius=radius,
    )


def _line(start: tuple[float, float], end: tuple[float, float]) -> dict[str, Any]:
    return {"type": "line", "start": start, "end": end}


def _arc(
    center: tuple[float, float],
    radius: float,
    start_angle: float,
    end_angle: float,
) -> dict[str, Any]:
    start = (
        float(center[0] + radius * np.cos(start_angle)),
        float(center[1] + radius * np.sin(start_angle)),
    )
    end = (
        float(center[0] + radius * np.cos(end_angle)),
        float(center[1] + radius * np.sin(end_angle)),
    )
    return {
        "type": "arc",
        "center": center,
        "radius": float(radius),
        "start_angle": float(start_angle),
        "end_angle": float(end_angle),
        "start": start,
        "end": end,
    }


def build_closed_rounded_rect_locus(profile: SquareTubeProfile) -> dict[str, Any]:
    """Build line/arc rounded-rectangle weld locus segments in profile coordinates."""

    min_u, min_v = profile.min_uv
    max_u, max_v = profile.max_uv
    radius = min(profile.corner_radius, 0.5 * (max_u - min_u), 0.5 * (max_v - min_v))
    if radius <= 0.0:
        raise ValueError("corner_radius must be positive")

    segments = [
        _line((max_u, max_v - radius), (max_u, min_v + radius)),
        _arc((max_u - radius, min_v + radius), radius, 0.0, -np.pi / 2),
        _line((max_u - radius, min_v), (min_u + radius, min_v)),
        _arc((min_u + radius, min_v + radius), radius, -np.pi / 2, -np.pi),
        _line((min_u, min_v + radius), (min_u, max_v - radius)),
        _arc((min_u + radius, max_v - radius), radius, np.pi, np.pi / 2),
        _line((min_u + radius, max_v), (max_u - radius, max_v)),
        _arc((max_u - radius, max_v - radius), radius, np.pi / 2, 0.0),
    ]
    return {"closed": True, "profile": asdict(profile), "segments": segments}


def sample_segments_2d(segments: list[dict[str, Any]], points_per_segment: int) -> np.ndarray:
    """Sample 2D line and circular arc segments."""

    if points_per_segment < 2:
        raise ValueError("points_per_segment must be at least 2")

    samples: list[np.ndarray] = []
    for idx, segment in enumerate(segments):
        t = np.linspace(0.0, 1.0, points_per_segment)
        if idx > 0:
            t = t[1:]
        if segment["type"] == "line":
            start = np.asarray(segment["start"], dtype=float)
            end = np.asarray(segment["end"], dtype=float)
            pts = start[None, :] + (end - start)[None, :] * t[:, None]
        elif segment["type"] == "arc":
            center = np.asarray(segment["center"], dtype=float)
            angles = segment["start_angle"] + (segment["end_angle"] - segment["start_angle"]) * t
            pts = np.column_stack(
                [
                    center[0] + float(segment["radius"]) * np.cos(angles),
                    center[1] + float(segment["radius"]) * np.sin(angles),
                ]
            )
        else:
            raise ValueError(f"unsupported segment type: {segment['type']!r}")
        samples.append(pts)

    points = np.vstack(samples)
    if len(points) and not np.allclose(points[0], points[-1], atol=1e-12, rtol=0.0):
        points = np.vstack([points, points[0]])
    return points


def sample_locus_3d(locus: dict[str, Any], points_per_segment: int) -> np.ndarray:
    """Sample a 2D locus into 3D points using the profile's original axes."""

    profile = locus["profile"]
    points_2d = sample_segments_2d(locus["segments"], points_per_segment)
    points_3d = np.zeros((len(points_2d), 3), dtype=float)
    plane_idx = _axis_index(profile["plane_axis"])
    u_idx = _axis_index(profile["profile_axes"][0])
    v_idx = _axis_index(profile["profile_axes"][1])
    points_3d[:, plane_idx] = float(profile["plane_value"])
    points_3d[:, u_idx] = points_2d[:, 0]
    points_3d[:, v_idx] = points_2d[:, 1]
    return points_3d


def closed_path_gap(locus: dict[str, Any]) -> float:
    """Return the endpoint distance between the last and first locus segments."""

    segments = locus["segments"]
    if not segments:
        return float("inf")
    start = np.asarray(segments[0]["start"], dtype=float)
    end = np.asarray(segments[-1]["end"], dtype=float)
    return float(np.linalg.norm(end - start))


def _nearest_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lhs = np.asarray(a, dtype=float)
    rhs = np.asarray(b, dtype=float)
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[1]:
        raise ValueError("point arrays must have matching shape (N, D) and (M, D)")
    if len(lhs) == 0 or len(rhs) == 0:
        raise ValueError("point arrays must be non-empty")
    diff = lhs[:, None, :] - rhs[None, :, :]
    return np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))


def symmetric_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Return symmetric nearest-neighbor RMSE between two sampled paths."""

    d_ab = _nearest_distances(a, b)
    d_ba = _nearest_distances(b, a)
    return float(np.sqrt(0.5 * (np.mean(d_ab * d_ab) + np.mean(d_ba * d_ba))))


def symmetric_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Return symmetric Hausdorff distance between two sampled paths."""

    d_ab = _nearest_distances(a, b)
    d_ba = _nearest_distances(b, a)
    return float(max(np.max(d_ab), np.max(d_ba)))
