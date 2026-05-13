"""Validation helpers for CatSpec-derived weld loci."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from catspec.locus import (
    build_closed_rounded_rect_locus,
    closed_path_gap,
    estimate_square_tube_profile,
    sample_locus_3d,
    sample_segments_2d,
    symmetric_hausdorff,
    symmetric_rmse,
)
from catspec.schema import load_catspec, resolve_asset_path
from weld.core import back_project, load_weld_mesh
from weld.strategies.square_tube import SquareTubeStrategy


EXPECTED_ROUNDED_RECT_TYPES = ["line", "arc", "line", "arc", "line", "arc", "line", "arc"]


def _circle_from_three_points(p0: np.ndarray, pm: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, float]:
    ax, ay = p0
    bx, by = pm
    cx, cy = p1
    denominator = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(denominator) < 1e-12:
        center = np.mean(np.vstack([p0, pm, p1]), axis=0)
        radius = float(np.max(np.linalg.norm(np.vstack([p0, pm, p1]) - center, axis=1)))
        return center, radius

    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / denominator
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / denominator
    center = np.array([ux, uy], dtype=float)
    radius = float(np.linalg.norm(p0 - center))
    return center, radius


def _unwrap_arc_angles(start: float, mid: float, end: float) -> tuple[float, float]:
    mid_ccw = (mid - start) % (2 * np.pi)
    end_ccw = (end - start) % (2 * np.pi)
    if mid_ccw < end_ccw:
        return start, start + end_ccw
    return start, start + end_ccw - 2 * np.pi


def _sample_reference_path_3d(path: dict[str, Any], points_per_segment: int) -> np.ndarray:
    """Sample a SquareTubeStrategy fitted path and back-project it to 3D."""

    segments_2d: list[dict[str, Any]] = []
    for segment in path["fitted"]:
        points = np.asarray(segment["points_2d"], dtype=float)
        if segment["type"] == "line":
            segments_2d.append(
                {
                    "type": "line",
                    "start": tuple(points[0]),
                    "end": tuple(points[-1]),
                }
            )
        elif segment["type"] == "arc":
            center, radius = _circle_from_three_points(points[0], points[1], points[2])
            start_angle = float(np.arctan2(points[0, 1] - center[1], points[0, 0] - center[0]))
            mid_angle = float(np.arctan2(points[1, 1] - center[1], points[1, 0] - center[0]))
            end_angle = float(np.arctan2(points[2, 1] - center[1], points[2, 0] - center[0]))
            start_angle, end_angle = _unwrap_arc_angles(start_angle, mid_angle, end_angle)
            segments_2d.append(
                {
                    "type": "arc",
                    "center": tuple(center),
                    "radius": radius,
                    "start_angle": start_angle,
                    "end_angle": end_angle,
                    "start": tuple(points[0]),
                    "end": tuple(points[-1]),
                }
            )
        else:
            raise ValueError(f"unsupported fitted segment type: {segment['type']!r}")

    points_2d = sample_segments_2d(segments_2d, points_per_segment)
    return back_project(points_2d, path["plane"])


def _project_to_plane(points_3d: np.ndarray, plane: dict[str, Any]) -> np.ndarray:
    """Project 3D points onto a PCA plane."""

    points = np.asarray(points_3d, dtype=float)
    diffs = points - np.asarray(plane["origin"], dtype=float)
    u = diffs @ np.asarray(plane["u"], dtype=float)
    v = diffs @ np.asarray(plane["v"], dtype=float)
    return np.column_stack([u, v])


def _write_overlay(
    reference_3d: np.ndarray,
    generated_3d: np.ndarray,
    plane: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Write a 2D PCA-plane overlay of reference and generated weld paths."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    reference_2d = _project_to_plane(reference_3d, plane)
    generated_2d = _project_to_plane(generated_3d, plane)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _write_basic_png_overlay(reference_2d, generated_2d, output)
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(
        reference_2d[:, 0],
        reference_2d[:, 1],
        color="tab:blue",
        linewidth=2.0,
        label="reference weld",
    )
    ax.plot(
        generated_2d[:, 0],
        generated_2d[:, 1],
        color="tab:orange",
        linewidth=1.6,
        linestyle="--",
        label="CatSpec locus",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _write_basic_png_overlay(reference_2d: np.ndarray, generated_2d: np.ndarray, output_path: Path) -> None:
    """Write a dependency-free RGB PNG when matplotlib is unavailable."""

    import struct
    import zlib

    width = 700
    height = 700
    pad = 40
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    all_points = np.vstack([reference_2d, generated_2d])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-12)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])

    def to_pixel(points: np.ndarray) -> np.ndarray:
        xy = (points - mins) * scale
        x = np.rint(xy[:, 0] + pad).astype(int)
        y = np.rint(height - pad - xy[:, 1]).astype(int)
        return np.column_stack([x, y])

    def draw_line(p0: np.ndarray, p1: np.ndarray, color: tuple[int, int, int]) -> None:
        x0, y0 = p0
        x1, y1 = p1
        steps = int(max(abs(x1 - x0), abs(y1 - y0), 1))
        xs = np.rint(np.linspace(x0, x1, steps + 1)).astype(int)
        ys = np.rint(np.linspace(y0, y1, steps + 1)).astype(int)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xx = np.clip(xs + dx, 0, width - 1)
                yy = np.clip(ys + dy, 0, height - 1)
                canvas[yy, xx] = color

    for pts, color in ((to_pixel(reference_2d), (31, 119, 180)), (to_pixel(generated_2d), (255, 127, 14))):
        for idx in range(len(pts) - 1):
            draw_line(pts[idx], pts[idx + 1], color)

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + canvas[row].tobytes() for row in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )
    output_path.write_bytes(png)


def _generate_square_tube_locus(spec: dict[str, Any], workpiece_path: str | Path) -> dict[str, Any]:
    """Generate the analytic CatSpec locus from the workpiece mesh only."""

    params = spec["welds"][0]["locus"]["params"]
    mesh = trimesh.load(workpiece_path, process=False)
    profile = estimate_square_tube_profile(
        mesh,
        plane_axis=str(params["plane_axis"]),
        plane_side=params["plane_side"],
        profile_axes=tuple(params["profile_axes"]),
        profile_quantile=float(params["profile_quantile"]),
    )
    return build_closed_rounded_rect_locus(profile)


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def validate_square_tube(spec_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Compare a CatSpec-derived square-tube locus to the weld OBJ reference."""

    spec_file = Path(spec_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    spec = load_catspec(spec_file)
    workpiece_path = resolve_asset_path(spec["provenance"]["source_mesh"], spec_file)
    weld_path = resolve_asset_path(spec["provenance"]["source_weld_mesh"], spec_file)
    points_per_segment = int(spec["welds"][0]["locus"]["params"]["sample_points_per_segment"])

    generated_locus = _generate_square_tube_locus(spec, workpiece_path)
    generated_3d = sample_locus_3d(generated_locus, points_per_segment)

    weld_mesh = load_weld_mesh(str(weld_path))
    reference_paths = SquareTubeStrategy().process(weld_mesh)
    reference_path = reference_paths[0]
    reference_3d = _sample_reference_path_3d(reference_path, points_per_segment)

    generated_types = [segment["type"] for segment in generated_locus["segments"]]
    reference_types = [segment["type"] for segment in reference_path["fitted"]]
    topology_match = (
        bool(generated_locus.get("closed"))
        and bool(reference_path.get("closed"))
        and generated_types == EXPECTED_ROUNDED_RECT_TYPES
        and reference_types == EXPECTED_ROUNDED_RECT_TYPES
    )

    report_path = output / "square_tube_catspec_validation.json"
    overlay_path = output / "square_tube_catspec_overlay.png"
    _write_overlay(reference_3d, generated_3d, reference_path["plane"], overlay_path)

    report = {
        "category": spec["category"],
        "schema_version": spec["schema_version"],
        "workpiece_path": str(workpiece_path),
        "reference_weld_path": str(weld_path),
        "topology_match": topology_match,
        "generated": {
            "closed": bool(generated_locus.get("closed")),
            "segment_types": generated_types,
            "points": int(len(generated_3d)),
        },
        "reference": {
            "closed": bool(reference_path.get("closed")),
            "segment_types": reference_types,
            "points": int(len(reference_3d)),
        },
        "metrics": {
            "centerline_rmse": symmetric_rmse(generated_3d, reference_3d),
            "hausdorff": symmetric_hausdorff(generated_3d, reference_3d),
            "closed_path_gap": closed_path_gap(generated_locus),
        },
        "report_path": str(report_path),
        "overlay_path": str(overlay_path),
    }

    report_path.write_text(json.dumps(report, indent=2, default=_jsonify) + "\n", encoding="utf-8")
    return report
