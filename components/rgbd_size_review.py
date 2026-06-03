from __future__ import annotations

import math
from typing import Any

import numpy as np


def _intrinsic_value(intrinsics: dict[str, Any], key: str) -> float:
    try:
        value = float(intrinsics[key])
    except Exception as exc:
        raise ValueError(f"missing camera intrinsic: {key}") from exc
    if not math.isfinite(value) or value == 0.0:
        raise ValueError(f"invalid camera intrinsic: {key}")
    return value


def depth_mask_to_camera_points(
    *,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict[str, Any],
) -> np.ndarray:
    depth_arr = np.asarray(depth, dtype=np.float64)
    mask_arr = np.asarray(mask)
    if depth_arr.ndim != 2:
        raise ValueError("depth must be a 2D array")
    if mask_arr.shape != depth_arr.shape:
        raise ValueError("mask shape must match depth shape")

    fx = _intrinsic_value(intrinsics, "fx")
    fy = _intrinsic_value(intrinsics, "fy")
    cx = _intrinsic_value(intrinsics, "cx")
    cy = _intrinsic_value(intrinsics, "cy")

    valid = (mask_arr > 0) & np.isfinite(depth_arr) & (depth_arr > 0.0)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)

    z = depth_arr[ys, xs]
    return np.column_stack(
        [
            (xs.astype(np.float64) - cx) * z / fx,
            (ys.astype(np.float64) - cy) * z / fy,
            z,
        ]
    )


def robust_pca_extents_mm(
    points_camera_m: np.ndarray,
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    min_points: int = 50,
) -> list[float]:
    points = np.asarray(points_camera_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera_m must be an Nx3 array")
    finite = points[np.all(np.isfinite(points), axis=1)]
    if len(finite) < min_points:
        raise ValueError("not enough finite RGB-D points for size review")

    center = np.median(finite, axis=0)
    centered = finite - center
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    projected = centered @ vectors[:, order]
    lower, upper = np.percentile(
        projected,
        [float(lower_percentile), float(upper_percentile)],
        axis=0,
    )
    extents = np.maximum(upper - lower, 0.0) * 1000.0
    return [float(value) for value in extents]


def _candidate_geometry_errors(
    observed_extents_mm: list[float],
    candidate_sizes_mm: list[list[float]],
) -> list[dict[str, Any]]:
    observed = sorted(float(value) for value in observed_extents_mm)
    errors: list[dict[str, Any]] = []
    for candidate in candidate_sizes_mm:
        candidate_values = [float(value) for value in candidate]
        candidate_sorted = sorted(candidate_values)
        axis_errors = [
            abs(observed_value - candidate_value) / max(candidate_value, 1e-6)
            for observed_value, candidate_value in zip(observed, candidate_sorted)
        ]
        score = sum(axis_errors) / 3.0
        errors.append(
            {
                "candidate_size_xyz_mm": candidate_values,
                "candidate_sorted_xyz_mm": candidate_sorted,
                "axis_relative_errors": [float(value) for value in axis_errors],
                "score": float(score),
            }
        )
    errors.sort(key=lambda item: (float(item["score"]), item["candidate_size_xyz_mm"]))
    return errors


def review_standard_size_match_from_points(
    *,
    base_match: dict[str, Any],
    candidate_sizes_mm: list[list[float]],
    points_camera_m: np.ndarray,
    confidence_threshold: float = 0.02,
    geometry_improvement_margin: float = 0.08,
    sigma: float = 0.05,
) -> dict[str, Any]:
    base_confidence = float(base_match.get("match_confidence", 0.0))
    if base_confidence >= confidence_threshold:
        return dict(base_match)

    observed_extents = robust_pca_extents_mm(points_camera_m)
    candidate_errors = _candidate_geometry_errors(observed_extents, candidate_sizes_mm)
    if not candidate_errors:
        return dict(base_match)

    best = candidate_errors[0]
    second_score = (
        float(candidate_errors[1]["score"]) if len(candidate_errors) > 1 else float("inf")
    )
    base_size = [float(value) for value in base_match.get("matched_size_xyz_mm", [])]
    best_size = [float(value) for value in best["candidate_size_xyz_mm"]]
    best_score = float(best["score"])
    should_override = (
        best_size != base_size
        and second_score - best_score >= float(geometry_improvement_margin)
    )
    if not should_override:
        result = dict(base_match)
        result["size_match_review"] = {
            "review_method": "rgbd_pca_extents_v1",
            "decision": "kept_base_match",
            "base_match_confidence": base_confidence,
            "confidence_threshold": float(confidence_threshold),
            "observed_extents_mm": observed_extents,
            "candidate_errors": candidate_errors,
        }
        return result

    return {
        "matched_size_xyz_mm": best_size,
        "size_match_error": best_score,
        "match_confidence": float(math.exp(-best_score / float(sigma))),
        "size_match_method": "rgbd_geometry_review_v1",
        "size_match_review": {
            "review_method": "rgbd_pca_extents_v1",
            "decision": "overrode_low_confidence_base_match",
            "base_match": dict(base_match),
            "base_match_confidence": base_confidence,
            "confidence_threshold": float(confidence_threshold),
            "observed_extents_mm": observed_extents,
            "candidate_errors": candidate_errors,
            "geometry_improvement_margin": float(geometry_improvement_margin),
        },
    }


def review_standard_size_match(
    *,
    base_match: dict[str, Any],
    candidate_sizes_mm: list[list[float]],
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict[str, Any],
    confidence_threshold: float = 0.02,
    geometry_improvement_margin: float = 0.08,
    sigma: float = 0.05,
) -> dict[str, Any]:
    points = depth_mask_to_camera_points(depth=depth, mask=mask, intrinsics=intrinsics)
    return review_standard_size_match_from_points(
        base_match=base_match,
        candidate_sizes_mm=candidate_sizes_mm,
        points_camera_m=points,
        confidence_threshold=confidence_threshold,
        geometry_improvement_margin=geometry_improvement_margin,
        sigma=sigma,
    )
