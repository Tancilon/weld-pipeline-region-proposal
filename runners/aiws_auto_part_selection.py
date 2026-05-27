from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class AutoPartSelectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AutoPartSelectionResult:
    accepted: bool
    selected_parts_path: Path | None
    diagnostics_path: Path
    reason: str


def _clamp01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _resolve_path(path_value: str | Path, output_root: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = output_root / path
    return path.resolve()


def _load_bool_mask(path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if mask.shape == target_shape:
        return mask
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    height, width = target_shape
    resized = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
        (width, height),
        resample=nearest,
    )
    return np.asarray(resized) > 0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _part_area_ideal(workpiece_type: str, part_name: str) -> float:
    table = {
        ("cover_plate", "tube"): 0.70,
        ("square_tube", "tube"): 0.55,
        ("bellmouth", "tube"): 0.45,
        ("bellmouth", "plate"): 0.28,
        ("square_tube", "plate"): 0.25,
    }
    return table.get((workpiece_type, part_name), 0.50 if part_name == "tube" else 0.30)


def _part_aspect_ideal(part_name: str) -> float:
    return 1.7 if part_name == "tube" else 3.0


def _score_area(area_ratio: float, ideal: float) -> float:
    tolerance = max(0.25, ideal * 0.75)
    return _clamp01(1.0 - abs(area_ratio - ideal) / tolerance)


def _score_aspect(bbox_xywh: list[int], ideal: float) -> float:
    width = max(float(bbox_xywh[2]), 1.0)
    height = max(float(bbox_xywh[3]), 1.0)
    aspect = max(width, height) / max(min(width, height), 1.0)
    return _clamp01(1.0 - abs(aspect - ideal) / max(ideal, 1.0))


def _evaluate_candidate(
    candidate: dict[str, Any],
    *,
    output_root: Path,
    object_area: int,
    target_shape: tuple[int, int],
    workpiece_type: str,
    part_name: str,
    min_area_ratio_of_object: float,
    max_area_ratio_of_object: float,
    min_object_overlap_ratio: float,
    min_depth_valid_ratio: float,
    min_depth_valid_pixels: int,
) -> dict[str, Any]:
    mask_id = str(candidate.get("mask_id", ""))
    mask_path_value = str(candidate.get("mask_path", ""))
    mask_path = _resolve_path(mask_path_value, output_root)
    rejection_reasons: list[str] = []
    if not mask_path.exists():
        rejection_reasons.append("mask_path_missing")
    mask_area = 0
    if mask_path.exists():
        mask = _load_bool_mask(mask_path, target_shape)
        mask_area = int(mask.sum())
    area_px = int(candidate.get("area_px", mask_area))
    if area_px <= 0:
        area_px = mask_area
    area_ratio = _safe_ratio(area_px, object_area)
    bbox_xywh = [int(value) for value in candidate.get("bbox_xywh", [0, 0, 0, 0])[:4]]
    object_overlap = candidate.get("object_overlap_ratio", 0.0)
    object_overlap = 0.0 if object_overlap is None else float(object_overlap)
    depth_valid_pixels = int(candidate.get("depth_valid_pixels", 0))
    depth_valid_ratio = float(candidate.get("depth_valid_ratio", 0.0))

    if object_overlap < min_object_overlap_ratio:
        rejection_reasons.append("object_overlap_below_min")
    if area_ratio < min_area_ratio_of_object:
        rejection_reasons.append("area_ratio_below_min")
    if area_ratio > max_area_ratio_of_object:
        rejection_reasons.append("area_ratio_above_max")
    if depth_valid_pixels < min_depth_valid_pixels:
        rejection_reasons.append("depth_valid_pixels_below_min")
    if depth_valid_ratio < min_depth_valid_ratio:
        rejection_reasons.append("depth_valid_ratio_below_min")
    if len(bbox_xywh) != 4 or bbox_xywh[2] <= 0 or bbox_xywh[3] <= 0:
        rejection_reasons.append("bbox_empty")

    area_score = _score_area(area_ratio, _part_area_ideal(workpiece_type, part_name))
    shape_score = _score_aspect(bbox_xywh, _part_aspect_ideal(part_name))
    score_components = {
        "object_overlap": _clamp01(
            (object_overlap - min_object_overlap_ratio)
            / max(1.0 - min_object_overlap_ratio, 1e-6)
        ),
        "depth_valid_ratio": _clamp01(depth_valid_ratio / 0.60),
        "area": area_score,
        "bbox_shape": shape_score,
        "predicted_iou": _clamp01(float(candidate.get("predicted_iou", 0.0))),
        "stability_score": _clamp01(float(candidate.get("stability_score", 0.0))),
    }
    score = (
        score_components["object_overlap"] * 0.25
        + score_components["depth_valid_ratio"] * 0.25
        + score_components["area"] * 0.25
        + score_components["bbox_shape"] * 0.15
        + score_components["predicted_iou"] * 0.05
        + score_components["stability_score"] * 0.05
    )
    return {
        "mask_id": mask_id,
        "mask_path": str(mask_path),
        "score": float(score),
        "score_components": score_components,
        "area_ratio_of_object": float(area_ratio),
        "bbox_xywh": bbox_xywh,
        "depth_valid_pixels": depth_valid_pixels,
        "depth_valid_ratio": depth_valid_ratio,
        "object_overlap_ratio": float(object_overlap),
        "rejection_reasons": rejection_reasons,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def select_weld_focus_masks(
    *,
    candidates_path: str | Path,
    output_root: str | Path,
    weld_focus: list[str],
    object_mask: np.ndarray,
    depth: np.ndarray,
    accept_score_threshold: float = 0.65,
    min_score_margin: float = 0.10,
    allow_low_confidence: bool = False,
    min_area_ratio_of_object: float = 0.03,
    max_area_ratio_of_object: float = 0.90,
    min_object_overlap_ratio: float = 0.95,
    min_depth_valid_ratio: float = 0.15,
    min_depth_valid_pixels: int = 100,
) -> AutoPartSelectionResult:
    output_root = Path(output_root).resolve()
    candidates_path = Path(candidates_path)
    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    object_mask_arr = np.asarray(object_mask, dtype=np.bool_)
    depth_arr = np.asarray(depth)
    if object_mask_arr.ndim != 2 or int(object_mask_arr.sum()) <= 0:
        raise AutoPartSelectionError("object mask is empty")
    if depth_arr.shape[:2] != object_mask_arr.shape:
        raise AutoPartSelectionError(
            f"depth shape must match object mask shape: {depth_arr.shape} "
            f"vs {object_mask_arr.shape}"
        )

    sample_id = str(payload.get("sample_id", ""))
    workpiece_type = str(payload.get("workpiece_type", ""))
    object_area = int(object_mask_arr.sum())
    diagnostics_parts: dict[str, Any] = {}
    selected_parts: dict[str, str] = {}
    accepted_all = True
    failure_reason = ""

    for part_name in weld_focus:
        evaluated = [
            _evaluate_candidate(
                candidate,
                output_root=output_root,
                object_area=object_area,
                target_shape=object_mask_arr.shape,
                workpiece_type=workpiece_type,
                part_name=part_name,
                min_area_ratio_of_object=min_area_ratio_of_object,
                max_area_ratio_of_object=max_area_ratio_of_object,
                min_object_overlap_ratio=min_object_overlap_ratio,
                min_depth_valid_ratio=min_depth_valid_ratio,
                min_depth_valid_pixels=min_depth_valid_pixels,
            )
            for candidate in payload.get("candidates", [])
        ]
        evaluated.sort(key=lambda item: (-float(item["score"]), str(item["mask_id"])))
        eligible = [item for item in evaluated if not item["rejection_reasons"]]
        best = eligible[0] if eligible else None
        second = eligible[1] if len(eligible) > 1 else None
        if best is None:
            accepted = False
            reason = "no_candidate_passed_filters"
        else:
            margin = (
                float(best["score"]) - float(second["score"])
                if second is not None
                else 1.0
            )
            if float(best["score"]) < accept_score_threshold:
                accepted = bool(allow_low_confidence)
                reason = (
                    "accepted_low_confidence"
                    if allow_low_confidence
                    else "score_below_threshold"
                )
            elif margin < min_score_margin:
                accepted = bool(allow_low_confidence)
                reason = "accepted_low_margin" if allow_low_confidence else "score_margin_below_threshold"
            else:
                accepted = True
                reason = "accepted"
        if not accepted:
            accepted_all = False
            failure_reason = "low_confidence_auto_selection"
        if accepted and best is not None:
            selected_parts[part_name] = str(best["mask_id"])
        diagnostics_parts[part_name] = {
            "decision": "accepted" if accepted else "rejected",
            "reason": reason,
            "selected_mask_id": str(best["mask_id"]) if accepted and best is not None else "",
            "score": float(best["score"]) if best is not None else 0.0,
            "score_margin": (
                float(best["score"]) - float(second["score"])
                if best is not None and second is not None
                else 1.0
                if best is not None
                else 0.0
            ),
            "candidates": evaluated,
        }

    diagnostics_path = output_root / "auto_part_selection.json"
    diagnostics_payload = {
        "schema_version": 1,
        "sample_id": sample_id,
        "workpiece_type": workpiece_type,
        "selection_source": "auto",
        "accept_score_threshold": float(accept_score_threshold),
        "min_score_margin": float(min_score_margin),
        "allow_low_confidence": bool(allow_low_confidence),
        "focused_parts": diagnostics_parts,
    }
    _write_json(diagnostics_path, diagnostics_payload)

    if not accepted_all:
        return AutoPartSelectionResult(
            accepted=False,
            selected_parts_path=None,
            diagnostics_path=diagnostics_path,
            reason=failure_reason or "low_confidence_auto_selection",
        )

    selected_path = output_root / "selected_parts.auto.json"
    _write_json(
        selected_path,
        {
            "schema_version": 1,
            "sample_id": sample_id,
            "workpiece_type": workpiece_type,
            "selection_source": "auto",
            "focused_parts": selected_parts,
            "scores": {
                part: {
                    "selected_mask_id": diagnostics_parts[part]["selected_mask_id"],
                    "score": diagnostics_parts[part]["score"],
                    "decision": diagnostics_parts[part]["decision"],
                }
                for part in selected_parts
            },
        },
    )
    return AutoPartSelectionResult(
        accepted=True,
        selected_parts_path=selected_path,
        diagnostics_path=diagnostics_path,
        reason="accepted",
    )
