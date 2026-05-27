from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class MaskCandidateError(RuntimeError):
    pass


def _as_bool_mask(mask_path: str | Path) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path).convert("L"))
    return mask > 0


def _resize_bool_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(np.bool_)
    height, width = target_shape
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    resized = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
        (width, height),
        resample=nearest,
    )
    return np.asarray(resized) > 0


def _bbox_xywh(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return [x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1)]


def _relative_path(path_value: str | Path, output_root: Path) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = (output_root / path).resolve()
    else:
        path = path.resolve()
    try:
        return str(path.relative_to(output_root.resolve()))
    except ValueError:
        return str(path)


def _mask_id(mask_path: str | Path) -> str:
    return Path(mask_path).stem


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mask_to_metrics_space(
    mask: np.ndarray,
    metrics_shape: tuple[int, int],
    crop_bbox_xywh: list[int] | None,
) -> tuple[np.ndarray, list[int] | None]:
    if crop_bbox_xywh is None:
        return _resize_bool_mask(mask, metrics_shape), None

    x, y, w, h = [int(value) for value in crop_bbox_xywh]
    crop_mask = _resize_bool_mask(mask, (h, w))
    full_mask = np.zeros(metrics_shape, dtype=np.bool_)
    height, width = metrics_shape
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        raise MaskCandidateError(f"crop bbox outside metrics shape: {crop_bbox_xywh}")
    crop_x0 = x0 - x
    crop_y0 = y0 - y
    crop_x1 = crop_x0 + (x1 - x0)
    crop_y1 = crop_y0 + (y1 - y0)
    full_mask[y0:y1, x0:x1] = crop_mask[crop_y0:crop_y1, crop_x0:crop_x1]
    return full_mask, _bbox_xywh(crop_mask)


def _candidate_from_record(
    record: dict[str, Any],
    depth: np.ndarray,
    object_mask: np.ndarray | None,
    output_root: Path,
    crop_bbox_xywh: list[int] | None,
) -> dict[str, Any]:
    mask_path = Path(record["mask_path"])
    source_mask = _as_bool_mask(mask_path)
    source_shape = [int(value) for value in source_mask.shape]
    mask, roi_bbox = _mask_to_metrics_space(
        source_mask,
        tuple(int(value) for value in depth.shape),
        crop_bbox_xywh,
    )
    valid_depth = np.isfinite(depth) & (depth > 0)
    depth_valid_pixels = int((mask & valid_depth).sum())
    area_px = int(mask.sum())
    if object_mask is None:
        object_overlap_ratio = None
    else:
        if object_mask.shape != mask.shape:
            raise MaskCandidateError(
                f"object mask shape mismatch: {object_mask.shape} vs {mask.shape}"
            )
        object_overlap_ratio = _safe_ratio(int((mask & object_mask).sum()), int(mask.sum()))

    return {
        "mask_id": _mask_id(mask_path),
        "mask_path": _relative_path(mask_path, output_root),
        "area_px": area_px,
        "bbox_xywh": _bbox_xywh(mask),
        "roi_bbox_xywh": roi_bbox,
        "crop_bbox_xywh": [int(value) for value in crop_bbox_xywh]
        if crop_bbox_xywh
        else None,
        "coordinate_space": "original_image",
        "mask_coordinate_space": "semantic_sam_crop"
        if crop_bbox_xywh
        else "original_image",
        "predicted_iou": float(record.get("predicted_iou", 0.0)),
        "stability_score": float(record.get("stability_score", 0.0)),
        "depth_valid_pixels": depth_valid_pixels,
        "depth_valid_ratio": _safe_ratio(depth_valid_pixels, area_px),
        "object_overlap_ratio": object_overlap_ratio,
        "source_shape_hw": source_shape,
        "metrics_shape_hw": [int(value) for value in mask.shape],
    }


def build_mask_candidates(
    metadata_path: str | Path,
    output_path: str | Path,
    sample_id: str,
    workpiece_type: str,
    depth: np.ndarray,
    object_mask: np.ndarray | None = None,
    output_root: str | Path | None = None,
    crop_bbox_xywh: list[int] | None = None,
) -> Path:
    metadata_path = Path(metadata_path)
    output_path = Path(output_path)
    output_root = (
        Path(output_root).resolve() if output_root is not None else output_path.parent.resolve()
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = metadata.get("masks", [])
    if not records:
        raise MaskCandidateError(f"Semantic-SAM metadata has no masks: {metadata_path}")

    candidates = [
        _candidate_from_record(
            record,
            np.asarray(depth),
            object_mask,
            output_root,
            crop_bbox_xywh,
        )
        for record in records
    ]
    candidates.sort(
        key=lambda item: (
            -int(item["depth_valid_pixels"]),
            -int(item["area_px"]),
            -float(item["predicted_iou"]),
            item["mask_id"],
        )
    )
    payload = {
        "schema_version": 1,
        "sample_id": sample_id,
        "workpiece_type": workpiece_type,
        "semantic_sam": {
            "metadata_path": _relative_path(metadata_path, output_root),
            "overlay_path": _relative_path(metadata.get("overlay_path", ""), output_root),
            "levels": metadata.get("levels", []),
            "model_type": metadata.get("model_type", ""),
        },
        "candidates": candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


def write_selected_parts_template(
    output_path: str | Path,
    sample_id: str,
    workpiece_type: str,
    weld_focus: list[str],
) -> Path:
    output_path = Path(output_path)
    payload = {
        "schema_version": 1,
        "sample_id": sample_id,
        "workpiece_type": workpiece_type,
        "focused_parts": {part: "" for part in weld_focus},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path
