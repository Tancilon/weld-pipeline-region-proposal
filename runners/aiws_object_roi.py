from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


class ObjectRoiError(RuntimeError):
    pass


EOMT_CLASS_NAME_MAP = {
    "盖板": "cover_plate",
    "方管": "square_tube",
    "喇叭口": "bellmouth",
}


@dataclass(frozen=True)
class ObjectRoiResult:
    workpiece_type: str
    class_id: int
    class_confidence: float
    object_mask: np.ndarray
    bbox_xywh: list[int]
    masked_full_rgb_path: Path
    object_mask_path: Path
    roi_metadata_path: Path


def normalize_eomt_class_name(class_name: str) -> str:
    try:
        return EOMT_CLASS_NAME_MAP[class_name]
    except KeyError as exc:
        raise ObjectRoiError(f"unsupported EoMT class: {class_name}") from exc


def _bbox_xywh(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ObjectRoiError("empty object mask")
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return [x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1)]


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _masked_full_rgb(rgb: Image.Image, object_mask: np.ndarray) -> Image.Image:
    rgb_array = np.asarray(rgb.convert("RGB")).copy()
    rgb_array[~object_mask.astype(np.bool_)] = [255, 255, 255]
    return Image.fromarray(rgb_array, mode="RGB")


def _largest_component_filter(object_mask: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    mask = object_mask.astype(np.bool_)
    total_area = int(mask.sum())
    metadata = {
        "enabled": True,
        "connectivity": 8,
        "num_components_before": 0,
        "kept_area": 0,
        "removed_area": 0,
        "removed_area_ratio": 0.0,
    }
    if total_area == 0:
        return mask, metadata

    try:
        import cv2

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        component_count = int(num_labels - 1)
        if component_count <= 0:
            return mask, metadata
        areas = stats[1:, cv2.CC_STAT_AREA]
        kept_label = int(np.argmax(areas) + 1)
        kept_area = int(areas[kept_label - 1])
        filtered = labels == kept_label
    except Exception:
        filtered, component_count, kept_area = _largest_component_filter_numpy(mask)

    removed_area = int(total_area - kept_area)
    metadata.update(
        {
            "num_components_before": int(component_count),
            "kept_area": int(kept_area),
            "removed_area": removed_area,
            "removed_area_ratio": float(removed_area / total_area),
        }
    )
    return filtered.astype(np.bool_), metadata


def _largest_component_filter_numpy(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=np.bool_)
    largest_component: list[tuple[int, int]] = []
    component_count = 0
    neighbors = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    for start_y, start_x in zip(*np.where(mask & ~visited)):
        if visited[start_y, start_x]:
            continue
        component_count += 1
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        component: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and mask[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(component) > len(largest_component):
            largest_component = component

    filtered = np.zeros(mask.shape, dtype=np.bool_)
    for y, x in largest_component:
        filtered[y, x] = True
    return filtered, component_count, len(largest_component)


def _write_roi_artifacts(
    rgb_path: Path,
    object_mask: np.ndarray,
    output_dir: Path,
    sample_id: str,
    workpiece_type: str,
    class_id: int,
    class_confidence: float,
    source_depth_path: Path | None = None,
) -> ObjectRoiResult:
    rgb = Image.open(rgb_path).convert("RGB")
    width, height = rgb.size
    if object_mask.shape != (height, width):
        raise ObjectRoiError(
            f"object mask shape mismatch: {object_mask.shape} vs {(height, width)}"
        )
    object_mask, component_filter = _largest_component_filter(object_mask)
    bbox = _bbox_xywh(object_mask)

    output_dir.mkdir(parents=True, exist_ok=True)
    roi_dir = output_dir / "object_roi"
    roi_dir.mkdir(parents=True, exist_ok=True)
    object_mask_path = output_dir / "object_mask.png"
    masked_full_rgb_path = roi_dir / "rgb_masked_full.png"
    roi_metadata_path = roi_dir / "object_roi.json"

    Image.fromarray(object_mask.astype(np.uint8) * 255, mode="L").save(object_mask_path)
    _masked_full_rgb(rgb, object_mask).save(masked_full_rgb_path)

    payload = {
        "schema_version": 1,
        "sample_id": sample_id,
        "source_rgb_path": _relative(rgb_path, output_dir),
        "source_depth_path": _relative(source_depth_path, output_dir)
        if source_depth_path
        else "",
        "workpiece_type": workpiece_type,
        "class_id": int(class_id),
        "class_confidence": float(class_confidence),
        "object_mask_path": _relative(object_mask_path, output_dir),
        "bbox_xywh": bbox,
        "semantic_sam_input_mode": "masked_full",
        "masked_full_rgb_path": _relative(masked_full_rgb_path, output_dir),
        "background_fill_rgb": [255, 255, 255],
        "source_shape_hw": [int(height), int(width)],
        "component_filter": component_filter,
    }
    roi_metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ObjectRoiResult(
        workpiece_type=workpiece_type,
        class_id=int(class_id),
        class_confidence=float(class_confidence),
        object_mask=object_mask.astype(np.bool_),
        bbox_xywh=bbox,
        masked_full_rgb_path=masked_full_rgb_path,
        object_mask_path=object_mask_path,
        roi_metadata_path=roi_metadata_path,
    )


def build_object_roi_from_mask(
    rgb_path: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    sample_id: str,
    workpiece_type: str,
    class_id: int,
    class_confidence: float,
    source_depth_path: str | Path | None = None,
) -> ObjectRoiResult:
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    return _write_roi_artifacts(
        rgb_path=Path(rgb_path),
        object_mask=mask,
        output_dir=Path(output_dir),
        sample_id=sample_id,
        workpiece_type=workpiece_type,
        class_id=class_id,
        class_confidence=class_confidence,
        source_depth_path=Path(source_depth_path) if source_depth_path else None,
    )


class ObjectRoiEstimator:
    def __init__(self, seg_ckpt: str, device: str = "cuda", img_size: int = 224):
        self.seg_ckpt = seg_ckpt
        self.device = device
        self.img_size = img_size
        self._runtime_bundle = None

    def _load_runtime_bundle(self):
        if self._runtime_bundle is not None:
            return self._runtime_bundle
        original_argv = list(sys.argv)
        try:
            sys.argv = sys.argv[:1]
            import cv2
            from datasets.datasets_nuclear import CLASS_NAMES
            from runners.infer_nuclear_full_lib import init_pipeline_agents

            args = SimpleNamespace(
                seg_ckpt=self.seg_ckpt,
                energy_ckpt=None,
                scale_ckpt=None,
                repeat_num=1,
                num_points=1024,
                img_size=self.img_size,
                device=self.device,
            )
            main_cfg, main_agent, energy_agent, scale_agent = init_pipeline_agents(args)
            _ = energy_agent
            _ = scale_agent
        finally:
            sys.argv = original_argv
        self._runtime_bundle = SimpleNamespace(
            cv2=cv2,
            torch=torch,
            CLASS_NAMES=CLASS_NAMES,
            main_cfg=main_cfg,
            main_agent=main_agent,
        )
        return self._runtime_bundle

    def _load_rgb(self, rgb_path: str | Path) -> np.ndarray:
        runtime = self._load_runtime_bundle()
        image = runtime.cv2.imread(str(rgb_path), runtime.cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        return runtime.cv2.cvtColor(image, runtime.cv2.COLOR_BGR2RGB)

    def _prepare_batch(self, rgb: np.ndarray) -> dict[str, torch.Tensor]:
        runtime = self._load_runtime_bundle()
        roi_rgb = runtime.cv2.resize(
            rgb,
            (self.img_size, self.img_size),
            interpolation=runtime.cv2.INTER_LINEAR,
        )
        roi_rgb = np.transpose(roi_rgb, (2, 0, 1)).astype(np.float32) / 255.0
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        for channel in range(3):
            roi_rgb[channel] = (roi_rgb[channel] - mean[channel]) / std[channel]
        return {
            "roi_rgb": torch.as_tensor(
                roi_rgb,
                dtype=torch.float32,
                device=runtime.main_cfg.device,
            ).unsqueeze(0)
        }

    def _top_prediction(
        self,
        class_logits: torch.Tensor,
        mask_logits: torch.Tensor,
        score_threshold: float,
    ) -> tuple[np.ndarray, int, float]:
        probs = class_logits.softmax(dim=-1)
        obj_probs = probs[:, :-1]
        max_scores, max_classes = obj_probs.max(dim=-1)
        best = None
        for idx in range(len(max_scores)):
            score = float(max_scores[idx].item())
            if score < score_threshold:
                continue
            cls_id = int(max_classes[idx].item())
            mask = mask_logits[idx].sigmoid().detach().cpu().numpy()
            if best is None or score > best[2]:
                best = (mask, cls_id, score)
        if best is None:
            raise ObjectRoiError(f"no EoMT instance above threshold {score_threshold}")
        return best

    def estimate(
        self,
        rgb_path: str | Path,
        output_dir: str | Path,
        sample_id: str,
        score_threshold: float,
        source_depth_path: str | Path | None = None,
    ) -> ObjectRoiResult:
        runtime = self._load_runtime_bundle()
        rgb = self._load_rgb(rgb_path)
        with torch.no_grad():
            class_logits, mask_logits = runtime.main_agent.net(
                self._prepare_batch(rgb),
                mode="segmentation",
            )
        mask_crop, cls_id, score = self._top_prediction(
            class_logits[0],
            mask_logits[0],
            score_threshold=score_threshold,
        )
        mask = runtime.cv2.resize(
            mask_crop.astype(np.float32),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=runtime.cv2.INTER_LINEAR,
        ) > 0.5
        workpiece_type = normalize_eomt_class_name(runtime.CLASS_NAMES[cls_id])
        return _write_roi_artifacts(
            rgb_path=Path(rgb_path),
            object_mask=mask,
            output_dir=Path(output_dir),
            sample_id=sample_id,
            workpiece_type=workpiece_type,
            class_id=cls_id,
            class_confidence=score,
            source_depth_path=Path(source_depth_path) if source_depth_path else None,
        )
