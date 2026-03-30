from __future__ import annotations

from typing import Any

import numpy as np

from datasets.types import (
    FrontendBatchOutput,
    InstancePrediction,
    RGBDFrame,
)
from networks.rgb_frontend import RGBDInstanceFrontend, ProvidedMaskFrontend


class SAM2AutoFrontend(RGBDInstanceFrontend):
    """SAM2 automatic mask generation frontend for single-workpiece scenes."""

    def __init__(
        self,
        sam2_checkpoint: str,
        sam2_model_cfg: str,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_area: int = 1000,
        max_masks: int = 1,
        device: str = "cuda",
    ) -> None:
        self.device_name = device
        self.min_mask_area = min_mask_area
        self.max_masks = max_masks
        self._fallback = ProvidedMaskFrontend(min_pixels=10)

        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )

    def predict(
        self, frame: RGBDFrame, data: dict[str, Any] | None = None
    ) -> FrontendBatchOutput:
        if data is not None and ("mask" in data or "masks" in data):
            return self._fallback.predict(frame, data=data)

        rgb = self._to_rgb_uint8(frame.rgb)
        masks_output = self.mask_generator.generate(rgb)

        if not masks_output:
            raise ValueError("SAM2 generated no mask candidates")

        selected = self._filter_and_select(masks_output, rgb.shape[:2])
        instances = self._build_instances(selected)
        image_size = (rgb.shape[0], rgb.shape[1])
        return FrontendBatchOutput(instances=instances, image_size=image_size)

    def _filter_and_select(
        self,
        masks_output: list[dict[str, Any]],
        image_shape: tuple[int, int],
    ) -> list[dict[str, Any]]:
        total_pixels = image_shape[0] * image_shape[1]
        max_area = total_pixels * 0.5

        filtered = [
            m
            for m in masks_output
            if m["area"] >= self.min_mask_area and m["area"] <= max_area
        ]

        if not filtered:
            filtered = sorted(masks_output, key=lambda m: m["predicted_iou"], reverse=True)
            if filtered:
                filtered = [filtered[0]]

        filtered.sort(key=lambda m: m["area"], reverse=True)
        return filtered[: self.max_masks]

    def _build_instances(
        self, selected: list[dict[str, Any]]
    ) -> list[InstancePrediction]:
        instances: list[InstancePrediction] = []
        for idx, mask_dict in enumerate(selected):
            binary_mask = mask_dict["segmentation"].astype(np.uint8)
            instances.append(
                InstancePrediction(
                    instance_id=idx,
                    mask=binary_mask,
                    score=float(mask_dict["predicted_iou"]),
                    class_id=None,
                    extra={
                        "stability_score": float(mask_dict["stability_score"]),
                        "area": int(mask_dict["area"]),
                    },
                )
            )
        return instances

    @staticmethod
    def _to_rgb_uint8(rgb) -> np.ndarray:
        import torch

        if isinstance(rgb, torch.Tensor):
            arr = rgb.detach().cpu().numpy()
        else:
            arr = np.asarray(rgb)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        return arr
