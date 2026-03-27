from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from datasets.types import (
    FrontendBatchOutput,
    InstancePrediction,
    RGBDFrame,
)
from networks.frontends.classification_head import InstanceClassificationHead
from networks.frontends.dinov2_backbone import DinoFeatureOutput, DinoV2Backbone
from networks.frontends.instance_head import InstanceSegmentationHead
from utils.datasets_utils import aug_bbox_eval, crop_resize_by_warp_affine
from utils.sgpa_utils import get_bbox


class RGBDInstanceFrontend:
    def predict(
        self, frame: RGBDFrame, data: dict[str, Any] | None = None
    ) -> FrontendBatchOutput:
        raise NotImplementedError


class DinoV2InstanceFrontend(RGBDInstanceFrontend, nn.Module):
    def __init__(
        self,
        backbone: DinoV2Backbone,
        classification_head: InstanceClassificationHead | None = None,
        instance_head: InstanceSegmentationHead | None = None,
        min_pixels: int = 10,
        img_size: int = 224,
    ) -> None:
        nn.Module.__init__(self)
        self.backbone = backbone
        self.classification_head = classification_head
        self.instance_head = instance_head
        self.min_pixels = min_pixels
        self.img_size = img_size

    def predict(
        self, frame: RGBDFrame, data: dict[str, Any] | None = None
    ) -> FrontendBatchOutput:
        rgb_tensor = self._prepare_rgb(frame.rgb)
        with torch.no_grad():
            features = self.backbone(rgb_tensor, return_global=True, return_dense=True)

        if data is not None and ("mask" in data or "masks" in data):
            instances = self._instances_from_masks(frame, features, data)
        elif self.instance_head is not None and features.feature_map is not None:
            instances = self._instances_from_head(features)
        else:
            raise NotImplementedError(
                "DinoV2InstanceFrontend currently needs provided masks or a trained instance head for prediction"
            )

        return FrontendBatchOutput(
            instances=instances,
            dense_feat_map=features.feature_map,
            global_feat=features.global_feat,
            image_size=self._image_size(frame.rgb),
        )

    def _instances_from_masks(
        self,
        frame: RGBDFrame,
        features: DinoFeatureOutput,
        data: dict[str, Any],
    ) -> list[InstancePrediction]:
        adapter = ProvidedMaskFrontend(min_pixels=self.min_pixels)
        base_output = adapter.predict(frame, data=data)
        enriched_instances: list[InstancePrediction] = []
        for instance in base_output.instances:
            bbox_xyxy = instance.bbox_xyxy
            if bbox_xyxy is None:
                bbox_xyxy = self._extract_bbox_from_mask(instance.mask)
            roi_rgb_raw = self._crop_roi_rgb(frame.rgb, bbox_xyxy)
            roi_features = self.backbone(
                self._prepare_rgb(roi_rgb_raw), return_global=True, return_dense=True
            )
            pooled_feat = (
                None
                if roi_features.global_feat is None
                else roi_features.global_feat[0]
            )
            class_id = instance.class_id
            extra = dict(instance.extra)
            if pooled_feat is not None:
                extra["pooled_rgb_feat"] = pooled_feat.detach().cpu()
                extra["roi_global_feat"] = pooled_feat.detach().cpu()
            if roi_features.patch_tokens is not None:
                extra["roi_patch_tokens"] = roi_features.patch_tokens[0].detach().cpu()
            if roi_features.feature_map is not None:
                extra["roi_feature_map"] = roi_features.feature_map[0].detach().cpu()
            if self.classification_head is not None and pooled_feat is not None:
                logits = self.classification_head(pooled_feat.unsqueeze(0)).squeeze(0)
                class_id = int(torch.argmax(logits).item())
                extra["class_logits"] = logits.detach().cpu()
            if self.instance_head is not None and roi_features.feature_map is not None:
                roi_head_output = self.instance_head(roi_features.feature_map)
                roi_mask_prob = torch.sigmoid(roi_head_output.mask_logits[0, 0])
                extra["roi_mask_logits"] = (
                    roi_head_output.mask_logits[0, 0].detach().cpu()
                )
                extra["roi_mask_pred"] = (
                    (roi_mask_prob > 0.5).detach().cpu().numpy().astype(np.uint8)
                )
            enriched_instances.append(
                InstancePrediction(
                    instance_id=instance.instance_id,
                    mask=instance.mask,
                    score=instance.score,
                    class_id=class_id,
                    class_name=instance.class_name,
                    bbox_xyxy=bbox_xyxy,
                    rgb_feat=None
                    if pooled_feat is None
                    else pooled_feat.detach().cpu(),
                    extra=extra,
                )
            )
        return enriched_instances

    def _instances_from_head(
        self, features: DinoFeatureOutput
    ) -> list[InstancePrediction]:
        if self.instance_head is None or features.feature_map is None:
            return []
        head_output = self.instance_head(features.feature_map)
        mask_prob = torch.sigmoid(head_output.mask_logits[0, 0])
        binary_mask = (mask_prob > 0.5).detach().cpu().numpy().astype(np.uint8)
        if np.sum(binary_mask) < self.min_pixels:
            return []
        score = float(torch.sigmoid(head_output.score_logits[0, 0]).item())
        class_id = None
        extra: dict[str, Any] = {"mask_logits": head_output.mask_logits.detach().cpu()}
        if head_output.class_logits is not None:
            logits = head_output.class_logits[0]
            class_id = int(torch.argmax(logits).item())
            extra["class_logits"] = logits.detach().cpu()
        pooled_feat = (
            None
            if features.global_feat is None
            else features.global_feat[0].detach().cpu()
        )
        bbox_xyxy = self._extract_bbox_from_mask(binary_mask)
        return [
            InstancePrediction(
                instance_id=0,
                mask=binary_mask,
                score=score,
                class_id=class_id,
                bbox_xyxy=bbox_xyxy,
                rgb_feat=pooled_feat,
                extra=extra,
            )
        ]

    def forward_train_batch(self, roi_rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(roi_rgb, return_global=True, return_dense=True)
        outputs: dict[str, torch.Tensor] = {}
        if features.global_feat is not None:
            outputs["global_feat"] = features.global_feat
        if features.feature_map is not None:
            outputs["feature_map"] = features.feature_map
        if self.classification_head is not None and features.global_feat is not None:
            outputs["class_logits"] = self.classification_head(features.global_feat)
        if self.instance_head is not None and features.feature_map is not None:
            head_output = self.instance_head(features.feature_map)
            outputs["mask_logits"] = head_output.mask_logits
            outputs["score_logits"] = head_output.score_logits
            if head_output.class_logits is not None:
                outputs["instance_class_logits"] = head_output.class_logits
        return outputs

    def predict_roi(
        self, roi_rgb: torch.Tensor, score_threshold: float = 0.5
    ) -> list[InstancePrediction]:
        outputs = self.forward_train_batch(roi_rgb)
        if "mask_logits" not in outputs:
            return []
        mask_prob = torch.sigmoid(outputs["mask_logits"][0, 0])
        binary_mask = (mask_prob > 0.5).detach().cpu().numpy().astype(np.uint8)
        if np.sum(binary_mask) < self.min_pixels:
            return []
        score = (
            float(torch.sigmoid(outputs["score_logits"][0, 0]).item())
            if "score_logits" in outputs
            else 1.0
        )
        if score < score_threshold:
            return []
        class_id = None
        if "class_logits" in outputs:
            class_id = int(torch.argmax(outputs["class_logits"][0]).item())
        elif "instance_class_logits" in outputs:
            class_id = int(torch.argmax(outputs["instance_class_logits"][0]).item())
        bbox_xyxy = self._extract_bbox_from_mask(binary_mask)
        return [
            InstancePrediction(
                instance_id=0,
                mask=binary_mask,
                score=score,
                class_id=class_id,
                bbox_xyxy=bbox_xyxy,
                rgb_feat=None
                if "global_feat" not in outputs
                else outputs["global_feat"][0].detach().cpu(),
                extra={"roi_mask_logits": outputs["mask_logits"][0, 0].detach().cpu()},
            )
        ]

    def _pool_instance_feature(
        self,
        mask: np.ndarray | torch.Tensor,
        features: DinoFeatureOutput,
    ) -> torch.Tensor | None:
        if features.feature_map is None:
            return None
        mask_tensor = torch.as_tensor(
            mask, dtype=torch.float32, device=features.feature_map.device
        )
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        resized = torch.nn.functional.interpolate(
            mask_tensor,
            size=features.feature_map.shape[-2:],
            mode="nearest",
        )[0, 0]
        valid = resized > 0
        if not torch.any(valid):
            return None
        feat_map = features.feature_map[0].permute(1, 2, 0)
        return feat_map[valid].mean(dim=0)

    def _extract_bbox_from_mask(self, mask: np.ndarray | torch.Tensor) -> np.ndarray:
        mask_np = ProvidedMaskFrontend._to_numpy(mask).astype(bool)
        ys, xs = np.argwhere(mask_np).transpose(1, 0)
        rmin, rmax, cmin, cmax = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
        rmin, rmax, cmin, cmax = get_bbox(
            [rmin, cmin, rmax, cmax], mask_np.shape[0], mask_np.shape[1]
        )
        return np.array([cmin, rmin, cmax, rmax], dtype=np.float32)

    def _crop_roi_rgb(
        self, rgb: np.ndarray | torch.Tensor, bbox_xyxy: np.ndarray
    ) -> np.ndarray:
        rgb_np = ProvidedMaskFrontend._to_numpy(rgb)
        img_height, img_width = rgb_np.shape[:2]
        bbox_center, scale = aug_bbox_eval(bbox_xyxy, img_height, img_width)
        return crop_resize_by_warp_affine(
            rgb_np, bbox_center, scale, self.img_size, interpolation=cv2.INTER_LINEAR
        )

    @staticmethod
    def _prepare_rgb(rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(rgb)
        if tensor.ndim == 3 and tensor.shape[0] != 3:
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor.float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor.unsqueeze(0)


class ProvidedMaskFrontend(RGBDInstanceFrontend):
    def __init__(
        self, background_ids: tuple[int, ...] = (255,), min_pixels: int = 10
    ) -> None:
        self.background_ids = background_ids
        self.min_pixels = min_pixels

    def predict(
        self, frame: RGBDFrame, data: dict[str, Any] | None = None
    ) -> FrontendBatchOutput:
        if data is None:
            raise ValueError(
                "ProvidedMaskFrontend expects raw input data with 'mask' or 'masks'"
            )

        if "mask" in data:
            instances = self._from_label_mask(data["mask"])
        elif "masks" in data:
            instances = self._from_binary_masks(data["masks"], data.get("obj_ids"))
        else:
            raise ValueError(
                "ProvidedMaskFrontend needs either 'mask' or 'masks' in input data"
            )

        image_shape = self._image_size(frame.rgb)
        return FrontendBatchOutput(instances=instances, image_size=image_shape)

    def _from_label_mask(self, mask) -> list[InstancePrediction]:
        label_mask = self._to_numpy(mask)
        instances: list[InstancePrediction] = []
        instance_id = 0
        for value in np.unique(label_mask):
            if int(value) in self.background_ids:
                continue
            binary_mask = label_mask == value
            if np.sum(binary_mask) < self.min_pixels:
                continue
            instances.append(
                InstancePrediction(
                    instance_id=instance_id,
                    mask=binary_mask.astype(np.uint8),
                    score=1.0,
                    class_id=None,
                    extra={"source_mask_id": int(value)},
                )
            )
            instance_id += 1
        return instances

    def _from_binary_masks(self, masks, obj_ids=None) -> list[InstancePrediction]:
        mask_tensor = (
            masks if isinstance(masks, torch.Tensor) else torch.as_tensor(masks)
        )
        obj_ids_array = None if obj_ids is None else np.asarray(obj_ids)
        instances: list[InstancePrediction] = []
        for idx in range(mask_tensor.shape[0]):
            binary_mask = mask_tensor[idx].detach().cpu().numpy().astype(np.uint8)
            if np.sum(binary_mask) < self.min_pixels:
                continue
            source_id = idx if obj_ids_array is None else int(obj_ids_array[idx])
            instances.append(
                InstancePrediction(
                    instance_id=idx,
                    mask=binary_mask,
                    score=1.0,
                    class_id=None,
                    extra={"source_mask_id": source_id},
                )
            )
        return instances

    @staticmethod
    def _image_size(rgb) -> tuple[int, int]:
        rgb_array = ProvidedMaskFrontend._to_numpy(rgb)
        if rgb_array.ndim == 3 and rgb_array.shape[0] == 3:
            return int(rgb_array.shape[1]), int(rgb_array.shape[2])
        return int(rgb_array.shape[0]), int(rgb_array.shape[1])

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)


def build_frontend_from_config(cfg) -> RGBDInstanceFrontend:
    frontend_mode = getattr(cfg, "frontend_mode", "legacy_mask")
    min_pixels = getattr(cfg, "frontend_min_pixels", 10)
    if frontend_mode in ["legacy_mask", "provided_mask", "external"]:
        return ProvidedMaskFrontend(min_pixels=min_pixels)
    if frontend_mode == "dinov2_scaffold":
        backbone = DinoV2Backbone(
            model_name=getattr(cfg, "frontend_model_name", "dinov2_vits14"),
            freeze=getattr(cfg, "freeze_frontend_backbone", False),
            device=cfg.device,
        )
        num_classes = getattr(cfg, "frontend_num_classes", 0)
        classification_head = None
        if num_classes > 0:
            classification_head = InstanceClassificationHead(
                backbone.embed_dim, num_classes
            ).to(cfg.device)
        instance_head = InstanceSegmentationHead(
            backbone.embed_dim, num_classes=num_classes
        ).to(cfg.device)
        frontend = DinoV2InstanceFrontend(
            backbone=backbone,
            classification_head=classification_head,
            instance_head=instance_head,
            min_pixels=min_pixels,
            img_size=getattr(cfg, "img_size", 224),
        )
        checkpoint_path = getattr(cfg, "frontend_checkpoint_path", None)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            frontend.load_state_dict(state_dict, strict=False)
        return frontend
    raise ValueError(f"Unsupported frontend_mode: {frontend_mode}")
