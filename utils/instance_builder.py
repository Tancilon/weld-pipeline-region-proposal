from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from cutoop.transform import pixel2xyz

from datasets.types import FrontendBatchOutput, PoseBatch, PoseInstance, RGBDFrame
from datasets.datasets_omni6dpose import Omni6DPoseDataSet
from utils.datasets_utils import aug_bbox_eval, crop_resize_by_warp_affine
from utils.rgbd_to_pointcloud import (
    build_coord_map,
    build_scaled_intrinsic_matrix,
    compute_zero_mean,
    depth_to_pointcloud,
    sample_points,
)
from utils.sgpa_utils import get_bbox


class PoseInstanceBuilder:
    def __init__(
        self, img_size: int = 224, n_points: int = 1024, device: str = "cuda"
    ) -> None:
        self.img_size = img_size
        self.n_points = n_points
        self.device = device

    def build_batch(
        self, frame: RGBDFrame, frontend_output: FrontendBatchOutput
    ) -> PoseBatch:
        pose_instances: list[PoseInstance] = []
        for instance in frontend_output.instances:
            pose_instance = self.build_instance(frame, instance)
            if pose_instance is not None:
                pose_instances.append(pose_instance)

        if not pose_instances:
            raise ValueError(
                "No valid instances with usable depth were produced for pose inference"
            )

        return self._stack_instances(pose_instances)

    def build_instance(self, frame: RGBDFrame, instance) -> PoseInstance | None:
        rgb = self._to_numpy(frame.rgb)
        depth = self._to_numpy(frame.depth).astype(np.float32)
        mask = self._to_numpy(instance.mask).astype(bool)
        if not np.any(mask):
            return None

        depth = depth.copy()
        depth[depth > 4.0] = 0
        img_height, img_width = rgb.shape[:2]
        if depth.shape[:2] != (img_height, img_width):
            raise ValueError("RGB and depth shapes do not match")

        bbox_xyxy = self.extract_bbox_from_mask(mask)
        roi_tensors = self.build_roi_tensors(frame, mask, bbox_xyxy)
        roi_depth = roi_tensors["roi_depth"].cpu().numpy()
        roi_mask = roi_tensors["roi_mask"].cpu().numpy()
        depth_valid = roi_depth > 0
        if np.sum(depth_valid) <= 1.0:
            return None
        roi_mask_depth_valid = roi_mask.astype(np.bool_) * depth_valid
        if np.sum(roi_mask_depth_valid) <= 1.0:
            return None

        point_tensors = self.build_points_from_roi(
            roi_depth=roi_depth,
            roi_mask=roi_mask,
            roi_coord_2d=roi_tensors["roi_coord_2d"],
            frame=frame,
        )
        if point_tensors is None:
            return None

        return PoseInstance(
            instance_id=instance.instance_id,
            class_id=instance.class_id,
            score=float(instance.score),
            bbox_xyxy=bbox_xyxy,
            mask=instance.mask,
            pcl_in=point_tensors["pcl_in"],
            pts=point_tensors["pts"],
            pts_color=point_tensors["pts_color"],
            zero_mean_pts=point_tensors["zero_mean_pts"],
            pts_center=point_tensors["pts_center"],
            roi_rgb=roi_tensors["roi_rgb"],
            roi_rgb_raw=roi_tensors["roi_rgb_raw"],
            roi_mask=roi_tensors["roi_mask"],
            roi_depth=roi_tensors["roi_depth"],
            roi_xs=point_tensors["roi_xs"],
            roi_ys=point_tensors["roi_ys"],
            roi_center_dir=roi_tensors["roi_center_dir"],
            rgb_feat=self._get_global_rgb_feat(instance),
            pointwise_rgb_feat=self._get_pointwise_rgb_feat(instance, point_tensors),
            extra=dict(instance.extra),
        )

    def to_legacy_batch_dict(self, pose_batch: PoseBatch) -> dict[str, torch.Tensor]:
        data = {
            "pts": pose_batch.pts.to(self.device),
            "pts_color": pose_batch.pts_color.to(self.device),
            "pcl_in": pose_batch.pcl_in.to(self.device),
            "zero_mean_pts": pose_batch.zero_mean_pts.to(self.device),
            "pts_center": pose_batch.pts_center.to(self.device),
            "roi_rgb": pose_batch.roi_rgb.to(self.device),
            "roi_xs": pose_batch.roi_xs.to(self.device),
            "roi_ys": pose_batch.roi_ys.to(self.device),
            "roi_center_dir": pose_batch.roi_center_dir.to(self.device),
        }
        if pose_batch.rgb_feat is not None:
            data["rgb_feat"] = pose_batch.rgb_feat.to(self.device)
        if pose_batch.pointwise_rgb_feat is not None:
            data["pointwise_rgb_feat"] = pose_batch.pointwise_rgb_feat.to(self.device)
        return data

    def extract_bbox_from_mask(self, mask: np.ndarray) -> np.ndarray:
        ys, xs = np.argwhere(mask).transpose(1, 0)
        rmin, rmax, cmin, cmax = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
        rmin, rmax, cmin, cmax = get_bbox(
            [rmin, cmin, rmax, cmax], mask.shape[0], mask.shape[1]
        )
        return np.array([cmin, rmin, cmax, rmax], dtype=np.float32)

    def build_roi_tensors(
        self, frame: RGBDFrame, mask: np.ndarray, bbox_xyxy: np.ndarray
    ) -> dict[str, torch.Tensor]:
        rgb = self._to_numpy(frame.rgb)
        depth = self._to_numpy(frame.depth).astype(np.float32)
        intrinsics = frame.intrinsics
        img_height, img_width = rgb.shape[:2]
        bbox_center, scale = aug_bbox_eval(bbox_xyxy, img_height, img_width)
        coord_2d = build_coord_map(img_width, img_height)
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)
        roi_rgb_raw = crop_resize_by_warp_affine(
            rgb, bbox_center, scale, self.img_size, interpolation=cv2.INTER_LINEAR
        )
        roi_rgb = Omni6DPoseDataSet.rgb_transform(roi_rgb_raw)
        roi_mask = crop_resize_by_warp_affine(
            mask.astype(np.float32),
            bbox_center,
            scale,
            self.img_size,
            interpolation=cv2.INTER_NEAREST,
        )
        roi_depth = crop_resize_by_warp_affine(
            depth, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_center_dir = pixel2xyz(img_height, img_height, bbox_center, intrinsics)
        return {
            "roi_coord_2d": torch.as_tensor(
                np.ascontiguousarray(roi_coord_2d), dtype=torch.float32
            ),
            "roi_rgb": torch.as_tensor(
                np.ascontiguousarray(roi_rgb), dtype=torch.float32
            ).contiguous(),
            "roi_rgb_raw": torch.as_tensor(
                np.ascontiguousarray(roi_rgb_raw), dtype=torch.uint8
            ).contiguous(),
            "roi_mask": torch.as_tensor(
                np.expand_dims(roi_mask, axis=0), dtype=torch.float32
            ).contiguous(),
            "roi_depth": torch.as_tensor(
                np.expand_dims(roi_depth, axis=0), dtype=torch.float32
            ).contiguous(),
            "roi_center_dir": torch.as_tensor(
                roi_center_dir, dtype=torch.float32
            ).contiguous(),
        }

    def build_points_from_roi(
        self, roi_depth, roi_mask, roi_coord_2d, frame: RGBDFrame
    ) -> dict[str, torch.Tensor] | None:
        valid = (np.squeeze(roi_depth, axis=0) > 0) * (np.squeeze(roi_mask, axis=0) > 0)
        indices = np.argwhere(valid)
        if len(indices) < 10:
            return None

        xs, ys = indices.transpose(1, 0)
        intrinsic_matrix = build_scaled_intrinsic_matrix(
            frame.intrinsics,
            self._to_numpy(frame.rgb).shape[1],
            self._to_numpy(frame.rgb).shape[0],
        )
        pcl_in = depth_to_pointcloud(
            roi_depth,
            intrinsic_matrix,
            roi_coord_2d.permute(1, 2, 0).cpu().numpy(),
            valid,
        )
        ids, pcl_in = sample_points(pcl_in, self.n_points)
        xs = torch.as_tensor(xs, dtype=torch.int64)[ids]
        ys = torch.as_tensor(ys, dtype=torch.int64)[ids]
        zero_mean_pts, pts_center = compute_zero_mean(pcl_in)
        return {
            "pcl_in": pcl_in.contiguous(),
            "pts": pcl_in.contiguous(),
            "pts_color": pcl_in.contiguous(),
            "zero_mean_pts": zero_mean_pts.contiguous(),
            "pts_center": pts_center.contiguous(),
            "roi_xs": xs.contiguous(),
            "roi_ys": ys.contiguous(),
        }

    def _stack_instances(self, pose_instances: list[PoseInstance]) -> PoseBatch:
        rgb_feats = [
            item.rgb_feat for item in pose_instances if item.rgb_feat is not None
        ]
        pointwise_rgb_feats = [
            item.pointwise_rgb_feat
            for item in pose_instances
            if item.pointwise_rgb_feat is not None
        ]
        rgb_feat = None
        if len(rgb_feats) == len(pose_instances):
            rgb_feat = torch.stack(rgb_feats, dim=0)
        pointwise_rgb_feat = None
        if len(pointwise_rgb_feats) == len(pose_instances):
            pointwise_rgb_feat = torch.stack(pointwise_rgb_feats, dim=0)

        return PoseBatch(
            pts=torch.stack([item.pts for item in pose_instances], dim=0),
            pts_color=torch.stack([item.pts_color for item in pose_instances], dim=0),
            pcl_in=torch.stack([item.pcl_in for item in pose_instances], dim=0),
            zero_mean_pts=torch.stack(
                [item.zero_mean_pts for item in pose_instances], dim=0
            ),
            pts_center=torch.stack([item.pts_center for item in pose_instances], dim=0),
            roi_rgb=torch.stack([item.roi_rgb for item in pose_instances], dim=0),
            roi_xs=torch.stack([item.roi_xs for item in pose_instances], dim=0),
            roi_ys=torch.stack([item.roi_ys for item in pose_instances], dim=0),
            roi_center_dir=torch.stack(
                [item.roi_center_dir for item in pose_instances], dim=0
            ),
            roi_rgb_raw=torch.stack(
                [item.roi_rgb_raw for item in pose_instances], dim=0
            ),
            class_ids=torch.as_tensor(
                [
                    (-1 if item.class_id is None else item.class_id)
                    for item in pose_instances
                ],
                dtype=torch.int64,
            ),
            scores=torch.as_tensor(
                [item.score for item in pose_instances], dtype=torch.float32
            ),
            instance_ids=[item.instance_id for item in pose_instances],
            rgb_feat=rgb_feat,
            pointwise_rgb_feat=pointwise_rgb_feat,
            extra={
                "source_mask_ids": [
                    item.extra.get("source_mask_id", item.instance_id)
                    for item in pose_instances
                ],
            },
        )

    def _get_global_rgb_feat(self, instance) -> torch.Tensor | None:
        if instance.rgb_feat is not None:
            return instance.rgb_feat
        roi_global_feat = instance.extra.get("roi_global_feat")
        if roi_global_feat is None:
            return None
        return torch.as_tensor(roi_global_feat, dtype=torch.float32)

    def _get_pointwise_rgb_feat(
        self, instance, point_tensors: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        patch_tokens = instance.extra.get("roi_patch_tokens")
        if patch_tokens is None:
            return None
        patch_tokens = torch.as_tensor(patch_tokens, dtype=torch.float32)
        if patch_tokens.ndim == 3:
            patch_tokens = patch_tokens.squeeze(0)
        token_grid = self.img_size // 14
        xs = point_tensors["roi_xs"] // 14
        ys = point_tensors["roi_ys"] // 14
        pos = xs * token_grid + ys
        return patch_tokens[pos]

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)
