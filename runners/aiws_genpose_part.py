from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


class GenPosePartError(RuntimeError):
    pass


class GenPosePartEstimator:
    def __init__(
        self,
        seg_ckpt: str,
        energy_ckpt: str | None,
        scale_ckpt: str | None,
        device: str = "cuda",
        repeat_num: int = 10,
        num_points: int = 1024,
        img_size: int = 224,
    ):
        self.seg_ckpt = seg_ckpt
        self.energy_ckpt = energy_ckpt
        self.scale_ckpt = scale_ckpt
        self.device = device
        self.repeat_num = repeat_num
        self.num_points = num_points
        self.img_size = img_size
        self._runtime_bundle = None

    def _load_runtime_bundle(self):
        if self._runtime_bundle is not None:
            return self._runtime_bundle
        original_argv = list(sys.argv)
        try:
            sys.argv = sys.argv[:1]
            import cv2
            from datasets.datasets_omni6dpose import Omni6DPoseDataSet
            from runners.infer_nuclear_full_lib import (
                infer_pose_and_size,
                init_pipeline_agents,
            )
            from utils.datasets_utils import (
                aug_bbox_eval,
                crop_resize_by_warp_affine,
                get_2d_coord_np,
            )

            args = SimpleNamespace(
                seg_ckpt=self.seg_ckpt,
                energy_ckpt=self.energy_ckpt,
                scale_ckpt=self.scale_ckpt,
                repeat_num=self.repeat_num,
                num_points=self.num_points,
                img_size=self.img_size,
                device=self.device,
            )
            main_cfg, main_agent, energy_agent, scale_agent = init_pipeline_agents(args)
        finally:
            sys.argv = original_argv
        self._runtime_bundle = SimpleNamespace(
            cv2=cv2,
            torch=torch,
            Omni6DPoseDataSet=Omni6DPoseDataSet,
            aug_bbox_eval=aug_bbox_eval,
            crop_resize_by_warp_affine=crop_resize_by_warp_affine,
            get_2d_coord_np=get_2d_coord_np,
            main_cfg=main_cfg,
            main_agent=main_agent,
            energy_agent=energy_agent,
            scale_agent=scale_agent,
            infer_pose_and_size=infer_pose_and_size,
        )
        return self._runtime_bundle

    @staticmethod
    def _intrinsics_to_matrix(intrinsics: dict[str, float]) -> np.ndarray:
        return np.asarray(
            [
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _load_mask(path: str | Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("L")) > 0

    def _load_rgb(self, path: str | Path) -> np.ndarray:
        runtime = self._load_runtime_bundle()
        rgb = runtime.cv2.imread(str(path), runtime.cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(f"RGB image not found: {path}")
        return runtime.cv2.cvtColor(rgb, runtime.cv2.COLOR_BGR2RGB)

    def _extract_pointcloud(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        K_33: np.ndarray,
    ) -> dict:
        runtime = self._load_runtime_bundle()
        image_height, image_width = rgb.shape[:2]
        ys_mask, xs_mask = np.where(mask)
        if len(ys_mask) == 0:
            raise GenPosePartError("selected part mask is empty")
        bbox_xyxy = np.array(
            [xs_mask.min(), ys_mask.min(), xs_mask.max(), ys_mask.max()],
            dtype=np.float32,
        )
        bbox_center, scale = runtime.aug_bbox_eval(bbox_xyxy, image_height, image_width)
        coord_2d = runtime.get_2d_coord_np(image_width, image_height).transpose(1, 2, 0)
        roi_rgb_np = runtime.crop_resize_by_warp_affine(
            rgb,
            bbox_center,
            scale,
            self.img_size,
            interpolation=runtime.cv2.INTER_LINEAR,
        )
        roi_depth = runtime.crop_resize_by_warp_affine(
            depth,
            bbox_center,
            scale,
            self.img_size,
            interpolation=runtime.cv2.INTER_NEAREST,
        )
        roi_mask = runtime.crop_resize_by_warp_affine(
            mask.astype(np.float32),
            bbox_center,
            scale,
            self.img_size,
            interpolation=runtime.cv2.INTER_NEAREST,
        )
        roi_coord_2d = runtime.crop_resize_by_warp_affine(
            coord_2d,
            bbox_center,
            scale,
            self.img_size,
            interpolation=runtime.cv2.INTER_NEAREST,
        ).transpose(2, 0, 1)
        valid_2d = (roi_depth > 0) & (roi_mask > 0.5)
        if int(valid_2d.sum()) < 10:
            raise GenPosePartError("insufficient valid depth in selected part mask")
        xs_crop, ys_crop = np.argwhere(valid_2d).T
        valid_flat = valid_2d.reshape(-1)
        pcl = runtime.Omni6DPoseDataSet.depth_to_pcl(
            roi_depth, K_33, roi_coord_2d, valid_flat
        )
        if len(pcl) < 10:
            raise GenPosePartError("insufficient point cloud points in selected part mask")
        ids, pcl = runtime.Omni6DPoseDataSet.sample_points(pcl, self.num_points)
        pts = torch.as_tensor(pcl, dtype=torch.float32)
        pts_center = pts[:, :3].mean(dim=0)
        zero_mean_pts = pts.clone()
        zero_mean_pts[:, :3] -= pts_center
        return {
            "pts": pts,
            "pcl_in": pts.clone(),
            "roi_rgb": torch.as_tensor(
                runtime.Omni6DPoseDataSet.rgb_transform(roi_rgb_np), dtype=torch.float32
            ),
            "roi_xs": torch.as_tensor(xs_crop[ids], dtype=torch.int64),
            "roi_ys": torch.as_tensor(ys_crop[ids], dtype=torch.int64),
            "pts_center": pts_center,
            "zero_mean_pts": zero_mean_pts,
        }

    def estimate_part(
        self,
        rgb_path: str | Path,
        depth: np.ndarray,
        mask_path: str | Path,
        intrinsics: dict[str, float],
    ) -> dict:
        runtime = self._load_runtime_bundle()
        rgb = self._load_rgb(rgb_path)
        mask = self._load_mask(mask_path)
        if mask.shape != depth.shape:
            raise GenPosePartError(f"mask/depth shape mismatch: {mask.shape} vs {depth.shape}")
        pt_data = self._extract_pointcloud(
            rgb=rgb,
            depth=np.asarray(depth, dtype=np.float32),
            mask=mask,
            K_33=self._intrinsics_to_matrix(intrinsics),
        )
        result = runtime.infer_pose_and_size(
            main_agent=runtime.main_agent,
            energy_agent=runtime.energy_agent,
            scale_agent=runtime.scale_agent,
            cfg=runtime.main_cfg,
            pt_data=pt_data,
            device=runtime.main_cfg.device,
            repeat_num=self.repeat_num,
        )
        if result is None:
            raise GenPosePartError("GenPose++ returned no result")
        return result
