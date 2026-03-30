from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

WORKPIECE_CLASSES = {
    0: "盖板",       # cover_plate
    1: "方管",       # square_tube
    2: "喇叭口",     # flared_opening
    3: "H型钢",      # h_beam
    4: "坡口",       # bevel
    5: "槽钢",       # channel_steel
}


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_any(cls, meta: Any) -> "CameraIntrinsics":
        if isinstance(meta, cls):
            return meta
        if isinstance(meta, dict):
            intrinsics = meta["camera"]["intrinsics"]
            return cls(
                fx=float(intrinsics["fx"]),
                fy=float(intrinsics["fy"]),
                cx=float(intrinsics["cx"]),
                cy=float(intrinsics["cy"]),
                width=int(intrinsics["width"]),
                height=int(intrinsics["height"]),
            )

        camera_intrinsics = meta.camera.intrinsics
        return cls(
            fx=float(camera_intrinsics.fx),
            fy=float(camera_intrinsics.fy),
            cx=float(camera_intrinsics.cx),
            cy=float(camera_intrinsics.cy),
            width=int(camera_intrinsics.width),
            height=int(camera_intrinsics.height),
        )


@dataclass
class RGBDFrame:
    rgb: np.ndarray | torch.Tensor
    depth: np.ndarray | torch.Tensor
    intrinsics: CameraIntrinsics
    frame_id: str | None = None
    meta: dict[str, Any] | Any | None = None


@dataclass
class InstancePrediction:
    instance_id: int
    mask: np.ndarray | torch.Tensor
    score: float
    class_id: int | None = None
    class_name: str | None = None
    bbox_xyxy: np.ndarray | torch.Tensor | None = None
    rgb_feat: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrontendBatchOutput:
    instances: list[InstancePrediction]
    dense_feat_map: torch.Tensor | None = None
    global_feat: torch.Tensor | None = None
    image_size: tuple[int, int] | None = None


@dataclass
class PoseInstance:
    instance_id: int
    class_id: int | None
    score: float
    bbox_xyxy: np.ndarray
    mask: np.ndarray | torch.Tensor
    pcl_in: torch.Tensor
    pts: torch.Tensor
    pts_color: torch.Tensor
    zero_mean_pts: torch.Tensor
    pts_center: torch.Tensor
    roi_rgb: torch.Tensor
    roi_rgb_raw: torch.Tensor
    roi_mask: torch.Tensor
    roi_depth: torch.Tensor
    roi_xs: torch.Tensor
    roi_ys: torch.Tensor
    roi_center_dir: torch.Tensor
    rgb_feat: torch.Tensor | None = None
    pointwise_rgb_feat: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PoseBatch:
    pts: torch.Tensor
    pts_color: torch.Tensor
    pcl_in: torch.Tensor
    zero_mean_pts: torch.Tensor
    pts_center: torch.Tensor
    roi_rgb: torch.Tensor
    roi_xs: torch.Tensor
    roi_ys: torch.Tensor
    roi_center_dir: torch.Tensor
    roi_rgb_raw: torch.Tensor | None = None
    class_ids: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    instance_ids: list[int] | None = None
    rgb_feat: torch.Tensor | None = None
    pointwise_rgb_feat: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)
