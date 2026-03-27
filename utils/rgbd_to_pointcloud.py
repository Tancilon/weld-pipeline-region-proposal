from __future__ import annotations

import numpy as np
import torch

from utils.datasets_utils import get_2d_coord_np


def build_coord_map(width: int, height: int) -> np.ndarray:
    return get_2d_coord_np(width, height).transpose(1, 2, 0)


def build_scaled_intrinsic_matrix(
    intrinsics, img_width: int, img_height: int
) -> np.ndarray:
    intrinsic_matrix = np.array(
        [
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )
    intrinsic_matrix[0] *= img_width / intrinsics.width
    intrinsic_matrix[1] *= img_height / intrinsics.height
    return intrinsic_matrix


def depth_to_pointcloud(depth, intrinsic_matrix, coord_map, valid_mask) -> torch.Tensor:
    depth_tensor = torch.as_tensor(depth.reshape(-1), dtype=torch.float32)
    valid_tensor = torch.as_tensor(valid_mask.reshape(-1), dtype=torch.bool)
    coord_tensor = torch.as_tensor(coord_map, dtype=torch.float32)
    intrinsic_tensor = torch.as_tensor(
        intrinsic_matrix.reshape(-1), dtype=torch.float32
    )

    cx, cy, fx, fy = (
        intrinsic_tensor[2],
        intrinsic_tensor[5],
        intrinsic_tensor[0],
        intrinsic_tensor[4],
    )
    depth_values = depth_tensor[valid_tensor]
    x_map = coord_tensor[:, :, 0].reshape(-1)[valid_tensor]
    y_map = coord_tensor[:, :, 1].reshape(-1)[valid_tensor]
    real_x = (x_map - cx) * depth_values / fx
    real_y = (y_map - cy) * depth_values / fy
    return torch.stack((real_x, real_y, depth_values), dim=-1).to(torch.float32)


def sample_points(
    points: torch.Tensor, n_points: int
) -> tuple[torch.Tensor, torch.Tensor]:
    total_points = points.shape[0]
    if total_points < n_points:
        repeat_count = n_points // total_points
        tail_count = n_points % total_points
        sampled = torch.cat(
            [points.repeat((repeat_count, 1)), points[:tail_count]], dim=0
        )
        ids = torch.cat(
            [
                torch.arange(total_points, dtype=torch.long).repeat(repeat_count),
                torch.arange(tail_count, dtype=torch.long),
            ],
            dim=0,
        )
        return ids, sampled

    ids = torch.randperm(total_points, dtype=torch.long)[:n_points]
    return ids, points[ids]


def compute_zero_mean(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if points.ndim == 2:
        center = torch.mean(points[:, :3], dim=0)
        zero_mean = points.clone()
        zero_mean[:, :3] -= center
        return zero_mean, center

    center = torch.mean(points[:, :, :3], dim=1)
    zero_mean = points.clone()
    zero_mean[:, :, :3] -= center.unsqueeze(1)
    return zero_mean, center
