"""
Full pipeline inference for nuclear workpieces:
RGB + Depth -> Segmentation -> 6D Pose (+ optional EnergyNet / ScaleNet).

Architecture note:
  - Segmentation runs in a dedicated SegNet-style agent with query injection.
  - 6D pose runs in a separate pure GenPose2 score agent.
  - EnergyNet / ScaleNet remain pure GenPose2 agents.
This file intentionally avoids the older "segmentation + pose hybrid backbone"
path because it degraded pose quality.

Usage:
    python runners/infer_nuclear_full.py \
        --nuclear_data_path ./data/aiws5.2_nuclear_workpieces \
        --seg_ckpt_path     ./results/ckpts/SegNet/ckpt_epoch100.pth \
        --pose_ckpt_path    ./results/ckpts/ScoreNet/scorenet.pth \
        --energy_ckpt_path  ./results/ckpts/EnergyNet/energynet.pth \
        --scale_ckpt_path   ./results/ckpts/ScaleNet/scalenet.pth \
        --split val \
        --output_dir ./results/full_pipeline \
        --num_vis 5 \
        --score_threshold 0.5

If --pose_ckpt_path is omitted the pose branch uses whatever weights are already
in the seg checkpoint (e.g. if seg training was started from a pretrained pose model).
If --energy_ckpt_path is omitted pose hypotheses are aggregated directly from
the score model outputs. If --scale_ckpt_path is omitted object size is estimated
geometrically from the segmented point cloud and aggregated pose.
"""

import sys
import os
import argparse
import copy

# --------------------------------------------------------------------------- #
# Parse args FIRST — downstream imports call get_config() at module level and
# will choke on our custom flags if they reach parse_args() first.
# --------------------------------------------------------------------------- #
_p = argparse.ArgumentParser(description="Nuclear workpiece full pipeline inference")
_p.add_argument("--nuclear_data_path", type=str, required=True)
_p.add_argument("--seg_ckpt_path",     type=str, required=True)
_p.add_argument("--pose_ckpt_path",    type=str, default=None,
                help="Optional pretrained GenPose2 pose checkpoint")
_p.add_argument("--energy_ckpt_path",  type=str, default=None,
                help="Optional EnergyNet checkpoint for pose hypothesis ranking")
_p.add_argument("--scale_ckpt_path",   type=str, default=None,
                help="Optional ScaleNet checkpoint for object size prediction")
_p.add_argument("--split",        type=str, default="val", choices=["train", "val"])
_p.add_argument("--output_dir",   type=str, default="./results/full_pipeline")
_p.add_argument("--num_vis",      type=int, default=5)
_p.add_argument("--score_threshold", type=float, default=0.5)
_p.add_argument("--repeat_num",   type=int, default=10,
                help="Number of pose hypothesis samples (more = more accurate but slower)")
_p.add_argument("--num_points",   type=int, default=1024)
_p.add_argument("--img_size",     type=int, default=224)
_p.add_argument("--device",       type=str, default="cuda")
_args = _p.parse_args()

sys.argv = sys.argv[:1]  # clear argv before heavy imports trigger get_config()

# --------------------------------------------------------------------------- #
import json
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.datasets_nuclear import (
    CLASS_NAMES, NUM_CLASSES, collate_nuclear, process_batch_seg,
    NuclearWorkpieceDataset,
)
from datasets.datasets_omni6dpose import Omni6DPoseDataSet
from cutoop.data_types import CameraIntrinsicsBase
from cutoop.eval_utils import DetectMatch
from networks.reward import sort_poses_by_energy
from utils.datasets_utils import aug_bbox_eval, crop_resize_by_warp_affine, get_2d_coord_np
from utils.misc import average_quaternion_batch
from utils.metrics import get_rot_matrix
from utils.transforms import matrix_to_quaternion, quaternion_to_matrix
from networks.posenet_agent import PoseNet


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

COLORS_BGR = [
    (0,   0, 255),   # 盖板   – red
    (0, 255,   0),   # 方管   – green
    (255, 0,   0),   # 喇叭口 – blue
    (0, 255, 255),   # H型钢  – yellow
    (255, 0, 255),   # 槽钢   – magenta
    (255, 255, 0),   # 坡口   – cyan
]

# Axis colours (BGR): X=red, Y=green, Z=blue
AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]


# --------------------------------------------------------------------------- #
# PIL text helper (CJK-capable)
# --------------------------------------------------------------------------- #

_FONT = None

def _get_font(size=20):
    global _FONT
    if _FONT is not None:
        return _FONT
    candidates = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
        '/System/Library/Fonts/PingFang.ttc',
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                _FONT = ImageFont.truetype(p, size)
                return _FONT
            except Exception:
                continue
    _FONT = ImageFont.load_default()
    return _FONT

def put_text(img_bgr, text, x, y, text_color=(255, 255, 255),
             bg_color=None, font_size=20):
    font = _get_font(font_size)
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    tc_rgb = text_color[::-1]
    if bg_color is not None:
        bb = draw.textbbox((x, y), text, font=font)
        draw.rectangle([bb[0]-2, bb[1]-2, bb[2]+2, bb[3]+2], fill=bg_color[::-1])
    draw.text((x, y), text, font=font, fill=tc_rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #

def load_segmentation_agent_checkpoint(agent, seg_ckpt_path):
    """Load the SegNet-style checkpoint used only for mask prediction."""
    print(f"Loading seg checkpoint: {seg_ckpt_path}")
    seg_ckpt = torch.load(seg_ckpt_path, map_location="cpu")
    missing, unexpected = agent.net.load_state_dict(
        seg_ckpt["model_state_dict"], strict=False
    )
    if missing:
        print(f"  [seg ckpt] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  [seg ckpt] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")


def load_model_only(agent, ckpt_path, name):
    """Load a regular GenPose2 checkpoint into a single agent."""
    print(f"Loading {name} checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = agent.net.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if missing:
        print(f"  [{name} ckpt] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  [{name} ckpt] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")


def load_pure_pose_score_checkpoint(agent, pose_ckpt_path=None, fallback_seg_ckpt_path=None):
    """Load the pure GenPose2 score agent used for downstream 6D pose.

    Preferred source is the original pose checkpoint. If it is unavailable,
    optionally fall back to the segmentation checkpoint to preserve the old
    "pose weights embedded in seg ckpt" behaviour.
    """
    if pose_ckpt_path is not None:
        load_model_only(agent, pose_ckpt_path, "pose")
        return
    if fallback_seg_ckpt_path is not None:
        print("Pose checkpoint not provided; falling back to seg checkpoint for score weights.")
        load_model_only(agent, fallback_seg_ckpt_path, "seg-as-pose")
        return
    raise ValueError("Either pose_ckpt_path or fallback_seg_ckpt_path must be provided")


# --------------------------------------------------------------------------- #
# Segmentation helpers
# --------------------------------------------------------------------------- #

def postprocess_seg(class_logits, mask_logits, score_threshold):
    """Return only the highest-confidence prediction above threshold."""
    probs = class_logits.softmax(dim=-1)        # [N, C+1]
    obj_probs = probs[:, :-1]                   # [N, C]
    max_scores, max_classes = obj_probs.max(dim=-1)

    best = None
    for i in range(len(max_scores)):
        s = max_scores[i].item()
        if s < score_threshold:
            continue
        cls_id = max_classes[i].item()
        mask = mask_logits[i].sigmoid().cpu().numpy()
        if best is None or s > best[2]:
            best = (mask, cls_id, s)

    if best is None:
        return [], [], []
    mask, cls_id, score = best
    return [mask], [cls_id], [score]


# --------------------------------------------------------------------------- #
# Point cloud extraction
# --------------------------------------------------------------------------- #

def extract_pointcloud(rgb_orig, depth_orig, mask_orig, K_33, img_size, num_points):
    """Crop region, extract point cloud, return all pose-estimation inputs.

    Args:
        rgb_orig:   [H, W, 3] uint8 BGR.
        depth_orig: [H, W] float32, metric depth (metres).
        mask_orig:  [H, W] bool.
        K_33:       [3, 3] camera intrinsic matrix.
        img_size:   crop target size (e.g. 224).
        num_points: number of points to sample.

    Returns dict with keys: pts, roi_rgb, roi_xs, roi_ys, pts_center,
    zero_mean_pts, bbox_center — or None if not enough valid depth.
    """
    im_H, im_W = rgb_orig.shape[:2]

    # --- bounding box from mask ---
    ys_m, xs_m = np.where(mask_orig)
    if len(ys_m) == 0:
        return None
    bbox_xyxy = np.array([xs_m.min(), ys_m.min(), xs_m.max(), ys_m.max()],
                         dtype=np.float32)
    bbox_center, scale = aug_bbox_eval(bbox_xyxy, im_H, im_W)

    # --- crop RGB, depth, mask, and the original-pixel coord map ---
    # coord_2d[0] = col-pixel, coord_2d[1] = row-pixel (original image coords)
    coord_2d_hw2 = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0)  # [H, W, 2]

    roi_rgb_np = crop_resize_by_warp_affine(
        rgb_orig, bbox_center, scale, img_size, interpolation=cv2.INTER_LINEAR)
    roi_depth = crop_resize_by_warp_affine(
        depth_orig, bbox_center, scale, img_size, interpolation=cv2.INTER_NEAREST)
    roi_mask = crop_resize_by_warp_affine(
        mask_orig.astype(np.float32), bbox_center, scale, img_size,
        interpolation=cv2.INTER_NEAREST)
    # coord map: preserves original pixel positions through affine crop
    roi_coord_2d = crop_resize_by_warp_affine(
        coord_2d_hw2, bbox_center, scale, img_size,
        interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)  # [2, 224, 224]

    # --- valid pixels in the crop ---
    valid_2d = (roi_depth > 0) & (roi_mask > 0.5)
    if valid_2d.sum() < 10:
        return None
    # NOTE: np.argwhere returns (row, col); xs=rows, ys=cols (GenPose2 convention)
    xs_crop, ys_crop = np.argwhere(valid_2d).T   # xs=row coords, ys=col coords
    valid_flat = valid_2d.reshape(-1)

    # --- backproject depth to 3D (using original-image coords + original K) ---
    pcl = Omni6DPoseDataSet.depth_to_pcl(roi_depth, K_33, roi_coord_2d, valid_flat)
    if len(pcl) < 10:
        return None

    ids, pcl = Omni6DPoseDataSet.sample_points(pcl, num_points)
    xs_s = xs_crop[ids]   # sampled row coords in [0, img_size)
    ys_s = ys_crop[ids]   # sampled col coords in [0, img_size)

    # --- ImageNet-normalise RGB crop ---
    roi_rgb_tensor = torch.tensor(
        Omni6DPoseDataSet.rgb_transform(
            cv2.cvtColor(roi_rgb_np, cv2.COLOR_BGR2RGB)
        ), dtype=torch.float32
    )  # [3, 224, 224]

    pts = torch.tensor(pcl, dtype=torch.float32)          # [N, 3]
    pts_center = pts[:, :3].mean(dim=0)                   # [3]
    zero_mean_pts = pts.clone()
    zero_mean_pts[:, :3] -= pts_center

    return {
        "pts":            pts,                            # [N, 3]
        "pcl_in":         pts.clone(),                    # [N, 3]
        "roi_rgb":        roi_rgb_tensor,                 # [3, H, H]
        "roi_xs":         torch.tensor(xs_s, dtype=torch.long),   # [N]
        "roi_ys":         torch.tensor(ys_s, dtype=torch.long),   # [N]
        "pts_center":     pts_center,                    # [3]
        "zero_mean_pts":  zero_mean_pts,                 # [N, 3]
        "roi_rgb_np":     roi_rgb_np,                    # for visualisation
        "bbox_center":    bbox_center,
    }


# --------------------------------------------------------------------------- #
# Pose and size inference
# --------------------------------------------------------------------------- #

def build_instance_batch(pt_data, device):
    """Convert per-instance point-cloud data into a batched network input."""
    return {
        "pts":           pt_data["pts"].unsqueeze(0).to(device),
        "pcl_in":        pt_data["pcl_in"].unsqueeze(0).to(device),
        "roi_rgb":       pt_data["roi_rgb"].unsqueeze(0).to(device),
        "roi_xs":        pt_data["roi_xs"].unsqueeze(0).to(device),
        "roi_ys":        pt_data["roi_ys"].unsqueeze(0).to(device),
        "pts_center":    pt_data["pts_center"].unsqueeze(0).to(device),
        "zero_mean_pts": pt_data["zero_mean_pts"].unsqueeze(0).to(device),
    }


def aggregate_pose(cfg, pred_pose, pred_energy=None):
    """Aggregate pose hypotheses into a single SE(3) pose."""
    bs, repeat_num, _ = pred_pose.shape
    if pred_energy is None:
        good_pose = pred_pose
    else:
        sorted_pose, _ = sort_poses_by_energy(pred_pose, pred_energy)
        retain_num = max(1, int(round(repeat_num * cfg.retain_ratio)))
        good_pose = sorted_pose[:, :retain_num, :]

    retain_num = good_pose.shape[1]
    rot_matrix = get_rot_matrix(
        good_pose[:, :, :-3].reshape(bs * retain_num, -1),
        cfg.pose_mode,
    )
    quat_wxyz = matrix_to_quaternion(rot_matrix).reshape(bs, retain_num, -1)
    aggregated_quat_wxyz = average_quaternion_batch(quat_wxyz)

    if getattr(cfg, "clustering", 0):
        min_samples = max(1, int(round(cfg.clustering_minpts * retain_num)))
        for j in range(bs):
            pairwise_distance = 1 - torch.sum(
                quat_wxyz[j].unsqueeze(0) * quat_wxyz[j].unsqueeze(1),
                dim=2,
            ) ** 2
            dbscan = DBSCAN(
                eps=cfg.clustering_eps,
                min_samples=min_samples,
            ).fit(pairwise_distance.cpu().numpy())
            labels = dbscan.labels_
            if np.any(labels >= 0):
                bins = np.bincount(labels[labels >= 0])
                best_label = np.argmax(bins)
                aggregated_quat_wxyz[j] = average_quaternion_batch(
                    quat_wxyz[j, labels == best_label].unsqueeze(0)
                )[0]

    aggregated_trans = torch.mean(good_pose[:, :, -3:], dim=1)
    aggregated_pose = torch.zeros(bs, 4, 4, device=pred_pose.device)
    aggregated_pose[:, 3, 3] = 1.0
    aggregated_pose[:, :3, :3] = quaternion_to_matrix(aggregated_quat_wxyz)
    aggregated_pose[:, :3, 3] = aggregated_trans
    return aggregated_pose


def estimate_size_from_geometry(points, pose):
    """Estimate axis-aligned object size in object frame from segmented points."""
    rotation = pose[:, :3, :3]
    translation = pose[:, :3, 3]
    obj_points = points - translation.unsqueeze(1)
    obj_points = torch.bmm(rotation.transpose(1, 2), obj_points.transpose(1, 2))
    obj_points = obj_points.transpose(1, 2)
    bbox_length, _ = torch.max(torch.abs(obj_points), dim=1)
    return bbox_length * 2.0


def infer_pose_and_size(score_agent, energy_agent, scale_agent, cfg, pt_data,
                        device, repeat_num):
    """Run score -> optional energy -> optional scale on one segmented instance."""
    data = build_instance_batch(pt_data, device)

    pred_pose, _ = score_agent.pred_func(data, repeat_num=repeat_num)
    pred_energy = None
    if energy_agent is not None:
        pred_energy = energy_agent.get_energy(
            data=data,
            pose_samples=pred_pose,
            T=None,
            mode="test",
            extract_feature=True,
        )

    aggregated_pose = aggregate_pose(cfg, pred_pose, pred_energy)
    final_pose = aggregated_pose.clone()

    if scale_agent is not None:
        scale_input = {
            "pts_feat": data["pts_feat"],
            "rgb_feat": data["rgb_feat"],
            "axes": aggregated_pose[:, :3, :3],
        }
        cal_mat, pred_size = scale_agent.pred_scale_func(scale_input)
        final_pose[:, :3, :3] = cal_mat
        size_source = "scale_net"
    else:
        pred_size = estimate_size_from_geometry(data["pcl_in"], aggregated_pose)
        size_source = "geometry"

    return {
        "R": final_pose[0, :3, :3].detach().cpu().numpy(),
        "t": final_pose[0, :3, 3].detach().cpu().numpy(),
        "size": pred_size[0].detach().cpu().numpy(),
        "size_source": size_source,
    }


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #

def draw_3d_bbox_and_axes(img_bgr, R, t, size, camera_intrinsics, axes_length=0.1):
    """Draw original GenPose2-style 3D bbox and pose axes."""
    affine = np.eye(4, dtype=np.float32)
    affine[:3, :3] = R.astype(np.float32)
    affine[:3, 3] = t.astype(np.float32)
    size = np.asarray(size, dtype=np.float32)
    return DetectMatch._draw_image(
        vis_img=img_bgr,
        pred_affine=affine,
        pred_size=size,
        gt_affine=None,
        gt_size=None,
        gt_sym_label=None,
        camera_intrinsics=camera_intrinsics,
        draw_pred=True,
        draw_gt=False,
        draw_label=False,
        draw_pred_axes_length=axes_length,
        draw_gt_axes_length=None,
        thickness=True,
    )


def visualize_result(rgb_orig, instances, camera_intrinsics, img_size):
    """Draw masks + original GenPose2-style 3D bbox/axes on the RGB image.

    instances: list of dicts with keys cls_id, score, mask_orig, R, t, size.
    """
    vis = rgb_orig.copy()
    im_H, im_W = vis.shape[:2]
    for inst in instances:
        cls_id = inst["cls_id"]
        color  = COLORS_BGR[cls_id % len(COLORS_BGR)]
        mask   = inst["mask_orig"]       # [H, W] bool

        # Colour fill
        overlay = vis.copy()
        overlay[mask] = color
        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)

        # Contour
        binary = mask.astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)

        # Label (class + score) near centroid
        if len(contours) > 0:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = contours[0][0][0]
            label = f"{CLASS_NAMES[cls_id]} {inst['score']:.2f}"
            vis = put_text(vis, label, cx, max(cy - 25, 0),
                           text_color=(255, 255, 255), bg_color=color)

        # Original GenPose2-style 3D bbox + axes
        if (
            inst.get("R") is not None
            and inst.get("t") is not None
            and inst.get("size") is not None
        ):
            vis = draw_3d_bbox_and_axes(
                vis,
                inst["R"],
                inst["t"],
                inst["size"],
                camera_intrinsics,
                axes_length=0.1,
            )

    return vis


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_cfg(args, agent_type="score", enable_segmentation=False):
    class Cfg:
        pass
    cfg = Cfg()
    cfg.device          = args.device if torch.cuda.is_available() else "cpu"
    cfg.dino            = "pointwise"
    cfg.pts_encoder     = "pointnet2"
    cfg.agent_type      = agent_type
    cfg.pose_mode       = "rot_matrix"
    cfg.regression_head = "Rx_Ry_and_T"
    cfg.sde_mode        = "ve"
    cfg.num_points      = args.num_points
    cfg.img_size        = args.img_size
    cfg.pointnet2_params = "light"
    cfg.parallel        = False
    cfg.is_train        = False
    cfg.eval            = False
    cfg.pred            = False
    cfg.use_pretrain    = False
    cfg.log_dir         = "infer_nuclear"
    cfg.ema_rate        = 0.999
    cfg.lr              = 1e-4
    cfg.lr_decay        = 0.99
    cfg.optimizer       = "Adam"
    cfg.warmup          = 50
    cfg.grad_clip       = 1.0
    cfg.sampling_steps  = 500
    cfg.sampler_mode    = ["ode"]
    cfg.energy_mode     = "IP"
    cfg.s_theta_mode    = "score"
    cfg.norm_energy     = "identical"
    cfg.scale_embedding = 180
    cfg.eval_repeat_num = 50
    cfg.repeat_num      = args.repeat_num
    cfg.num_gpu         = 1
    cfg.scale_batch_size = 64
    cfg.save_video            = False
    cfg.retain_ratio          = 0.4
    cfg.clustering            = 1
    cfg.clustering_eps        = 0.05
    cfg.clustering_minpts     = 0.1667
    cfg.enable_segmentation   = enable_segmentation
    cfg.num_queries           = 50
    cfg.query_inject_layer    = -4
    cfg.num_object_classes    = 6
    cfg.unfreeze_dino_last_n  = 4
    cfg.seg_loss_weight       = 1.0
    cfg.cls_loss_weight       = 2.0
    return cfg


def init_pipeline_agents(args):
    """Build the pipeline as separate segmentation and pure GenPose2 agents."""
    seg_cfg = build_cfg(args, agent_type="score", enable_segmentation=True)
    print("Building segmentation agent (SegNet path)...")
    seg_agent = PoseNet(seg_cfg)
    load_segmentation_agent_checkpoint(seg_agent, args.seg_ckpt_path)
    seg_agent.net.eval()

    score_cfg = build_cfg(args, agent_type="score", enable_segmentation=False)
    print("Building pure GenPose2 score agent (no segmentation wrapper)...")
    score_agent = PoseNet(score_cfg)
    load_pure_pose_score_checkpoint(
        score_agent,
        pose_ckpt_path=args.pose_ckpt_path,
        fallback_seg_ckpt_path=args.seg_ckpt_path,
    )
    score_agent.net.eval()

    energy_agent = None
    if args.energy_ckpt_path:
        energy_cfg = build_cfg(args, agent_type="energy", enable_segmentation=False)
        print("Building pure GenPose2 energy agent...")
        energy_agent = PoseNet(energy_cfg)
        load_model_only(energy_agent, args.energy_ckpt_path, "energy")
        energy_agent.net.eval()

    scale_agent = None
    if args.scale_ckpt_path:
        scale_cfg = build_cfg(args, agent_type="scale", enable_segmentation=False)
        print("Building pure GenPose2 scale agent...")
        scale_agent = PoseNet(scale_cfg)
        load_model_only(scale_agent, args.scale_ckpt_path, "scale")
        scale_agent.net.eval()

    return seg_cfg, score_cfg, seg_agent, score_agent, energy_agent, scale_agent


def read_meta(meta_path):
    """Return camera intrinsics in both matrix and object forms."""
    with open(meta_path) as f:
        meta = json.load(f)
    cam = meta["camera"]["intrinsics"]
    K = np.array([
        [cam["fx"],       0, cam["cx"]],
        [      0, cam["fy"], cam["cy"]],
        [      0,       0,        1  ],
    ], dtype=np.float32)
    intrinsics = CameraIntrinsicsBase(
        fx=cam["fx"],
        fy=cam["fy"],
        cx=cam["cx"],
        cy=cam["cy"],
        width=cam["width"],
        height=cam["height"],
    )
    return K, int(cam["height"]), int(cam["width"]), intrinsics


def main():
    args = _args
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # ---- Build separate segmentation and pure GenPose2 agents ----
    seg_cfg, score_cfg, seg_agent, score_agent, energy_agent, scale_agent = \
        init_pipeline_agents(args)

    # ---- Load val/train annotation list ----
    ann_file = os.path.join(args.nuclear_data_path, "annotations", f"{args.split}.json")
    dataset = NuclearWorkpieceDataset(
        cfg=seg_cfg,
        data_dir=args.nuclear_data_path,
        annotation_file=ann_file,
        mode="seg",
        img_size=args.img_size,
    )

    num_vis = min(args.num_vis, len(dataset))
    print(f"Running full pipeline on {num_vis} images from '{args.split}' split...\n")

    for idx in range(num_vis):
        sample   = dataset[idx]
        img_id   = sample["image_id"].item()
        img_info = dataset.coco.loadImgs(img_id)[0]
        stem     = os.path.splitext(img_info["file_name"])[0]

        # ---- Load full-res RGB + depth + intrinsics ----
        img_path   = os.path.join(args.nuclear_data_path, "images", img_info["file_name"])
        depth_path = os.path.join(args.nuclear_data_path, "depth", stem + ".exr")
        meta_path  = os.path.join(args.nuclear_data_path, "meta",  stem + ".json")

        rgb_orig = cv2.imread(img_path)                             # BGR
        depth_orig = cv2.imread(depth_path,
                                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth_orig is not None and len(depth_orig.shape) == 3:
            depth_orig = depth_orig[:, :, 0]                       # take first channel
        K_33, im_H, im_W, camera_intrinsics = read_meta(meta_path)

        # ---- Segmentation (top-1 mask from SegNet path) ----
        batch = collate_nuclear([sample])
        batch = process_batch_seg(batch, device)
        with torch.no_grad():
            class_logits, mask_logits = seg_agent.net(batch, mode="segmentation")
        pred_masks, pred_classes, pred_scores = postprocess_seg(
            class_logits[0], mask_logits[0], args.score_threshold)

        print(f"[{idx+1}/{num_vis}] {img_info['file_name']}: "
              f"{len(pred_masks)} detection(s) above threshold {args.score_threshold}")

        instances = []
        for mask_crop, cls_id, score in zip(pred_masks, pred_classes, pred_scores):
            # Resize predicted mask (224x224) back to original image resolution
            mask_orig = cv2.resize(mask_crop.astype(np.float32),
                                   (im_W, im_H),
                                   interpolation=cv2.INTER_LINEAR) > 0.5

            inst_info = {
                "cls_id": cls_id,
                "score": score,
                "mask_orig": mask_orig,
                "R": None,
                "t": None,
                "size": None,
                "size_source": None,
            }

            if depth_orig is None:
                print(f"  ↳ {CLASS_NAMES[cls_id]} ({score:.2f}): "
                      "no depth map — skipping pose estimation")
                instances.append(inst_info)
                continue

            # ---- Extract point cloud from depth + mask ----
            pt_data = extract_pointcloud(
                rgb_orig, depth_orig, mask_orig, K_33,
                args.img_size, args.num_points)

            if pt_data is None:
                print(f"  ↳ {CLASS_NAMES[cls_id]} ({score:.2f}): "
                      "insufficient depth in mask region — skipping pose")
                instances.append(inst_info)
                continue

            # ---- Pure GenPose2 pose + size inference ----
            try:
                pose_res = infer_pose_and_size(
                    score_agent=score_agent,
                    energy_agent=energy_agent,
                    scale_agent=scale_agent,
                    cfg=score_cfg,
                    pt_data=pt_data,
                    device=device,
                    repeat_num=args.repeat_num,
                )
                inst_info["R"] = pose_res["R"]
                inst_info["t"] = pose_res["t"]
                inst_info["size"] = pose_res["size"]
                inst_info["size_source"] = pose_res["size_source"]
                t = pose_res["t"]
                size = pose_res["size"]
                print(f"  ↳ {CLASS_NAMES[cls_id]} ({score:.2f}): "
                      f"t = [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m, "
                      f"size({pose_res['size_source']}) = "
                      f"[{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}] m")
            except Exception as e:
                print(f"  ↳ {CLASS_NAMES[cls_id]} ({score:.2f}): "
                      f"pose estimation failed: {e}")

            instances.append(inst_info)

        # ---- Visualise ----
        vis = visualize_result(rgb_orig, instances, camera_intrinsics, args.img_size)
        out_path = os.path.join(args.output_dir,
                                f"{args.split}_{img_info['file_name']}")
        cv2.imwrite(out_path, vis)
        print(f"  Saved → {out_path}\n")

    print(f"Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
