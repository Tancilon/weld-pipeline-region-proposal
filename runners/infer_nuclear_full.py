"""
Full pipeline inference for nuclear workpieces:
RGB + Depth -> Segmentation -> 6D Pose (+ optional EnergyNet / ScaleNet).

Architecture note:
  - Segmentation and 6D pose run through one main segmentation-enabled agent.
  - EnergyNet / ScaleNet remain optional pure GenPose2 auxiliary agents.

Usage:
    python runners/infer_nuclear_full.py \
        --nuclear_data_path ./data/aiws5.2_nuclear_workpieces \
        --seg_ckpt          ./results/ckpts/SegNet/ckpt_epoch100.pth \
        --energy_ckpt       ./results/ckpts/EnergyNet/energynet.pth \
        --scale_ckpt        ./results/ckpts/ScaleNet/scalenet.pth \
        --split val \
        --output_dir ./results/full_pipeline \
        --num_vis 5 \
        --score_threshold 0.5

If --energy_ckpt is omitted pose hypotheses are aggregated directly from the
main agent outputs. If --scale_ckpt is omitted object size is estimated
geometrically from the segmented point cloud and aggregated pose.
"""

import sys
import os

# --------------------------------------------------------------------------- #
# Parse args FIRST — downstream imports call get_config() at module level and
# will choke on our custom flags if they reach parse_args() first.
# --------------------------------------------------------------------------- #
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from runners.infer_nuclear_full_lib import (
    build_arg_parser,
    infer_pose_and_size,
    init_pipeline_agents,
)

_args = build_arg_parser().parse_args()

sys.argv = sys.argv[:1]  # clear argv before heavy imports trigger get_config()

# --------------------------------------------------------------------------- #
import json
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from datasets.datasets_nuclear import (
    CLASS_NAMES, NUM_CLASSES, collate_nuclear, process_batch_seg,
    NuclearWorkpieceDataset,
)
from datasets.datasets_omni6dpose import Omni6DPoseDataSet
from cutoop.data_types import CameraIntrinsicsBase
from cutoop.eval_utils import DetectMatch
from utils.datasets_utils import aug_bbox_eval, crop_resize_by_warp_affine, get_2d_coord_np


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

    # ---- Build the single main runtime agent plus optional aux agents ----
    main_cfg, main_agent, energy_agent, scale_agent = init_pipeline_agents(args)

    # ---- Load val/train annotation list ----
    ann_file = os.path.join(args.nuclear_data_path, "annotations", f"{args.split}.json")
    dataset = NuclearWorkpieceDataset(
        cfg=main_cfg,
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
            class_logits, mask_logits = main_agent.net(batch, mode="segmentation")
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

            # ---- Main-agent pose + optional aux size inference ----
            try:
                pose_res = infer_pose_and_size(
                    main_agent=main_agent,
                    energy_agent=energy_agent,
                    scale_agent=scale_agent,
                    cfg=main_cfg,
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
