"""
Visualize EoMT segmentation predictions on nuclear workpiece images.

Loads a trained checkpoint, runs inference on val (or train) images, and
saves overlay visualizations with predicted masks and class labels.

Usage:
    python runners/visualize_seg.py \
        --nuclear_data_path ./data/aiws5.2_nuclear_workpieces \
        --ckpt_path ./results/ckpts/SegNet/ckpt_epoch100.pth \
        --split val \
        --output_dir ./results/vis_seg \
        --num_vis 20 \
        --score_threshold 0.5
"""

import sys
import os
import argparse

# --------------------------------------------------------------------------- #
# IMPORTANT: Parse our own args BEFORE any downstream imports.
# networks/pts_encoder/pointnet2.py calls get_config() at module level, which
# in turn calls parser.parse_args().  If our custom flags are still in sys.argv
# at that point, argparse raises "unrecognized arguments".  We parse them here
# first, then strip sys.argv down to just the program name so get_config() sees
# a clean argument list.
# --------------------------------------------------------------------------- #
_vis_parser = argparse.ArgumentParser(description="Visualize EoMT segmentation results")
_vis_parser.add_argument("--nuclear_data_path", type=str, required=True)
_vis_parser.add_argument("--ckpt_path", type=str, required=True,
                         help="Path to trained SegNet checkpoint")
_vis_parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
_vis_parser.add_argument("--output_dir", type=str, default="./results/vis_seg")
_vis_parser.add_argument("--num_vis", type=int, default=20,
                         help="Max number of images to visualize")
_vis_parser.add_argument("--score_threshold", type=float, default=0.5)
_vis_parser.add_argument("--device", type=str, default="cuda")
_vis_parser.add_argument("--img_size", type=int, default=224)
_args = _vis_parser.parse_args()

# Strip custom flags so get_config()'s parse_args() doesn't see them
sys.argv = sys.argv[:1]

# Now safe to import modules that trigger get_config() at import time
import cv2
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.datasets_nuclear import (
    NuclearWorkpieceDataset, CLASS_NAMES, NUM_CLASSES, collate_nuclear, process_batch_seg
)
from networks.posenet_agent import PoseNet


# Per-class colors (BGR for cv2)
COLORS = [
    (0, 0, 255),     # 盖板   - red
    (0, 255, 0),     # 方管   - green
    (255, 0, 0),     # 喇叭口 - blue
    (0, 255, 255),   # H型钢  - yellow
    (255, 0, 255),   # 槽钢   - magenta
    (255, 255, 0),   # 坡口   - cyan
]


def visualize_one(rgb_orig, pred_masks, pred_classes, pred_scores,
                  gt_masks=None, gt_classes=None, alpha=0.5):
    """Create a visualization image with predicted (and optionally GT) masks.

    Args:
        rgb_orig: [H, W, 3] uint8 BGR image.
        pred_masks: list of [H', W'] binary masks.
        pred_classes: list of int class indices.
        pred_scores: list of float confidence scores.
        gt_masks: optional list of [H', W'] GT binary masks.
        gt_classes: optional list of int GT class indices.
        alpha: mask overlay transparency.

    Returns:
        vis: visualization image (BGR uint8).
    """
    h, w = rgb_orig.shape[:2]

    # Prediction overlay
    pred_vis = rgb_orig.copy()
    for mask, cls_id, score in zip(pred_masks, pred_classes, pred_scores):
        mask_resized = cv2.resize(mask.astype(np.float32), (w, h),
                                  interpolation=cv2.INTER_LINEAR)
        binary = (mask_resized > 0.5).astype(np.uint8)

        color = COLORS[cls_id % len(COLORS)]
        overlay = pred_vis.copy()
        overlay[binary == 1] = color
        pred_vis = cv2.addWeighted(overlay, alpha, pred_vis, 1 - alpha, 0)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(pred_vis, contours, -1, color, 2)

        if len(contours) > 0:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = contours[0][0][0]
            label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(pred_vis, (cx, cy - th - 4), (cx + tw, cy + 4), color, -1)
            cv2.putText(pred_vis, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if gt_masks is not None and gt_classes is not None:
        gt_vis = rgb_orig.copy()
        for mask, cls_id in zip(gt_masks, gt_classes):
            mask_resized = cv2.resize(mask.astype(np.float32), (w, h),
                                      interpolation=cv2.INTER_LINEAR)
            binary = (mask_resized > 0.5).astype(np.uint8)
            color = COLORS[cls_id % len(COLORS)]
            overlay = gt_vis.copy()
            overlay[binary == 1] = color
            gt_vis = cv2.addWeighted(overlay, alpha, gt_vis, 1 - alpha, 0)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gt_vis, contours, -1, color, 2)
            if len(contours) > 0:
                M = cv2.moments(contours[0])
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = contours[0][0][0]
                label = f"{CLASS_NAMES[cls_id]} (GT)"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(gt_vis, (cx, cy - th - 4), (cx + tw, cy + 4), color, -1)
                cv2.putText(gt_vis, label, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        vis = np.concatenate([gt_vis, pred_vis], axis=1)
        cv2.putText(vis, "Ground Truth", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(vis, "Prediction", (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    else:
        vis = pred_vis

    return vis


def postprocess_predictions(class_logits, mask_logits, score_threshold=0.5):
    """Post-process raw model outputs to extract instance predictions.

    Args:
        class_logits: [N, num_classes+1] tensor.
        mask_logits: [N, H, W] tensor.
        score_threshold: minimum confidence to keep a prediction.

    Returns:
        masks: list of [H, W] numpy arrays (sigmoid probabilities).
        classes: list of int class indices.
        scores: list of float confidence scores.
    """
    probs = class_logits.softmax(dim=-1)  # [N, C+1]
    obj_probs = probs[:, :-1]             # [N, C] exclude no-object
    max_scores, max_classes = obj_probs.max(dim=-1)

    masks, classes, scores = [], [], []
    for i in range(len(max_scores)):
        score = max_scores[i].item()
        if score < score_threshold:
            continue
        cls_id = max_classes[i].item()
        mask = mask_logits[i].sigmoid().cpu().numpy()
        masks.append(mask)
        classes.append(cls_id)
        scores.append(score)

    return masks, classes, scores


def main():
    args = _args  # parsed at module level before imports

    os.makedirs(args.output_dir, exist_ok=True)

    # Build a minimal config for model construction (avoids calling get_config() again)
    class Cfg:
        pass

    cfg = Cfg()
    cfg.device = args.device if torch.cuda.is_available() else "cpu"
    cfg.dino = "pointwise"
    cfg.pts_encoder = "pointnet2"
    cfg.agent_type = "score"
    cfg.pose_mode = "rot_matrix"
    cfg.regression_head = "Rx_Ry_and_T"
    cfg.sde_mode = "ve"
    cfg.num_points = 1024
    cfg.img_size = args.img_size
    cfg.pointnet2_params = "light"
    cfg.parallel = False
    cfg.is_train = False
    cfg.eval = False
    cfg.pred = False
    cfg.use_pretrain = False
    cfg.log_dir = "vis_seg"
    cfg.ema_rate = 0.999
    cfg.lr = 1e-4
    cfg.lr_decay = 0.99
    cfg.optimizer = "Adam"
    cfg.warmup = 50
    cfg.grad_clip = 1.0
    cfg.sampling_steps = 500
    cfg.sampler_mode = ["ode"]
    cfg.energy_mode = "IP"
    cfg.s_theta_mode = "score"
    cfg.norm_energy = "identical"
    cfg.scale_embedding = 180
    cfg.eval_repeat_num = 50
    cfg.repeat_num = 20
    cfg.num_gpu = 1
    cfg.enable_segmentation = True
    cfg.num_queries = 50
    cfg.query_inject_layer = -4
    cfg.num_object_classes = 6
    cfg.unfreeze_dino_last_n = 4
    cfg.seg_loss_weight = 1.0
    cfg.cls_loss_weight = 2.0

    print("Building model...")
    agent = PoseNet(cfg)
    print(f"Loading checkpoint: {args.ckpt_path}")
    agent.load_ckpt(model_dir=args.ckpt_path, model_path=True, load_model_only=True)
    # NOTE: save_ckpt() already applies ema.copy_to() before saving model_state_dict,
    # so the checkpoint weights are already EMA-averaged. Do NOT call ema.copy_to()
    # here — the ema shadow is freshly random-initialized and would overwrite good weights.
    agent.net.eval()

    ann_file = os.path.join(args.nuclear_data_path, "annotations", f"{args.split}.json")
    dataset = NuclearWorkpieceDataset(
        cfg=cfg,
        data_dir=args.nuclear_data_path,
        annotation_file=ann_file,
        mode="seg",
        img_size=args.img_size,
    )

    num_vis = min(args.num_vis, len(dataset))
    print(f"Visualizing {num_vis} images from '{args.split}' split...")

    for idx in range(num_vis):
        sample = dataset[idx]
        img_id = sample["image_id"].item()

        img_info = dataset.coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.nuclear_data_path, "images", img_info["file_name"])
        rgb_orig = cv2.imread(img_path)

        batch = collate_nuclear([sample])
        batch = process_batch_seg(batch, cfg.device)

        with torch.no_grad():
            class_logits, mask_logits = agent.net(batch, mode="segmentation")

        pred_masks, pred_classes, pred_scores = postprocess_predictions(
            class_logits[0], mask_logits[0], args.score_threshold
        )

        n_inst = sample["num_instances"].item()
        gt_masks_np = sample["gt_masks"][:n_inst].numpy()
        gt_classes_np = sample["gt_classes"][:n_inst].numpy().tolist()

        vis = visualize_one(
            rgb_orig, pred_masks, pred_classes, pred_scores,
            gt_masks=gt_masks_np, gt_classes=gt_classes_np,
        )

        out_name = f"{args.split}_{img_info['file_name']}"
        out_path = os.path.join(args.output_dir, out_name)
        cv2.imwrite(out_path, vis)
        pred_info = ", ".join(
            f"{CLASS_NAMES[c]}({s:.2f})" for c, s in zip(pred_classes, pred_scores)
        )
        print(f"  [{idx+1}/{num_vis}] {img_info['file_name']}: "
              f"{pred_info if pred_info else 'no detections'} -> {out_name}")

    print(f"\nDone! Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
