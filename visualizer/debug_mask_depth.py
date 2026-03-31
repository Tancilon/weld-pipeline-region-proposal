"""
Debug SegNet mask vs depth validity for the nuclear full pipeline.

This script reproduces the mask-selection and depth-validity checks used by
`runners/infer_nuclear_full.py` and saves a multi-panel visualization for each
sample:
  1. RGB image
  2. Top-1 predicted mask overlay
  3. Depth heatmap
  4. Mask vs valid-depth overlap on the full image
  5. ROI RGB crop + ROI mask
  6. ROI depth heatmap + ROI valid-depth overlap

Usage:
    python visualizer/debug_mask_depth.py \
        --nuclear_data_path ./data/aiws5.2_nuclear_workpieces \
        --ckpt_path ./results/ckpts/SegNet/ckpt_epoch100.pth \
        --split val \
        --image_name 0009.png \
        --output_dir ./results/debug_mask_depth
"""

import sys
import os
import argparse

_p = argparse.ArgumentParser(description="Debug mask and depth overlap for nuclear inference")
_p.add_argument("--nuclear_data_path", type=str, required=True)
_p.add_argument("--ckpt_path", type=str, required=True, help="Path to SegNet checkpoint")
_p.add_argument("--split", type=str, default="val", choices=["train", "val"])
_p.add_argument("--output_dir", type=str, default="./results/debug_mask_depth")
_p.add_argument("--num_vis", type=int, default=20)
_p.add_argument("--image_name", type=str, default=None,
                help="Optional file name filter, e.g. 0009.png")
_p.add_argument("--score_threshold", type=float, default=0.5)
_p.add_argument("--device", type=str, default="cuda")
_p.add_argument("--img_size", type=int, default=224)
_p.add_argument("--panel_width", type=int, default=640)
_p.add_argument("--panel_height", type=int, default=360)
_args = _p.parse_args()

sys.argv = sys.argv[:1]

import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.datasets_nuclear import (  # noqa: E402
    CLASS_NAMES,
    NuclearWorkpieceDataset,
    collate_nuclear,
    process_batch_seg,
)
from networks.posenet_agent import PoseNet  # noqa: E402
from utils.datasets_utils import aug_bbox_eval, crop_resize_by_warp_affine  # noqa: E402


COLORS_BGR = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
]

_FONT = None


def _get_font(size=22):
    global _FONT
    if _FONT is not None:
        return _FONT
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                _FONT = ImageFont.truetype(path, size)
                return _FONT
            except Exception:
                continue
    _FONT = ImageFont.load_default()
    return _FONT


def put_text(img_bgr, text, x, y, text_color=(255, 255, 255), bg_color=None, font_size=22):
    font = _get_font(font_size)
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    if bg_color is not None:
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle([bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3],
                       fill=bg_color[::-1])
    draw.text((x, y), text, font=font, fill=text_color[::-1])
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def build_cfg(args):
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
    cfg.log_dir = "debug_mask_depth"
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
    cfg.save_video = False
    cfg.enable_segmentation = True
    cfg.num_queries = 50
    cfg.query_inject_layer = -4
    cfg.num_object_classes = 6
    cfg.unfreeze_dino_last_n = 4
    cfg.seg_loss_weight = 1.0
    cfg.cls_loss_weight = 2.0
    return cfg


def load_seg_agent(cfg, ckpt_path):
    print("Building segmentation model...")
    agent = PoseNet(cfg)
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = agent.net.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if missing:
        print(f"  [seg ckpt] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  [seg ckpt] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    agent.net.eval()
    return agent


def postprocess_seg(class_logits, mask_logits, score_threshold):
    probs = class_logits.softmax(dim=-1)
    obj_probs = probs[:, :-1]
    max_scores, max_classes = obj_probs.max(dim=-1)

    best = None
    for i in range(len(max_scores)):
        score = max_scores[i].item()
        if score < score_threshold:
            continue
        cls_id = max_classes[i].item()
        mask = mask_logits[i].sigmoid().cpu().numpy()
        if best is None or score > best[2]:
            best = (mask, cls_id, score)

    if best is None:
        return [], [], []
    mask, cls_id, score = best
    return [mask], [cls_id], [score]


def read_depth(depth_path):
    if not os.path.exists(depth_path):
        return None
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        return None
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.float32)


def resize_with_pad(image, target_w, target_h, bg_color=(20, 20, 20)):
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def add_title(panel, title):
    titled = panel.copy()
    cv2.rectangle(titled, (0, 0), (titled.shape[1], 38), (0, 0, 0), -1)
    return put_text(titled, title, 10, 8, bg_color=(0, 0, 0), font_size=22)


def colorize_depth(depth):
    vis = np.full(depth.shape + (3,), 32, dtype=np.uint8)
    valid = depth > 0
    if not np.any(valid):
        return vis
    values = depth[valid]
    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[valid] = heat[valid]
    return vis


def draw_mask_overlay(rgb, mask, color, alpha=0.45):
    out = rgb.copy()
    if mask is None or not np.any(mask):
        return out
    overlay = out.copy()
    overlay[mask] = color
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def draw_valid_overlap(rgb, mask, valid_depth):
    out = cv2.addWeighted(rgb, 0.55, np.zeros_like(rgb), 0.45, 0)
    if mask is None:
        return out
    valid_in_mask = mask & valid_depth
    invalid_in_mask = mask & (~valid_depth)
    overlay = out.copy()
    overlay[invalid_in_mask] = (0, 0, 255)
    overlay[valid_in_mask] = (0, 220, 0)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
    out = put_text(out, "green=mask&depth  red=mask&invalid_depth", 10, 44,
                   bg_color=(0, 0, 0), font_size=18)
    return out


def draw_roi_overlay(roi_rgb, roi_mask):
    return draw_mask_overlay(roi_rgb, roi_mask, (0, 255, 255), alpha=0.35)


def draw_roi_depth_overlay(roi_depth, roi_mask):
    base = colorize_depth(roi_depth)
    valid = (roi_depth > 0) & roi_mask
    invalid = roi_mask & (~(roi_depth > 0))
    overlay = base.copy()
    overlay[invalid] = (0, 0, 255)
    overlay[valid] = (0, 220, 0)
    out = cv2.addWeighted(overlay, 0.45, base, 0.55, 0)
    out = put_text(out, "green=roi valid  red=roi invalid", 10, 44,
                   bg_color=(0, 0, 0), font_size=18)
    return out


def make_text_panel(lines, width, height):
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    y = 14
    for line in lines:
        panel = put_text(panel, line, 12, y, bg_color=(0, 0, 0), font_size=20)
        y += 30
    return panel


def analyze_depth_debug(rgb_orig, depth_orig, mask_orig, img_size):
    result = {
        "mask_pixels": int(mask_orig.sum()),
        "orig_valid_pixels": 0,
        "roi_mask_pixels": 0,
        "roi_valid_pixels": 0,
        "status": "no_depth",
        "roi_rgb": None,
        "roi_depth": None,
        "roi_mask": None,
        "orig_valid_mask": None,
    }

    if depth_orig is None:
        return result

    orig_valid = (depth_orig > 0) & mask_orig
    result["orig_valid_mask"] = orig_valid
    result["orig_valid_pixels"] = int(orig_valid.sum())

    ys_m, xs_m = np.where(mask_orig)
    if len(ys_m) == 0:
        result["status"] = "empty_mask"
        return result

    bbox_xyxy = np.array([xs_m.min(), ys_m.min(), xs_m.max(), ys_m.max()], dtype=np.float32)
    bbox_center, scale = aug_bbox_eval(bbox_xyxy, rgb_orig.shape[0], rgb_orig.shape[1])

    roi_rgb = crop_resize_by_warp_affine(
        rgb_orig, bbox_center, scale, img_size, interpolation=cv2.INTER_LINEAR
    )
    roi_depth = crop_resize_by_warp_affine(
        depth_orig, bbox_center, scale, img_size, interpolation=cv2.INTER_NEAREST
    )
    roi_mask = crop_resize_by_warp_affine(
        mask_orig.astype(np.float32), bbox_center, scale, img_size, interpolation=cv2.INTER_NEAREST
    ) > 0.5
    roi_valid = (roi_depth > 0) & roi_mask

    result["roi_rgb"] = roi_rgb
    result["roi_depth"] = roi_depth
    result["roi_mask"] = roi_mask
    result["roi_mask_pixels"] = int(roi_mask.sum())
    result["roi_valid_pixels"] = int(roi_valid.sum())
    result["status"] = "pose_ready" if result["roi_valid_pixels"] >= 10 else "insufficient_depth"
    return result


def build_preview(rgb_orig, depth_orig, file_name, cls_name, score, mask_orig, debug_info,
                  panel_width, panel_height):
    color = COLORS_BGR[0] if cls_name is None else COLORS_BGR[CLASS_NAMES.index(cls_name)]

    if mask_orig is None:
        pred_overlay = put_text(rgb_orig.copy(), "No detection above threshold", 20, 20,
                                bg_color=(0, 0, 0), font_size=26)
        overlap = make_text_panel(["No predicted mask"], rgb_orig.shape[1], rgb_orig.shape[0])
    else:
        pred_overlay = draw_mask_overlay(rgb_orig, mask_orig, color)
        if cls_name is not None and score is not None:
            pred_overlay = put_text(pred_overlay, f"{cls_name} {score:.2f}", 20, 20,
                                    bg_color=color, font_size=24)
        overlap = draw_valid_overlap(rgb_orig, mask_orig, depth_orig > 0 if depth_orig is not None else np.zeros(mask_orig.shape, dtype=bool))

    if depth_orig is None:
        depth_panel = make_text_panel(["Depth file missing or unreadable"], rgb_orig.shape[1], rgb_orig.shape[0])
    else:
        depth_panel = colorize_depth(depth_orig)

    if debug_info["roi_rgb"] is None:
        roi_rgb_panel = make_text_panel([f"status: {debug_info['status']}"], panel_width, panel_height)
        roi_depth_panel = make_text_panel(["No ROI debug data"], panel_width, panel_height)
    else:
        roi_rgb_panel = draw_roi_overlay(debug_info["roi_rgb"], debug_info["roi_mask"])
        roi_depth_panel = draw_roi_depth_overlay(debug_info["roi_depth"], debug_info["roi_mask"])

    panels = [
        add_title(resize_with_pad(rgb_orig, panel_width, panel_height), "RGB"),
        add_title(resize_with_pad(pred_overlay, panel_width, panel_height), "Top-1 Pred Mask"),
        add_title(resize_with_pad(depth_panel, panel_width, panel_height), "Depth Heatmap"),
        add_title(resize_with_pad(overlap, panel_width, panel_height), "Mask vs Valid Depth"),
        add_title(resize_with_pad(roi_rgb_panel, panel_width, panel_height), "ROI RGB + Mask"),
        add_title(resize_with_pad(roi_depth_panel, panel_width, panel_height), "ROI Depth + Valid"),
    ]

    row1 = np.concatenate(panels[:3], axis=1)
    row2 = np.concatenate(panels[3:], axis=1)

    footer_h = 112
    footer = np.full((footer_h, row1.shape[1], 3), 18, dtype=np.uint8)
    footer = put_text(footer, f"image: {file_name}", 12, 10, bg_color=(0, 0, 0), font_size=22)
    footer = put_text(
        footer,
        f"status: {debug_info['status']} | class: {cls_name or 'none'} | score: {score if score is not None else 0.0:.2f}",
        12, 40, bg_color=(0, 0, 0), font_size=22,
    )
    footer = put_text(
        footer,
        f"mask_pixels: {debug_info['mask_pixels']} | orig_valid_in_mask: {debug_info['orig_valid_pixels']} | "
        f"roi_mask_pixels: {debug_info['roi_mask_pixels']} | roi_valid_in_mask: {debug_info['roi_valid_pixels']} | "
        "pose threshold: roi_valid >= 10",
        12, 70, bg_color=(0, 0, 0), font_size=20,
    )

    return np.concatenate([row1, row2, footer], axis=0)


def get_target_indices(dataset, image_name, num_vis):
    if image_name is not None:
        for idx, img_id in enumerate(dataset.image_ids):
            img_info = dataset.coco.loadImgs(img_id)[0]
            if img_info["file_name"] == image_name:
                return [idx]
        raise FileNotFoundError(f"Image '{image_name}' not found in split '{_args.split}'")
    return list(range(min(num_vis, len(dataset))))


def main():
    args = _args
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = build_cfg(args)
    agent = load_seg_agent(cfg, args.ckpt_path)

    ann_file = os.path.join(args.nuclear_data_path, "annotations", f"{args.split}.json")
    dataset = NuclearWorkpieceDataset(
        cfg=cfg,
        data_dir=args.nuclear_data_path,
        annotation_file=ann_file,
        mode="seg",
        img_size=args.img_size,
    )
    indices = get_target_indices(dataset, args.image_name, args.num_vis)
    print(f"Debug visualizing {len(indices)} image(s) from '{args.split}' split...")

    for vis_idx, dataset_idx in enumerate(indices, start=1):
        sample = dataset[dataset_idx]
        img_id = sample["image_id"].item()
        img_info = dataset.coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]

        img_path = os.path.join(args.nuclear_data_path, "images", file_name)
        depth_path = os.path.join(
            args.nuclear_data_path, "depth", os.path.splitext(file_name)[0] + ".exr"
        )
        rgb_orig = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if rgb_orig is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        depth_orig = read_depth(depth_path)

        batch = collate_nuclear([sample])
        batch = process_batch_seg(batch, cfg.device)
        with torch.no_grad():
            class_logits, mask_logits = agent.net(batch, mode="segmentation")
        pred_masks, pred_classes, pred_scores = postprocess_seg(
            class_logits[0], mask_logits[0], args.score_threshold
        )

        cls_name = None
        score = None
        mask_orig = None
        debug_info = {
            "mask_pixels": 0,
            "orig_valid_pixels": 0,
            "roi_mask_pixels": 0,
            "roi_valid_pixels": 0,
            "status": "no_detection",
            "roi_rgb": None,
            "roi_depth": None,
            "roi_mask": None,
            "orig_valid_mask": None,
        }

        if pred_masks:
            mask_crop = pred_masks[0]
            cls_id = pred_classes[0]
            score = pred_scores[0]
            cls_name = CLASS_NAMES[cls_id]
            mask_orig = cv2.resize(
                mask_crop.astype(np.float32),
                (rgb_orig.shape[1], rgb_orig.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ) > 0.5
            debug_info = analyze_depth_debug(rgb_orig, depth_orig, mask_orig, args.img_size)

        preview = build_preview(
            rgb_orig=rgb_orig,
            depth_orig=depth_orig,
            file_name=file_name,
            cls_name=cls_name,
            score=score,
            mask_orig=mask_orig,
            debug_info=debug_info,
            panel_width=args.panel_width,
            panel_height=args.panel_height,
        )

        out_path = os.path.join(args.output_dir, f"{args.split}_{file_name}")
        ok = cv2.imwrite(out_path, preview)
        if not ok:
            raise RuntimeError(f"Failed to write preview image: {out_path}")

        print(
            f"[{vis_idx}/{len(indices)}] {file_name}: "
            f"status={debug_info['status']}, "
            f"mask_pixels={debug_info['mask_pixels']}, "
            f"orig_valid_in_mask={debug_info['orig_valid_pixels']}, "
            f"roi_valid_in_mask={debug_info['roi_valid_pixels']} -> {out_path}"
        )


if __name__ == "__main__":
    main()
