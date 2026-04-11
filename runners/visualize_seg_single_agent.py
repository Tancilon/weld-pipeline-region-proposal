import argparse
import os
import sys
from typing import Iterable

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from runners.infer_nuclear_full_lib import build_cfg, load_main_agent_checkpoint


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Visualize single-agent segmentation predictions on nuclear data"
    )
    parser.add_argument("--nuclear_data_path", type=str, required=True)
    parser.add_argument("--seg_ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output_dir", type=str, default="./results/vis_seg_single_agent")
    parser.add_argument("--num_vis", type=int, default=20)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--image_names", type=str, default="")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--repeat_num", type=int, default=10)
    return parser


def normalize_image_names(image_names):
    if not image_names:
        return None
    if isinstance(image_names, str):
        image_names = image_names.split(",")
    normalized = [name.strip() for name in image_names if name and name.strip()]
    return normalized or None


def select_sample_indices(dataset, image_names=None, num_vis=None):
    normalized_names = normalize_image_names(image_names)
    if normalized_names:
        filename_to_index = {}
        for index, image_id in enumerate(dataset.image_ids):
            img_info = dataset.coco.loadImgs([image_id])[0]
            filename_to_index[img_info["file_name"]] = index
        indices = []
        missing = []
        for name in normalized_names:
            matched_index = filename_to_index.get(name)
            if matched_index is None:
                missing.append(name)
                continue
            indices.append(matched_index)
        return indices, missing

    if num_vis is None:
        raise ValueError("num_vis is required when image_names is not provided")
    total = len(getattr(dataset, "image_ids", []))
    return list(range(min(num_vis, total))), []


def _load_cjk_font(size=22):
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def put_text(img_bgr, text, x, y, text_color_bgr=(255, 255, 255),
             bg_color_bgr=None, font_size=22):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    font = _load_cjk_font(font_size)
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    text_color_rgb = text_color_bgr[::-1]
    if bg_color_bgr is not None:
        bg_color_rgb = bg_color_bgr[::-1]
        bbox = draw.textbbox((x, y), text, font=font)
        pad = 3
        draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill=bg_color_rgb,
        )
    draw.text((x, y), text, font=font, fill=text_color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
]


def postprocess_predictions(class_logits, mask_logits, score_threshold=0.5):
    probs = class_logits.softmax(dim=-1)
    obj_probs = probs[:, :-1]
    max_scores, max_classes = obj_probs.max(dim=-1)

    best = None
    for idx in range(len(max_scores)):
        score = max_scores[idx].item()
        if score < score_threshold:
            continue
        cls_id = max_classes[idx].item()
        mask = mask_logits[idx].sigmoid().cpu().numpy()
        if best is None or score > best[2]:
            best = (mask, cls_id, score)

    if best is None:
        return [], [], []
    mask, cls_id, score = best
    return [mask], [cls_id], [score]


def _draw_instances(base_image, masks, classes, class_names, labels, alpha=0.5):
    import cv2
    import numpy as np

    h, w = base_image.shape[:2]
    vis = base_image.copy()
    for mask, cls_id, label in zip(masks, classes, labels):
        mask_resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        binary = (mask_resized > 0.5).astype(np.uint8)
        color = COLORS[cls_id % len(COLORS)]
        overlay = vis.copy()
        overlay[binary == 1] = color
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        if contours:
            moments = cv2.moments(contours[0])
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = contours[0][0][0]
            vis = put_text(
                vis,
                f"{class_names[cls_id]} {label}".strip(),
                cx,
                cy,
                text_color_bgr=(255, 255, 255),
                bg_color_bgr=color,
            )
    return vis


def visualize_comparison(rgb_orig, pred_masks, pred_classes, pred_scores,
                         gt_masks, gt_classes, class_names):
    import numpy as np

    gt_labels = ["(GT)"] * len(gt_classes)
    gt_vis = _draw_instances(rgb_orig, gt_masks, gt_classes, class_names, gt_labels)
    gt_summary = ", ".join(class_names[class_id] for class_id in gt_classes) if gt_classes else "none"
    gt_vis = put_text(
        gt_vis, f"GT: {gt_summary}", 10, 8,
        text_color_bgr=(255, 255, 255), bg_color_bgr=(0, 0, 0), font_size=28
    )

    if pred_classes:
        pred_labels = [f"{score:.2f}" for score in pred_scores]
        pred_vis = _draw_instances(rgb_orig, pred_masks, pred_classes, class_names, pred_labels)
        pred_summary = f"{class_names[pred_classes[0]]} ({pred_scores[0]:.2f})"
    else:
        pred_vis = rgb_orig.copy()
        pred_summary = "none"
    pred_vis = put_text(
        pred_vis, f"Pred: {pred_summary}", 10, 8,
        text_color_bgr=(255, 255, 255), bg_color_bgr=(0, 0, 0), font_size=28
    )

    vis = np.concatenate([gt_vis, pred_vis], axis=1)
    vis = put_text(
        vis, "Ground Truth", 10, 48,
        text_color_bgr=(255, 255, 255), bg_color_bgr=(0, 0, 0), font_size=24
    )
    vis = put_text(
        vis, "Prediction", rgb_orig.shape[1] + 10, 48,
        text_color_bgr=(255, 255, 255), bg_color_bgr=(0, 0, 0), font_size=24
    )
    return vis


def _build_agent(args):
    from networks.posenet_agent import PoseNet

    cfg = build_cfg(args, agent_type="score", enable_segmentation=True)
    cfg.log_dir = "visualize_seg_single_agent"
    agent = PoseNet(cfg)
    load_main_agent_checkpoint(agent, args.seg_ckpt)
    agent.net.eval()
    return cfg, agent


def run_visualization(args):
    import cv2
    import torch

    from datasets.datasets_nuclear import (
        CLASS_NAMES,
        NuclearWorkpieceDataset,
        collate_nuclear,
        process_batch_seg,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    cfg, agent = _build_agent(args)
    ann_file = os.path.join(args.nuclear_data_path, "annotations", f"{args.split}.json")
    dataset = NuclearWorkpieceDataset(
        cfg=cfg,
        data_dir=args.nuclear_data_path,
        annotation_file=ann_file,
        mode="seg",
        img_size=args.img_size,
    )

    selected_indices, missing_names = select_sample_indices(
        dataset,
        image_names=args.image_names,
        num_vis=args.num_vis,
    )
    if missing_names:
        print(f"Warning: requested image_names not found: {', '.join(missing_names)}")

    print(
        f"Visualizing {len(selected_indices)} image(s) from '{args.split}' split "
        f"into {args.output_dir}"
    )

    for order, dataset_index in enumerate(selected_indices, start=1):
        sample = dataset[dataset_index]
        img_id = sample["image_id"].item()
        img_info = dataset.coco.loadImgs([img_id])[0]
        img_path = os.path.join(args.nuclear_data_path, "images", img_info["file_name"])
        rgb_orig = cv2.imread(img_path)
        if rgb_orig is None:
            print(f"Warning: failed to load image '{img_path}', skipping")
            continue

        batch = collate_nuclear([sample])
        batch = process_batch_seg(batch, cfg.device)

        with torch.no_grad():
            class_logits, mask_logits = agent.net(batch, mode="segmentation")

        pred_masks, pred_classes, pred_scores = postprocess_predictions(
            class_logits[0], mask_logits[0], args.score_threshold
        )

        n_inst = sample["num_instances"].item()
        gt_masks = [sample["gt_masks"][i].numpy() for i in range(n_inst)]
        gt_classes = [sample["gt_classes"][i].item() for i in range(n_inst)]

        vis = visualize_comparison(
            rgb_orig=rgb_orig,
            pred_masks=pred_masks,
            pred_classes=pred_classes,
            pred_scores=pred_scores,
            gt_masks=gt_masks,
            gt_classes=gt_classes,
            class_names=CLASS_NAMES,
        )

        out_name = f"{args.split}_{img_info['file_name']}"
        out_path = os.path.join(args.output_dir, out_name)
        cv2.imwrite(out_path, vis)
        pred_summary = "none"
        if pred_classes:
            pred_summary = f"{CLASS_NAMES[pred_classes[0]]} ({pred_scores[0]:.2f})"
        print(f"[{order}/{len(selected_indices)}] {img_info['file_name']}: {pred_summary} -> {out_name}")

    print(f"Done. Visualizations saved to: {args.output_dir}")


def main(argv: Iterable[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if argv is None:
        sys.argv = sys.argv[:1]
    run_visualization(args)


if __name__ == "__main__":
    main()
