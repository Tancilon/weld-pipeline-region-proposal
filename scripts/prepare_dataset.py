#!/usr/bin/env python3
"""
数据集预处理脚本 —— 为 Stage 1 分类训练准备数据

完成三步：
  1. 自动生成 mask.exr（SAM2 自动分割 或 深度阈值分割 fallback）
  2. 将现有简化版 meta.json 转为 cutoop ImageMetaData 格式
  3. 生成 configs/obj_meta.json（全局类别注册表）

用法：
  # SAM2 模式（推荐，需要安装 sam2 和下载 checkpoint）
  python scripts/prepare_dataset.py \
      --data_dir data/aiws5.1-dataset-test \
      --mode sam2 \
      --sam2_checkpoint path/to/sam2.1_hiera_tiny.pt \
      --sam2_model_cfg camera/configs/sam2.1/sam2.1_hiera_t.yaml

  # 深度阈值 fallback（无需 SAM2，适合单工件+平面桌面场景）
  python scripts/prepare_dataset.py \
      --data_dir data/aiws5.1-dataset-test \
      --mode depth
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# ──────────────────────────────────────────────
# 类别映射：class_name (拼音) → class_label (int)
# 与 datasets/types.py WORKPIECE_CLASSES 保持一致（0-indexed）
# ──────────────────────────────────────────────
CLASS_NAME_TO_LABEL = {
    "gaiban": 0,       # 盖板
    "fangguan": 1,     # 方管
    "labakou": 2,      # 喇叭口
    "Hxinggang": 3,    # H型钢
    "hxinggang": 3,    # H型钢（小写兼容）
    "pokou": 4,        # 坡口
    "caogang": 5,      # 槽钢
}

# cutoop obj_meta.json 的规范类名列表（按 class_label 排序）
CANONICAL_CLASSES = [
    {"name": "gaiban",    "label": 0},
    {"name": "fangguan",  "label": 1},
    {"name": "labakou",   "label": 2},
    {"name": "Hxinggang", "label": 3},
    {"name": "pokou",     "label": 4},
    {"name": "caogang",   "label": 5},
]

# 默认对称性标签
DEFAULT_SYMMETRY = {
    "any": False,
    "x": "none",
    "y": "none",
    "z": "none",
}

# 默认 ObjectTag（cutoop ObjectInfo.tag）
DEFAULT_OBJECT_TAG = {
    "datatype": "train",
    "sceneChanger": False,
    "symmetry": DEFAULT_SYMMETRY,
    "materialOptions": ["metal"],
    "upAxis": ["Y"],
}


# ============================================================
# Step 1: 生成 mask.exr
# ============================================================

def save_mask_exr(filepath: str, mask: np.ndarray):
    """保存 uint32 单通道掩码为 .exr 文件。"""
    import OpenEXR
    import Imath

    mask = mask.astype(np.uint32)
    h, w = mask.shape
    header = OpenEXR.Header(w, h)
    header["channels"] = {"Y": Imath.Channel(Imath.PixelType(Imath.PixelType.UINT))}
    exr = OpenEXR.OutputFile(filepath, header)
    exr.writePixels({"Y": mask.tobytes()})
    exr.close()


def generate_mask_sam2(rgb: np.ndarray, mask_generator) -> np.ndarray:
    """用 SAM2 AutomaticMaskGenerator 生成单工件掩码。"""
    masks_output = mask_generator.generate(rgb)
    if not masks_output:
        return np.zeros(rgb.shape[:2], dtype=np.uint32)

    h, w = rgb.shape[:2]
    total_pixels = h * w
    max_area = total_pixels * 0.5

    # 过滤：面积在合理范围内
    filtered = [
        m for m in masks_output
        if m["area"] >= 1000 and m["area"] <= max_area
    ]
    if not filtered:
        # fallback: 选 IoU 最高的
        filtered = sorted(masks_output, key=lambda m: m["predicted_iou"], reverse=True)[:1]

    # 选最大的作为工件
    filtered.sort(key=lambda m: m["area"], reverse=True)
    best = filtered[0]

    mask = np.zeros((h, w), dtype=np.uint32)
    mask[best["segmentation"]] = 1
    return mask


def generate_mask_depth(depth: np.ndarray, percentile_low: float = 5, percentile_high: float = 80) -> np.ndarray:
    """
    用深度图阈值分割生成单工件掩码。

    适用于：单工件放在平面桌面上的场景。
    逻辑：桌面是最大的平面（深度值最集中的区域），
          工件比桌面高（深度值更小，离相机更近）。
    """
    h, w = depth.shape
    valid = depth > 0
    if np.sum(valid) < 100:
        return np.zeros((h, w), dtype=np.uint32)

    valid_depths = depth[valid]

    # 桌面通常占据大部分像素，用中位数近似桌面深度
    table_depth = np.median(valid_depths)

    # 工件比桌面高（深度更小），但要排除噪声
    depth_min = np.percentile(valid_depths, percentile_low)
    # 工件深度 < 桌面深度 - 阈值
    depth_threshold = table_depth * 0.97  # 工件至少比桌面近 3%

    object_mask = valid & (depth > depth_min) & (depth < depth_threshold)

    # 形态学操作清理噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    object_mask_u8 = object_mask.astype(np.uint8)
    object_mask_u8 = cv2.morphologyEx(object_mask_u8, cv2.MORPH_OPEN, kernel)
    object_mask_u8 = cv2.morphologyEx(object_mask_u8, cv2.MORPH_CLOSE, kernel)

    # 取最大连通域作为工件
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(object_mask_u8, connectivity=8)
    if num_labels <= 1:
        return np.zeros((h, w), dtype=np.uint32)

    # label 0 是背景，从 1 开始找最大区域
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1

    mask = np.zeros((h, w), dtype=np.uint32)
    mask[labels == largest_label] = 1
    return mask


def load_depth_exr(filepath: str) -> np.ndarray:
    """用 OpenCV 读取 .exr 深度图。"""
    depth = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth: {filepath}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.float32)


# ============================================================
# Step 2: 转换 meta.json 为 cutoop ImageMetaData 格式
# ============================================================

def convert_meta_json(old_meta: dict) -> dict:
    """
    将简化版 meta.json 转为 cutoop ImageMetaData 格式。

    cutoop 关键格式要求：
    - objects 是 dict，key 格式为 "maskid_oid"，mask_id 从 key 前缀解析
    - camera 是 ViewInfo（继承 Pose），包含 quaternion/translation + intrinsics
    - ObjectPoseInfo 需要 id, material, world_quaternion_wxyz, world_translation 等字段
    - ObjectMetaInfo 需要 instance_path, scale, is_background 等字段
    """
    intrinsics = old_meta["camera"]["intrinsics"]
    annotation = old_meta["annotation"]
    class_name = annotation["class_name"]
    dimensions = annotation["dimensions"]

    class_label = CLASS_NAME_TO_LABEL.get(class_name)
    if class_label is None:
        class_label = CLASS_NAME_TO_LABEL.get(class_name.lower(), 0)
        print(f"  [WARN] Unknown class_name '{class_name}', mapped to label {class_label}")

    oid = f"{class_name}_001"
    mask_id = 1  # 我们生成的 mask 中工件统一标为 1

    # cutoop objects dict: key = "maskid_oid"
    objects_dict = {
        f"{mask_id}_{oid}": {
            "meta": {
                "oid": oid,
                "class_name": class_name,
                "class_label": class_label,
                "instance_path": "",
                "scale": [1.0, 1.0, 1.0],
                "is_background": False,
                "bbox_side_len": dimensions,
            },
            # Stage 1 分类训练不需要真实 pose，填占位值
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],  # identity rotation
            "translation": [0.0, 0.0, 1.0],             # 1m in front of camera
            "is_valid": True,
            "id": 0,
            "material": ["metal"],
            "world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "world_translation": [0.0, 0.0, 1.0],
        }
    }

    # cutoop camera 是 ViewInfo（继承自 Pose）
    camera = {
        "quaternion": [1.0, 0.0, 0.0, 0.0],   # identity（相机在世界原点）
        "translation": [0.0, 0.0, 0.0],
        "intrinsics": intrinsics,
        "scene_obj_path": "",
        "background_image_path": "",
        "background_depth_path": "",
        "distances": [0.5, 2.0],
        "kind": "real",
    }

    new_meta = {
        "objects": objects_dict,
        "camera": camera,
        "scene_dataset": "aiws",
        "env_param": {},
        "face_up": True,
        "concentrated": False,
        "comments": "auto-generated for Stage 1 classification training",
        "runtime_seed": 0,
        "baseline_dis": 0,
        "emitter_dist_l": 0,
    }
    return new_meta


# ============================================================
# Step 3: 生成 configs/obj_meta.json
# ============================================================

def generate_obj_meta(data_dir: str, output_path: str):
    """
    扫描数据集，生成 cutoop ObjectMetaData 格式的 obj_meta.json。

    cutoop 格式要求：
    - class_list: list[ClassListItem]，每项含 name, label, instance_ids, stat
    - instance_dict: dict[str, ObjectInfo]，key 为 oid
    - ObjectInfo 含 object_id, source, name, obj_path, tag, class_label, class_name, dimensions
    """
    # 收集所有出现过的 (class_name, oid, dimensions)
    instance_info = {}  # oid -> (class_name, dimensions)
    meta_files = sorted(glob.glob(os.path.join(data_dir, "*_meta.json")))

    for mf in meta_files:
        # 跳过备份文件
        if "_meta_original.json" in mf:
            continue
        with open(mf) as f:
            meta = json.load(f)
        if "objects" in meta:
            # 新格式（cutoop dict 或已转换的）
            objs = meta["objects"]
            if isinstance(objs, dict):
                for key, obj in objs.items():
                    cn = obj["meta"]["class_name"]
                    oid = obj["meta"]["oid"]
                    dims = obj["meta"].get("bbox_side_len", [0.1, 0.1, 0.1])
                    instance_info[oid] = (cn, dims)
            elif isinstance(objs, list):
                for obj in objs:
                    cn = obj["meta"]["class_name"]
                    oid = obj["meta"]["oid"]
                    dims = obj["meta"].get("bbox_side_len", [0.1, 0.1, 0.1])
                    instance_info[oid] = (cn, dims)
        elif "annotation" in meta:
            cn = meta["annotation"]["class_name"]
            dims = meta["annotation"]["dimensions"]
            oid = f"{cn}_001"
            instance_info[oid] = (cn, dims)

    # 构建 class_list（ClassListItem 格式）
    # 按 class 聚合 instance_ids
    class_to_oids = {cls["name"]: [] for cls in CANONICAL_CLASSES}
    for oid, (cn, _) in sorted(instance_info.items()):
        if cn in class_to_oids:
            class_to_oids[cn].append(oid)
        elif cn.lower() in {k.lower() for k in class_to_oids}:
            # 大小写兼容
            for key in class_to_oids:
                if key.lower() == cn.lower():
                    class_to_oids[key].append(oid)
                    break

    class_list = []
    for cls in CANONICAL_CLASSES:
        class_list.append({
            "name": cls["name"],
            "label": cls["label"],
            "instance_ids": class_to_oids.get(cls["name"], []),
            "stat": {},
        })

    # 构建 instance_dict（ObjectInfo 格式）
    instance_dict = {}
    for oid, (class_name, dimensions) in sorted(instance_info.items()):
        label = CLASS_NAME_TO_LABEL.get(class_name, CLASS_NAME_TO_LABEL.get(class_name.lower(), 0))
        instance_dict[oid] = {
            "object_id": oid,
            "source": "real",
            "name": oid,
            "obj_path": "",
            "tag": DEFAULT_OBJECT_TAG,
            "class_label": label,
            "class_name": class_name,
            "dimensions": dimensions,
        }

    obj_meta = {
        "class_list": class_list,
        "instance_dict": instance_dict,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj_meta, f, indent=2, ensure_ascii=False)
    print(f"[Step 3] obj_meta.json saved to {output_path}")
    print(f"         {len(class_list)} classes, {len(instance_dict)} instances")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Stage 1 classification training")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory (e.g. data/aiws5.1-dataset-test)")
    parser.add_argument("--mode", type=str, default="depth", choices=["sam2", "depth"],
                        help="Mask generation mode: 'sam2' or 'depth' (default: depth)")
    parser.add_argument("--obj_meta_output", type=str, default="configs/obj_meta.json",
                        help="Output path for obj_meta.json")
    # SAM2 参数
    parser.add_argument("--sam2_checkpoint", type=str, default=None)
    parser.add_argument("--sam2_model_cfg", type=str, default="camera/configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2_points_per_side", type=int, default=32)
    parser.add_argument("--sam2_pred_iou_thresh", type=float, default=0.88)
    parser.add_argument("--sam2_stability_score_thresh", type=float, default=0.95)
    # 深度分割参数
    parser.add_argument("--depth_percentile_low", type=float, default=5)
    parser.add_argument("--depth_percentile_high", type=float, default=80)
    # 控制参数
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mask.exr and meta.json")
    parser.add_argument("--skip_mask", action="store_true", help="Skip mask generation (only convert meta)")
    parser.add_argument("--visualize", action="store_true", help="Save visualization of generated masks")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"Error: {data_dir} does not exist")
        sys.exit(1)

    # 找到所有帧：根据 XXXX_color.png 前缀
    color_files = sorted(glob.glob(os.path.join(data_dir, "*_color.png")))
    if not color_files:
        print(f"Error: No *_color.png found in {data_dir}")
        sys.exit(1)

    prefixes = [f.replace("_color.png", "") for f in color_files]
    print(f"Found {len(prefixes)} frames in {data_dir}")

    # ── 初始化 mask generator ──
    mask_generator = None
    if not args.skip_mask and args.mode == "sam2":
        if args.sam2_checkpoint is None:
            print("Error: --sam2_checkpoint is required for sam2 mode")
            sys.exit(1)
        print("Loading SAM2 model...")
        sys.path.insert(0, "./segment-anything-2-real-time")
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_model = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=args.sam2_points_per_side,
            pred_iou_thresh=args.sam2_pred_iou_thresh,
            stability_score_thresh=args.sam2_stability_score_thresh,
        )
        print(f"SAM2 ready on {device}")

    # ── 可视化输出目录 ──
    vis_dir = None
    if args.visualize:
        vis_dir = os.path.join(data_dir, "_mask_vis")
        os.makedirs(vis_dir, exist_ok=True)

    # ── Step 1 & 2: 逐帧处理 ──
    success_count = 0
    skip_count = 0
    fail_count = 0

    for prefix in prefixes:
        frame_id = os.path.basename(prefix)
        color_path = prefix + "_color.png"
        depth_path = prefix + "_depth.exr"
        meta_path = prefix + "_meta.json"
        mask_path = prefix + "_mask.exr"

        # 读取原始 meta
        if not os.path.exists(meta_path):
            print(f"  [SKIP] {frame_id}: missing meta.json")
            skip_count += 1
            continue

        with open(meta_path) as f:
            old_meta = json.load(f)

        # ── Step 1: 生成 mask ──
        if not args.skip_mask:
            if os.path.exists(mask_path) and not args.overwrite:
                print(f"  [SKIP] {frame_id}_mask.exr already exists (use --overwrite to force)")
            else:
                try:
                    if args.mode == "sam2":
                        rgb = cv2.imread(color_path)
                        if rgb is None:
                            raise FileNotFoundError(f"Cannot read: {color_path}")
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                        mask = generate_mask_sam2(rgb, mask_generator)
                    else:
                        depth = load_depth_exr(depth_path)
                        mask = generate_mask_depth(
                            depth,
                            percentile_low=args.depth_percentile_low,
                            percentile_high=args.depth_percentile_high,
                        )

                    n_pixels = np.sum(mask > 0)
                    if n_pixels < 100:
                        print(f"  [WARN] {frame_id}: mask has only {n_pixels} pixels")

                    save_mask_exr(mask_path, mask)
                    print(f"  [MASK] {frame_id}: {n_pixels} pixels ({args.mode})")

                    # 可视化
                    if vis_dir is not None:
                        rgb_vis = cv2.imread(color_path)
                        if rgb_vis is not None:
                            overlay = rgb_vis.copy()
                            overlay[mask > 0] = [0, 255, 0]
                            blended = cv2.addWeighted(rgb_vis, 0.6, overlay, 0.4, 0)
                            cv2.imwrite(os.path.join(vis_dir, f"{frame_id}_mask_vis.png"), blended)

                except Exception as e:
                    print(f"  [FAIL] {frame_id}: mask generation failed: {e}")
                    fail_count += 1
                    continue

        # ── Step 2: 转换 meta.json ──
        # 仅在旧格式（有 annotation 字段、无 objects 字段）时转换
        if "annotation" in old_meta and ("objects" not in old_meta or args.overwrite):
            new_meta = convert_meta_json(old_meta)
            # 备份原始 meta
            backup_path = prefix + "_meta_original.json"
            if not os.path.exists(backup_path):
                with open(backup_path, "w") as f:
                    json.dump(old_meta, f, indent=2, ensure_ascii=False)

            with open(meta_path, "w") as f:
                json.dump(new_meta, f, indent=2, ensure_ascii=False)
            print(f"  [META] {frame_id}: converted (class={old_meta['annotation']['class_name']})")
        elif "objects" in old_meta:
            print(f"  [SKIP] {frame_id}: meta.json already in cutoop format")

        success_count += 1

    print(f"\n{'='*50}")
    print(f"Processed: {success_count}, Skipped: {skip_count}, Failed: {fail_count}")

    # ── Step 3: 生成 obj_meta.json ──
    generate_obj_meta(data_dir, args.obj_meta_output)

    print(f"\nDone! Dataset is ready for Stage 1 classification training.")
    print(f"\nReminder: make sure cutoop is installed before training:")
    print(f"  pip install cutoop")


if __name__ == "__main__":
    main()
