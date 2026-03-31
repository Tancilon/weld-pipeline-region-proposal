"""
Convert ISAT annotations to COCO format with train/val split.

Usage:
    python scripts/convert_isat_to_coco.py \
        --isat_dir data/aiws5.2_nuclear_workpieces/isat_annotations \
        --output_dir data/aiws5.2_nuclear_workpieces/annotations \
        --val_ratio 0.2 \
        --seed 42
"""

import argparse
import json
import os
import random
from collections import Counter


CATEGORIES = [
    {"id": 1, "name": "盖板"},
    {"id": 2, "name": "方管"},
    {"id": 3, "name": "喇叭口"},
    {"id": 4, "name": "H型钢"},
    {"id": 5, "name": "槽钢"},
    {"id": 6, "name": "坡口"},
]

CAT_NAME_TO_ID = {c["name"]: c["id"] for c in CATEGORIES}


def parse_isat_file(filepath):
    """Parse a single ISAT annotation JSON.

    Returns:
        image_info: dict with file_name, width, height.
        objects: list of dicts with category, segmentation, bbox, area.
    """
    with open(filepath) as f:
        data = json.load(f)

    info = data["info"]
    image_info = {
        "file_name": info["name"],
        "width": info["width"],
        "height": info["height"],
    }

    objects = []
    for obj in data.get("objects", []):
        cat_name = obj["category"]
        if cat_name not in CAT_NAME_TO_ID:
            print(f"  [Warning] Unknown category '{cat_name}' in {filepath}, skipping")
            continue

        # ISAT segmentation: list of [x, y] pairs -> flatten to [x1,y1,x2,y2,...]
        seg_points = obj["segmentation"]
        flat_seg = []
        for pt in seg_points:
            flat_seg.extend([float(pt[0]), float(pt[1])])

        # Compute bbox from polygon points
        xs = [pt[0] for pt in seg_points]
        ys = [pt[1] for pt in seg_points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]  # [x, y, w, h]

        # Area from annotation or compute from bbox
        area = obj.get("area", bbox[2] * bbox[3])

        objects.append({
            "category_id": CAT_NAME_TO_ID[cat_name],
            "segmentation": [flat_seg],
            "bbox": bbox,
            "area": float(area),
            "iscrowd": 0,
        })

    return image_info, objects


def build_coco_json(image_entries, categories):
    """Build a COCO-format dict from image entries.

    Args:
        image_entries: list of (image_info, objects) tuples.
        categories: list of category dicts.

    Returns:
        COCO dict.
    """
    images = []
    annotations = []
    ann_id = 0

    for img_id, (img_info, objects) in enumerate(image_entries):
        images.append({
            "id": img_id,
            "file_name": img_info["file_name"],
            "width": img_info["width"],
            "height": img_info["height"],
        })

        for obj in objects:
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": obj["category_id"],
                "segmentation": obj["segmentation"],
                "bbox": obj["bbox"],
                "area": obj["area"],
                "iscrowd": obj["iscrowd"],
            })
            ann_id += 1

    return {
        "info": {"description": "Nuclear workpieces (converted from ISAT)"},
        "licenses": [],
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isat_dir", required=True, help="Path to ISAT annotation directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for COCO JSONs")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse all ISAT files
    all_entries = []  # list of (image_info, objects)
    isat_files = sorted(f for f in os.listdir(args.isat_dir) if f.endswith(".json"))

    for fname in isat_files:
        filepath = os.path.join(args.isat_dir, fname)
        img_info, objects = parse_isat_file(filepath)
        if len(objects) == 0:
            print(f"  [Warning] No valid objects in {fname}, skipping")
            continue
        all_entries.append((img_info, objects))

    print(f"\nParsed {len(all_entries)} images, "
          f"{sum(len(objs) for _, objs in all_entries)} total instances")

    # Statistics
    cat_counter = Counter()
    multi_count = 0
    for _, objs in all_entries:
        if len(objs) > 1:
            multi_count += 1
        for obj in objs:
            cat_counter[obj["category_id"]] += 1

    cat_id_to_name = {c["id"]: c["name"] for c in CATEGORIES}
    print("\nPer-category counts:")
    for cid, count in sorted(cat_counter.items()):
        print(f"  {cat_id_to_name[cid]}: {count}")
    print(f"Multi-instance images: {multi_count}")

    # Split train/val
    random.seed(args.seed)
    indices = list(range(len(all_entries)))
    random.shuffle(indices)

    val_size = max(1, int(len(indices) * args.val_ratio))
    val_indices = set(indices[:val_size])
    train_indices = set(indices[val_size:])

    train_entries = [all_entries[i] for i in sorted(train_indices)]
    val_entries = [all_entries[i] for i in sorted(val_indices)]

    print(f"\nSplit: train={len(train_entries)}, val={len(val_entries)}")
    print(f"  Train instances: {sum(len(objs) for _, objs in train_entries)}")
    print(f"  Val instances: {sum(len(objs) for _, objs in val_entries)}")

    # Build and save COCO JSONs
    train_coco = build_coco_json(train_entries, CATEGORIES)
    val_coco = build_coco_json(val_entries, CATEGORIES)

    train_path = os.path.join(args.output_dir, "train.json")
    val_path = os.path.join(args.output_dir, "val.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {train_path}")
    print(f"Saved: {val_path}")


if __name__ == "__main__":
    main()
