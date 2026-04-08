import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_nuclear_seg_dataset.py"

CLASS_NAMES = ["盖板", "方管", "喇叭口", "H型钢", "槽钢", "坡口"]


def _write_coco_fixture(root, split_name, image_name, category_name, image_id=1, annotation_id=1):
    annotations_dir = root / "annotations"
    images_dir = root / "images"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    (images_dir / image_name).write_bytes(b"")
    payload = {
        "images": [
            {
                "id": image_id,
                "file_name": image_name,
                "width": 8,
                "height": 8,
            }
        ],
        "annotations": [
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]],
                "area": 1,
                "bbox": [0, 0, 1, 1],
                "iscrowd": 0,
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": category_name,
            }
        ],
    }
    (annotations_dir / f"{split_name}.json").write_text(json.dumps(payload), encoding="utf-8")


def _run_checker(data_path):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--nuclear_data_path", str(data_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_checker_reports_summary_for_valid_dataset(tmp_path):
    data_path = tmp_path / "dataset"
    annotations_dir = data_path / "annotations"
    images_dir = data_path / "images"
    annotations_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    train_payload = {
        "images": [
            {"id": 1, "file_name": "train_a.png", "width": 8, "height": 8},
            {"id": 2, "file_name": "train_b.png", "width": 8, "height": 8},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]],
                "area": 1,
                "bbox": [0, 0, 1, 1],
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 2,
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]],
                "area": 1,
                "bbox": [0, 0, 1, 1],
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": idx, "name": name} for idx, name in enumerate(CLASS_NAMES, start=1)
        ],
    }
    val_payload = {
        "images": [
            {"id": 3, "file_name": "val_a.png", "width": 8, "height": 8}
        ],
        "annotations": [
            {
                "id": 3,
                "image_id": 3,
                "category_id": 1,
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]],
                "area": 1,
                "bbox": [0, 0, 1, 1],
                "iscrowd": 0,
            }
        ],
        "categories": [
            {"id": idx, "name": name} for idx, name in enumerate(CLASS_NAMES, start=1)
        ],
    }
    (annotations_dir / "train.json").write_text(json.dumps(train_payload), encoding="utf-8")
    (annotations_dir / "val.json").write_text(json.dumps(val_payload), encoding="utf-8")
    for image_name in ["train_a.png", "train_b.png", "val_a.png"]:
        (images_dir / image_name).write_bytes(b"")

    result = _run_checker(data_path)

    assert result.returncode == 0, result.stderr
    assert "train images: 2" in result.stdout
    assert "val images: 1" in result.stdout
    assert "盖板: 2" in result.stdout
    assert "方管: 1" in result.stdout


def test_checker_rejects_missing_files(tmp_path):
    data_path = tmp_path / "dataset"
    annotations_dir = data_path / "annotations"
    images_dir = data_path / "images"
    annotations_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    payload = {
        "images": [
            {"id": 1, "file_name": "missing.png", "width": 8, "height": 8}
        ],
        "annotations": [],
        "categories": [
            {"id": idx, "name": name} for idx, name in enumerate(CLASS_NAMES, start=1)
        ],
    }
    (annotations_dir / "train.json").write_text(json.dumps(payload), encoding="utf-8")
    (annotations_dir / "val.json").write_text(json.dumps(payload), encoding="utf-8")

    result = _run_checker(data_path)

    assert result.returncode != 0
    assert "missing image file" in result.stderr.lower()


@pytest.mark.parametrize("missing_split", ["train", "val"])
def test_checker_rejects_missing_required_annotation_json(tmp_path, missing_split):
    data_path = tmp_path / "dataset"
    annotations_dir = data_path / "annotations"
    images_dir = data_path / "images"
    annotations_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    present_split = "val" if missing_split == "train" else "train"
    _write_coco_fixture(
        data_path,
        present_split,
        f"{present_split}.png",
        CLASS_NAMES[0],
    )

    result = _run_checker(data_path)

    assert result.returncode != 0
    assert f"missing required annotation file" in result.stderr.lower()
    assert f"{missing_split}.json" in result.stderr


def test_checker_rejects_invalid_category_names(tmp_path):
    data_path = tmp_path / "dataset"
    annotations_dir = data_path / "annotations"
    images_dir = data_path / "images"
    annotations_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    (images_dir / "train.png").write_bytes(b"")
    train_payload = {
        "images": [{"id": 1, "file_name": "train.png", "width": 8, "height": 8}],
        "annotations": [],
        "categories": [{"id": 1, "name": "bad-class"}],
    }
    (annotations_dir / "train.json").write_text(json.dumps(train_payload), encoding="utf-8")
    (annotations_dir / "val.json").write_text(json.dumps(train_payload), encoding="utf-8")

    result = _run_checker(data_path)

    assert result.returncode != 0
    assert "unsupported category" in result.stderr.lower()


def test_checker_allows_background_category_in_coco_metadata(tmp_path):
    data_path = tmp_path / "dataset"
    annotations_dir = data_path / "annotations"
    images_dir = data_path / "images"
    annotations_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    (images_dir / "train.png").write_bytes(b"")
    (images_dir / "val.png").write_bytes(b"")

    categories = [{"id": 0, "name": "__background__"}] + [
        {"id": idx, "name": name} for idx, name in enumerate(CLASS_NAMES, start=1)
    ]
    train_payload = {
        "images": [{"id": 1, "file_name": "train.png", "width": 8, "height": 8}],
        "annotations": [],
        "categories": categories,
    }
    val_payload = {
        "images": [{"id": 2, "file_name": "val.png", "width": 8, "height": 8}],
        "annotations": [],
        "categories": categories,
    }
    (annotations_dir / "train.json").write_text(json.dumps(train_payload), encoding="utf-8")
    (annotations_dir / "val.json").write_text(json.dumps(val_payload), encoding="utf-8")

    result = _run_checker(data_path)

    assert result.returncode == 0, result.stderr
    assert "train images: 1" in result.stdout
    assert "val images: 1" in result.stdout


def test_checker_rejects_duplicate_file_names_across_splits(tmp_path):
    data_path = tmp_path / "dataset"
    annotations_dir = data_path / "annotations"
    images_dir = data_path / "images"
    annotations_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    for image_name in ["shared.png", "train_only.png", "val_only.png"]:
        (images_dir / image_name).write_bytes(b"")

    categories = [{"id": idx, "name": name} for idx, name in enumerate(CLASS_NAMES, start=1)]
    train_payload = {
        "images": [{"id": 1, "file_name": "shared.png", "width": 8, "height": 8}],
        "annotations": [],
        "categories": categories,
    }
    val_payload = {
        "images": [{"id": 1, "file_name": "shared.png", "width": 8, "height": 8}],
        "annotations": [],
        "categories": categories,
    }
    (annotations_dir / "train.json").write_text(json.dumps(train_payload), encoding="utf-8")
    (annotations_dir / "val.json").write_text(json.dumps(val_payload), encoding="utf-8")

    result = _run_checker(data_path)

    assert result.returncode != 0
    assert "duplicate file name" in result.stderr.lower()


def test_checker_allows_duplicate_image_ids_across_splits(tmp_path):
    data_path = tmp_path / "dataset"
    annotations_dir = data_path / "annotations"
    images_dir = data_path / "images"
    annotations_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    for image_name in ["train_only.png", "val_only.png"]:
        (images_dir / image_name).write_bytes(b"")

    categories = [{"id": idx, "name": name} for idx, name in enumerate(CLASS_NAMES, start=1)]
    train_payload = {
        "images": [{"id": 1, "file_name": "train_only.png", "width": 8, "height": 8}],
        "annotations": [],
        "categories": categories,
    }
    val_payload = {
        "images": [{"id": 1, "file_name": "val_only.png", "width": 8, "height": 8}],
        "annotations": [],
        "categories": categories,
    }
    (annotations_dir / "train.json").write_text(json.dumps(train_payload), encoding="utf-8")
    (annotations_dir / "val.json").write_text(json.dumps(val_payload), encoding="utf-8")

    result = _run_checker(data_path)

    assert result.returncode == 0, result.stderr
    assert "train images: 1" in result.stdout
    assert "val images: 1" in result.stdout
