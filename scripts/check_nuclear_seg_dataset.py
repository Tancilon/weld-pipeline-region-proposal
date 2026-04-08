#!/usr/bin/env python3
"""Validate the nuclear segmentation dataset before training."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable


SUPPORTED_CLASSES = [
    "盖板",
    "方管",
    "喇叭口",
    "H型钢",
    "槽钢",
    "坡口",
]


class DatasetValidationError(RuntimeError):
    pass


@dataclass
class SplitSummary:
    split: str
    image_count: int
    instance_counts: Dict[str, int]
    image_ids: set[int]
    file_names: set[str]


def _fail(message: str) -> None:
    raise DatasetValidationError(message)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        _fail(f"missing required annotation file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"invalid json in {path}: {exc}")
    raise AssertionError("unreachable")


def _validate_categories(split: str, payload: dict) -> Dict[int, str]:
    categories = payload.get("categories", [])
    if not isinstance(categories, list):
        _fail(f"{split}: categories must be a list")

    category_map: Dict[int, str] = {}
    unsupported = set()
    for category in categories:
        if not isinstance(category, dict):
            _fail(f"{split}: category entries must be objects")
        cat_id = category.get("id")
        name = category.get("name")
        if cat_id is None or name is None:
            _fail(f"{split}: every category needs id and name")
        category_map[int(cat_id)] = str(name)
        if name not in SUPPORTED_CLASSES:
            unsupported.add(str(name))

    if unsupported:
        _fail(
            f"{split}: unsupported category name(s): "
            f"{', '.join(sorted(unsupported))}"
        )
    return category_map


def _validate_split(root: Path, split: str) -> SplitSummary:
    annotation_path = root / "annotations" / f"{split}.json"
    payload = _load_json(annotation_path)
    category_map = _validate_categories(split, payload)

    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    if not isinstance(images, list):
        _fail(f"{split}: images must be a list")
    if not isinstance(annotations, list):
        _fail(f"{split}: annotations must be a list")

    image_ids: set[int] = set()
    file_names: set[str] = set()
    image_by_id: Dict[int, dict] = {}
    image_instances = {name: 0 for name in SUPPORTED_CLASSES}

    for image in images:
        if not isinstance(image, dict):
            _fail(f"{split}: image entries must be objects")
        image_id = image.get("id")
        file_name = image.get("file_name")
        if image_id is None or file_name is None:
            _fail(f"{split}: every image needs id and file_name")
        image_id = int(image_id)
        file_name = str(file_name)
        if image_id in image_by_id:
            _fail(f"{split}: duplicate image id {image_id}")
        if file_name in file_names:
            _fail(f"{split}: duplicate file name {file_name}")
        image_by_id[image_id] = image
        image_ids.add(image_id)
        file_names.add(file_name)

        image_path = root / "images" / file_name
        if not image_path.is_file():
            _fail(f"{split}: missing image file: {image_path}")

    for annotation in annotations:
        if not isinstance(annotation, dict):
            _fail(f"{split}: annotation entries must be objects")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if image_id is None or category_id is None:
            _fail(f"{split}: every annotation needs image_id and category_id")

        image_id = int(image_id)
        if image_id not in image_by_id:
            _fail(f"{split}: annotation references unknown image_id {image_id}")

        category_name = category_map.get(int(category_id))
        if category_name is None:
            _fail(f"{split}: annotation references unknown category_id {category_id}")
        if category_name not in SUPPORTED_CLASSES:
            _fail(f"{split}: unsupported category name {category_name}")
        image_instances[category_name] += 1

    return SplitSummary(
        split=split,
        image_count=len(images),
        instance_counts=image_instances,
        image_ids=image_ids,
        file_names=file_names,
    )


def _format_summary(summaries: Iterable[SplitSummary]) -> str:
    summaries = list(summaries)
    total_instances = {name: 0 for name in SUPPORTED_CLASSES}
    lines = []
    for summary in summaries:
        lines.append(f"{summary.split} images: {summary.image_count}")
        for class_name in SUPPORTED_CLASSES:
            total_instances[class_name] += summary.instance_counts[class_name]

    lines.append("instance counts:")
    for class_name in SUPPORTED_CLASSES:
        lines.append(f"  {class_name}: {total_instances[class_name]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nuclear_data_path",
        required=True,
        help="Path to the nuclear dataset root.",
    )
    args = parser.parse_args(argv)

    root = Path(args.nuclear_data_path).expanduser().resolve()
    summaries = [_validate_split(root, "train"), _validate_split(root, "val")]

    train_summary, val_summary = summaries
    errors = []
    duplicate_ids = train_summary.image_ids & val_summary.image_ids
    duplicate_file_names = train_summary.file_names & val_summary.file_names
    if duplicate_ids:
        errors.append(
            "duplicate image id(s) across splits: "
            + ", ".join(str(item) for item in sorted(duplicate_ids))
        )
    if duplicate_file_names:
        errors.append(
            "duplicate file name(s) across splits: "
            + ", ".join(sorted(duplicate_file_names))
        )
    if errors:
        _fail("; ".join(errors))

    print(_format_summary(summaries))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
