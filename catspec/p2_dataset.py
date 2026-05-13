"""Dataset adapter for the CatSpec-Pose P2 pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

from catspec.mini_dataset import DatasetMode, derive_targets
from catspec.spec_encoder import batch_encode_specs


P2DatasetMode = DatasetMode
P2_SPLITS = {"train", "val", "test", "all"}
GEOMETRY_DIM = 8


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_manifest_path(manifest_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = manifest_path.parent / path
    return candidate if candidate.exists() else path


def _load_manifest(manifest_or_index: str | Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    path = Path(manifest_or_index)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return None, rows
    manifest = _load_json(path)
    if "records" in manifest and manifest["records"]:
        return manifest, list(manifest["records"])
    index_path = _resolve_manifest_path(path, manifest["sample_index_path"])
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return manifest, rows


def _split_ids(manifest: dict[str, Any] | None, split: str) -> set[str] | None:
    if split not in P2_SPLITS:
        raise ValueError(f"unsupported P2 split: {split}")
    if split == "all" or manifest is None:
        return None
    if "splits" in manifest and split in manifest["splits"]:
        return set(manifest["splits"][split])
    split_path = Path(manifest["split_path"])
    splits = _load_json(split_path)
    return set(splits["splits"][split])


def load_p2_rows(
    manifest_or_index: str | Path,
    split: Literal["train", "val", "test", "all"] = "train",
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load P2 sample rows and filter by split/category."""

    manifest, rows = _load_manifest(manifest_or_index)
    wanted_ids = _split_ids(manifest, split)
    wanted_categories = set(categories) if categories else None
    filtered = []
    for row in rows:
        if wanted_ids is not None and row["sample_id"] not in wanted_ids:
            continue
        if wanted_categories is not None and row["category"] not in wanted_categories:
            continue
        filtered.append(row)
    return filtered


def _geometry_features(row: dict[str, Any]) -> torch.Tensor:
    summary = row.get("source_geometry", {})
    extents = [float(value) for value in summary.get("bbox_extent", [0.0, 0.0, 0.0])[:3]]
    while len(extents) < 3:
        extents.append(0.0)
    max_extent = max(extents) if extents else 0.0
    aspect = [value / max_extent if max_extent > 0 else 0.0 for value in extents]
    values = [
        *extents,
        *aspect,
        float(torch.log1p(torch.tensor(float(summary.get("vertex_count", 0.0))))),
        float(torch.log1p(torch.tensor(float(summary.get("face_count", 0.0))))),
    ]
    return torch.tensor(values[:GEOMETRY_DIM], dtype=torch.float32)


def _prompt_row_for_mode(rows: list[dict[str, Any]], idx: int, mode: P2DatasetMode) -> dict[str, Any] | None:
    if mode == "spec_correct":
        return rows[idx]
    if mode == "spec_shuffle":
        if not rows:
            return None
        return rows[(idx + 1) % len(rows)]
    if mode == "no_spec":
        return None
    raise ValueError(f"unsupported dataset mode: {mode}")


class CatSpecP2Dataset(Dataset):
    """P2 manifest-backed dataset with split filtering and spec-conditioned inputs."""

    geometry_dim = GEOMETRY_DIM

    def __init__(
        self,
        manifest_or_index: str | Path,
        split: Literal["train", "val", "test", "all"] = "train",
        mode: P2DatasetMode = "spec_correct",
        max_tokens: int = 96,
        categories: list[str] | None = None,
    ) -> None:
        self.manifest_or_index = str(manifest_or_index)
        self.split = split
        self.mode = mode
        self.max_tokens = max_tokens
        self.rows = load_p2_rows(manifest_or_index, split=split, categories=categories)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        prompt = _prompt_row_for_mode(self.rows, idx, self.mode)
        encoded = batch_encode_specs([prompt], max_tokens=self.max_tokens)
        targets = derive_targets(row)
        model_inputs = {
            "token_ids": encoded["token_ids"][0],
            "token_mask": encoded["token_mask"][0],
            "geometry_features": _geometry_features(row),
        }
        return {
            "sample_id": row["sample_id"],
            "category": row["category"],
            "prompt_category": prompt["category"] if prompt is not None else "<no_spec>",
            "split": self.split,
            "mode": self.mode,
            "source_mesh": row["source_mesh"],
            "model_inputs": model_inputs,
            "targets": targets,
            "raw": row,
        }


def collate_p2_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate P2 dataset samples for DataLoader."""

    return {
        "sample_id": [item["sample_id"] for item in items],
        "category": [item["category"] for item in items],
        "prompt_category": [item["prompt_category"] for item in items],
        "split": [item["split"] for item in items],
        "mode": [item["mode"] for item in items],
        "source_mesh": [item["source_mesh"] for item in items],
        "model_inputs": {
            "token_ids": torch.stack([item["model_inputs"]["token_ids"] for item in items]),
            "token_mask": torch.stack([item["model_inputs"]["token_mask"] for item in items]),
            "geometry_features": torch.stack([item["model_inputs"]["geometry_features"] for item in items]),
        },
        "targets": {
            "topology_label": torch.tensor([item["targets"]["topology_label"] for item in items], dtype=torch.long),
            "path_count_label": torch.tensor([item["targets"]["path_count_label"] for item in items], dtype=torch.long),
            "segment_sequence_label": torch.tensor(
                [item["targets"]["segment_sequence_label"] for item in items],
                dtype=torch.long,
            ),
        },
        "target_names": {
            "topology_name": [item["targets"]["topology_name"] for item in items],
            "path_count": [item["targets"]["path_count"] for item in items],
            "segment_sequence_name": [item["targets"]["segment_sequence_name"] for item in items],
        },
        "raw": [item["raw"] for item in items],
    }


def category_balanced_weights(rows: list[dict[str, Any]]) -> list[float]:
    """Return inverse-frequency category weights aligned with rows."""

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return [1.0 / counts[row["category"]] for row in rows]


def summarize_p2_manifest(manifest_or_index: str | Path) -> dict[str, Any]:
    """Return integrity and split stats for a P2 manifest."""

    manifest, rows = _load_manifest(manifest_or_index)
    category_counts: dict[str, int] = {}
    for row in rows:
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
    split_counts = {}
    if manifest and "splits" in manifest:
        split_counts = {name: len(values) for name, values in manifest["splits"].items()}
    return {
        "sample_count": len(rows),
        "failure_count": int(manifest.get("failure_count", 0)) if manifest else 0,
        "category_counts": category_counts,
        "split_counts": split_counts,
    }
