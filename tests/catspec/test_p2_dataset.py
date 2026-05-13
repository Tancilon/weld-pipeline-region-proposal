from pathlib import Path

import torch

from catspec.p2_dataset import (
    CatSpecP2Dataset,
    category_balanced_weights,
    collate_p2_batch,
    load_p2_rows,
    summarize_p2_manifest,
)
from catspec.p2_preprocess import preprocess_p2_dataset


DATASET_ROOT = Path("/home/dq/mnt/localgit/aiws5.2/datasets/obj_share_models")


def _manifest(tmp_path: Path) -> Path:
    result = preprocess_p2_dataset(DATASET_ROOT, tmp_path / "p2", seed=7, force=True)
    return Path(result["manifest_path"])


def test_load_p2_rows_and_summary_from_manifest(tmp_path):
    manifest_path = _manifest(tmp_path)

    rows = load_p2_rows(manifest_path, split="all")
    summary = summarize_p2_manifest(manifest_path)

    assert len(rows) == 4
    assert summary["sample_count"] == 4
    assert summary["failure_count"] == 1
    assert summary["category_counts"]["square_tube"] == 1
    assert summary["split_counts"]["all"] == 4


def test_p2_dataset_modes_include_geometry_and_exclude_reference_from_model_inputs(tmp_path):
    manifest_path = _manifest(tmp_path)
    correct = CatSpecP2Dataset(manifest_path, split="all", mode="spec_correct", max_tokens=48)
    shuffled = CatSpecP2Dataset(manifest_path, split="all", mode="spec_shuffle", max_tokens=48)
    no_spec = CatSpecP2Dataset(manifest_path, split="all", mode="no_spec", max_tokens=48)

    item = correct[0]
    shuffled_item = shuffled[0]
    no_spec_item = no_spec[0]

    assert item["category"] == "square_tube"
    assert item["prompt_category"] == "square_tube"
    assert shuffled_item["prompt_category"] != item["category"]
    assert no_spec_item["prompt_category"] == "<no_spec>"
    assert item["model_inputs"]["token_ids"].shape == (48,)
    assert item["model_inputs"]["geometry_features"].dtype == torch.float32
    assert item["model_inputs"]["geometry_features"].numel() == correct.geometry_dim
    assert "reference_weld_path" in item["raw"]
    assert "reference_weld_path" not in item["model_inputs"]
    assert no_spec_item["model_inputs"]["token_mask"].sum().item() == 1


def test_collate_p2_batch_and_balanced_weights(tmp_path):
    dataset = CatSpecP2Dataset(_manifest(tmp_path), split="all", mode="spec_correct", max_tokens=32)

    batch = collate_p2_batch([dataset[0], dataset[1]])
    weights = category_balanced_weights(dataset.rows)

    assert batch["model_inputs"]["token_ids"].shape == (2, 32)
    assert batch["model_inputs"]["token_mask"].dtype == torch.bool
    assert batch["model_inputs"]["geometry_features"].shape == (2, dataset.geometry_dim)
    assert batch["targets"]["topology_label"].shape == (2,)
    assert len(weights) == len(dataset)
    assert all(weight > 0 for weight in weights)
