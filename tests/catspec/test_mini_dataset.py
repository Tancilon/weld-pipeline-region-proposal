from pathlib import Path

import torch

from catspec.autogt import export_autogt_manifest
from catspec.mini_dataset import (
    CatSpecMiniDataset,
    collate_mini_batch,
    derive_targets,
    load_autogt_rows,
)


def _make_manifest(tmp_path: Path) -> Path:
    manifest = export_autogt_manifest(
        [
            Path("specs/categories/square_tube.yaml"),
            Path("specs/categories/channel_steel.yaml"),
            Path("specs/categories/H_beam.yaml"),
            Path("specs/categories/bellmouth.yaml"),
        ],
        tmp_path / "autogt",
    )
    return Path(manifest["manifest_path"])


def test_load_autogt_rows_from_manifest(tmp_path):
    manifest_path = _make_manifest(tmp_path)

    rows = load_autogt_rows(manifest_path)

    assert [row["category"] for row in rows] == ["square_tube", "channel_steel", "H_beam", "bellmouth"]
    for row in rows:
        assert "source_mesh" in row
        assert "reference_weld_path" in row
        assert "generated_locus" in row
        assert "weld_meta" in row
        assert "topology" in row
        assert "metrics" in row


def test_derive_targets_from_autogt_row(tmp_path):
    row = load_autogt_rows(_make_manifest(tmp_path))[0]

    targets = derive_targets(row)

    assert targets["topology_name"] == "closed_rounded_rect"
    assert targets["path_count"] == 1
    assert targets["segment_sequence_name"] == "line_arc_line_arc_line_arc_line_arc"
    assert isinstance(targets["topology_label"], int)
    assert isinstance(targets["path_count_label"], int)
    assert isinstance(targets["segment_sequence_label"], int)


def test_dataset_modes_keep_reference_out_of_model_inputs(tmp_path):
    manifest_path = _make_manifest(tmp_path)
    correct = CatSpecMiniDataset(manifest_path, mode="spec_correct", max_tokens=32)
    shuffled = CatSpecMiniDataset(manifest_path, mode="spec_shuffle", max_tokens=32)
    no_spec = CatSpecMiniDataset(manifest_path, mode="no_spec", max_tokens=32)

    item = correct[0]
    shuffled_item = shuffled[0]
    no_spec_item = no_spec[0]

    assert item["category"] == "square_tube"
    assert item["prompt_category"] == "square_tube"
    assert shuffled_item["prompt_category"] == "channel_steel"
    assert no_spec_item["prompt_category"] == "<no_spec>"
    assert "reference_weld_path" in item["raw"]
    assert "reference_weld_path" not in item["model_inputs"]
    assert item["model_inputs"]["token_ids"].shape == (32,)
    assert no_spec_item["model_inputs"]["token_mask"].sum().item() == 1


def test_collate_mini_batch_stacks_tensors(tmp_path):
    dataset = CatSpecMiniDataset(_make_manifest(tmp_path), mode="spec_correct", max_tokens=32)

    batch = collate_mini_batch([dataset[0], dataset[1]])

    assert batch["model_inputs"]["token_ids"].shape == (2, 32)
    assert batch["model_inputs"]["token_mask"].dtype == torch.bool
    assert batch["targets"]["topology_label"].shape == (2,)
    assert batch["targets"]["path_count_label"].shape == (2,)
    assert batch["targets"]["segment_sequence_label"].shape == (2,)
    assert batch["raw"][0]["category"] == "square_tube"
