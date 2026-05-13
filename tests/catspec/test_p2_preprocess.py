import json
from pathlib import Path

from catspec.p2_preprocess import preprocess_p2_dataset, scan_obj_pairs


DATASET_ROOT = Path("/home/dq/mnt/localgit/aiws5.2/datasets/obj_share_models")


def test_scan_obj_pairs_finds_source_and_weld_pairs():
    pairs = scan_obj_pairs(DATASET_ROOT)

    by_category = {pair["category"]: pair for pair in pairs}
    assert {"square_tube", "channel_steel", "H_beam", "bellmouth", "cover_plate"} <= set(by_category)
    assert by_category["square_tube"]["source_obj"].endswith("square_tube.obj")
    assert by_category["square_tube"]["reference_weld_obj"].endswith("square_tube_weld.obj")
    assert not Path(by_category["square_tube"]["source_obj"]).stem.endswith("_weld")


def test_preprocess_p2_dataset_writes_manifest_splits_stats_and_failures(tmp_path):
    result = preprocess_p2_dataset(DATASET_ROOT, tmp_path / "p2", seed=11, force=True)

    manifest_path = Path(result["manifest_path"])
    sample_index_path = Path(result["sample_index_path"])
    split_path = Path(result["split_path"])
    stats_path = Path(result["stats_path"])
    failures_path = Path(result["failures_path"])

    assert manifest_path.exists()
    assert sample_index_path.exists()
    assert split_path.exists()
    assert stats_path.exists()
    assert failures_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_rows = [json.loads(line) for line in sample_index_path.read_text(encoding="utf-8").splitlines()]
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    failures = [json.loads(line) for line in failures_path.read_text(encoding="utf-8").splitlines()]

    assert manifest["schema_version"] == "catspec.p2_manifest.v0.1"
    assert manifest["sample_count"] == 4
    assert manifest["failure_count"] == 1
    assert len(index_rows) == 4
    assert set(splits["splits"]) == {"train", "val", "test", "all"}
    assert set(splits["splits"]["all"]) == {row["sample_id"] for row in index_rows}
    assert any(row["category"] == "cover_plate" and "unsupported" in row["reason"] for row in failures)

    first = index_rows[0]
    assert "source_mesh" in first
    assert "reference_weld_path" in first
    assert "generated_locus" in first
    assert "weld_meta" in first
    assert "topology" in first
    assert "metrics" in first
    assert Path(first["resolved_spec_path"]).exists()
    assert Path(first["validation_report_path"]).exists()

    cached = preprocess_p2_dataset(DATASET_ROOT, tmp_path / "p2", seed=11, force=False)
    assert cached["cache_hit"] is True
    assert cached["manifest_path"] == result["manifest_path"]
