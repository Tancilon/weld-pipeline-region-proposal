import json
from pathlib import Path

from catspec.autogt import export_autogt_manifest


def test_export_autogt_manifest_writes_required_jsonl_fields(tmp_path):
    manifest = export_autogt_manifest(
        [
            Path("specs/categories/square_tube.yaml"),
            Path("specs/categories/bellmouth.yaml"),
        ],
        output_dir=tmp_path,
    )

    jsonl_path = Path(manifest["jsonl_path"])
    manifest_path = Path(manifest["manifest_path"])
    assert jsonl_path.exists()
    assert manifest_path.exists()
    assert manifest["category_count"] == 2

    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert [row["category"] for row in rows] == ["square_tube", "bellmouth"]
    for row in rows:
        assert {
            "category",
            "source_mesh",
            "reference_weld_path",
            "generated_locus",
            "weld_meta",
            "topology",
            "metrics",
        } <= set(row)
        assert row["source_mesh"].endswith(".obj")
        assert row["reference_weld_path"].endswith("_weld.obj")
        assert isinstance(row["generated_locus"], list)
        assert row["weld_meta"]["weld_type_prior"] == "fillet"
        assert "topology_match" in row["topology"]
        assert "centerline_rmse" in row["metrics"]
        assert "failure_rate" in row["metrics"]
        assert "correct_contact_edge_recall" in row["metrics"]

    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest["jsonl_path"] == str(jsonl_path)
    assert saved_manifest["records"] == rows
