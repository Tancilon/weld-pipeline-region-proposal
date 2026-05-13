import json
from pathlib import Path

from catspec.shuffle import evaluate_spec_shuffle


def test_evaluate_spec_shuffle_writes_report_with_correct_and_shuffled_records(tmp_path):
    report = evaluate_spec_shuffle(
        [
            Path("specs/categories/square_tube.yaml"),
            Path("specs/categories/bellmouth.yaml"),
        ],
        output_dir=tmp_path,
    )

    assert Path(report["report_path"]).exists()
    assert report["target_count"] == 2
    assert report["aggregate"]["correct_count"] == 2
    assert report["aggregate"]["shuffle_count"] == 2
    assert "failure_rate" in report["aggregate"]
    assert "correct_contact_edge_recall" in report["aggregate"]

    correct = [record for record in report["records"] if record["mode"] == "spec_correct"]
    shuffled = [record for record in report["records"] if record["mode"] == "spec_shuffle"]
    assert all(record["is_spec_correct"] is True for record in correct)
    assert all(record["topology_match"] is True for record in correct)
    assert all(record["metrics"]["failure_rate"] == 0.0 for record in correct)
    assert all(record["is_spec_correct"] is False for record in shuffled)
    assert any(record["topology_match"] is False for record in shuffled)
    assert all("centerline_rmse" in record["metrics"] for record in report["records"])
    assert all(Path(record["overlay_path"]).exists() for record in report["records"] if record.get("overlay_path"))

    saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert saved["aggregate"] == report["aggregate"]
