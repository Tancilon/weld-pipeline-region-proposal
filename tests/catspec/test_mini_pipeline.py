import json
from pathlib import Path

from catspec.autogt import export_autogt_manifest
from catspec.mini_pipeline import evaluate_predictions, infer_mini, train_mini


def _manifest(tmp_path: Path) -> Path:
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


def test_train_mini_writes_checkpoint_metrics_and_log(tmp_path):
    manifest_path = _manifest(tmp_path)

    result = train_mini(manifest_path, tmp_path / "train", epochs=80, batch_size=4, seed=7)

    assert Path(result["checkpoint_path"]).exists()
    assert Path(result["metrics_path"]).exists()
    assert Path(result["log_path"]).exists()
    metrics = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    assert metrics["epochs"] == 80
    assert metrics["train_accuracy"]["segment_sequence_accuracy"] >= 0.75


def test_infer_mini_writes_all_modes_predictions(tmp_path):
    manifest_path = _manifest(tmp_path)
    train = train_mini(manifest_path, tmp_path / "train", epochs=80, batch_size=4, seed=7)

    result = infer_mini(manifest_path, train["checkpoint_path"], tmp_path / "infer")

    predictions_path = Path(result["predictions_path"])
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    assert predictions_path.exists()
    assert result["prediction_count"] == 12
    assert {row["mode"] for row in rows} == {"spec_correct", "spec_shuffle", "no_spec"}
    assert all("predicted_topology_name" in row for row in rows)
    assert all("target_segment_sequence_name" in row for row in rows)
    assert all("reference_weld_path" in row["raw"] for row in rows)
    assert all("reference_weld_path" not in row["model_inputs"] for row in rows)


def test_evaluate_predictions_reports_gate(tmp_path):
    manifest_path = _manifest(tmp_path)
    train = train_mini(manifest_path, tmp_path / "train", epochs=120, batch_size=4, seed=7)
    infer = infer_mini(manifest_path, train["checkpoint_path"], tmp_path / "infer")

    report = evaluate_predictions(infer["predictions_path"], tmp_path / "eval")

    assert Path(report["report_path"]).exists()
    assert set(report["modes"]) == {"spec_correct", "spec_shuffle", "no_spec"}
    assert report["modes"]["spec_correct"]["segment_sequence_accuracy"] >= 0.75
    assert report["gate"]["passed"] is True
    assert report["gate"]["metric"] == "segment_sequence_accuracy"
    assert "centerline_rmse" in report["modes"]["spec_correct"]["static_validation_metrics"]
    assert "failure_rate" in report["modes"]["spec_correct"]
    assert "correct_contact_edge_recall" in report["modes"]["spec_correct"]
