import json
from pathlib import Path

from catspec.p2_pipeline import evaluate_p2_predictions, infer_p2, train_p2
from catspec.p2_preprocess import preprocess_p2_dataset


DATASET_ROOT = Path("/home/dq/mnt/localgit/aiws5.2/datasets/obj_share_models")


def _manifest(tmp_path: Path) -> Path:
    result = preprocess_p2_dataset(DATASET_ROOT, tmp_path / "p2_preprocess", seed=7, force=True)
    return Path(result["manifest_path"])


def test_train_p2_writes_latest_best_metrics_and_log(tmp_path):
    manifest_path = _manifest(tmp_path)

    result = train_p2(
        manifest_path,
        tmp_path / "p2_train",
        split="all",
        val_split="all",
        epochs=120,
        batch_size=4,
        seed=7,
        lr=0.03,
        device="cpu",
    )

    assert Path(result["latest_checkpoint_path"]).exists()
    assert Path(result["best_checkpoint_path"]).exists()
    assert Path(result["metrics_path"]).exists()
    assert Path(result["log_path"]).exists()
    metrics = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    assert metrics["train_accuracy"]["segment_sequence_accuracy"] >= 0.75
    assert metrics["val_accuracy"]["segment_sequence_accuracy"] >= 0.75


def test_train_p2_can_resume_from_latest_checkpoint(tmp_path):
    manifest_path = _manifest(tmp_path)
    first = train_p2(
        manifest_path,
        tmp_path / "p2_train",
        split="all",
        val_split="all",
        epochs=1,
        batch_size=4,
        seed=7,
        lr=0.03,
        device="cpu",
    )

    resumed = train_p2(
        manifest_path,
        tmp_path / "p2_train_resumed",
        split="all",
        val_split="all",
        epochs=2,
        batch_size=4,
        seed=7,
        lr=0.03,
        device="cpu",
        resume_checkpoint=first["latest_checkpoint_path"],
    )

    metrics = json.loads(Path(resumed["metrics_path"]).read_text(encoding="utf-8"))
    assert metrics["resume_checkpoint"] == first["latest_checkpoint_path"]
    assert metrics["history"][0]["epoch"] == 2


def test_infer_and_evaluate_p2_predictions_report_gate_and_failures(tmp_path):
    manifest_path = _manifest(tmp_path)
    train = train_p2(
        manifest_path,
        tmp_path / "p2_train",
        split="all",
        val_split="all",
        epochs=120,
        batch_size=4,
        seed=7,
        lr=0.03,
        device="cpu",
    )

    infer = infer_p2(
        manifest_path,
        train["best_checkpoint_path"],
        tmp_path / "p2_infer",
        split="all",
        batch_size=4,
        device="cpu",
    )
    rows = [json.loads(line) for line in Path(infer["predictions_path"]).read_text(encoding="utf-8").splitlines()]
    assert infer["prediction_count"] == 12
    assert {row["mode"] for row in rows} == {"spec_correct", "spec_shuffle", "no_spec"}
    assert all("source_mesh" in row for row in rows)
    assert all("logits_summary" in row for row in rows)
    assert all("reference_weld_path" not in row["model_inputs"] for row in rows)

    report = evaluate_p2_predictions(infer["predictions_path"], tmp_path / "p2_eval", manifest=manifest_path)

    assert Path(report["report_path"]).exists()
    assert report["gate"]["passed"] is True
    assert report["aggregate"]["spec_correct"]["segment_sequence_accuracy"] >= 0.75
    assert set(report["per_category"]) >= {"square_tube", "channel_steel", "H_beam", "bellmouth"}
    assert any(row["category"] == "cover_plate" for row in report["preprocess_failures"])
    assert "failure_samples" in report
