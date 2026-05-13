import json
import subprocess
import sys
from pathlib import Path


DATASET_ROOT = Path("/home/dq/mnt/localgit/aiws5.2/datasets/obj_share_models")


def test_catspec_p2_cli_help():
    help_outputs = {}
    for script in (
        "scripts/preprocess_catspec_p2.py",
        "scripts/train_catspec_p2.py",
        "scripts/infer_catspec_p2.py",
        "scripts/evaluate_catspec_p2.py",
    ):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0
        assert "--output-dir" in result.stdout
        help_outputs[script] = result.stdout

    assert "--dataset-root" in help_outputs["scripts/preprocess_catspec_p2.py"]
    assert "--dataset-root" in help_outputs["scripts/train_catspec_p2.py"]


def test_catspec_p2_cli_preprocess_train_infer_eval_smoke(tmp_path):
    preprocess = subprocess.run(
        [
            sys.executable,
            "scripts/preprocess_catspec_p2.py",
            "--dataset-root",
            str(DATASET_ROOT),
            "--output-dir",
            str(tmp_path / "p2_preprocess"),
            "--force",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    manifest = json.loads(preprocess.stdout)["manifest_path"]

    train = subprocess.run(
        [
            sys.executable,
            "scripts/train_catspec_p2.py",
            "--manifest",
            manifest,
            "--output-dir",
            str(tmp_path / "p2_train"),
            "--split",
            "all",
            "--val-split",
            "all",
            "--epochs",
            "120",
            "--batch-size",
            "4",
            "--seed",
            "7",
            "--device",
            "cpu",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    checkpoint = json.loads(train.stdout)["best_checkpoint_path"]
    assert Path(checkpoint).exists()

    infer = subprocess.run(
        [
            sys.executable,
            "scripts/infer_catspec_p2.py",
            "--manifest",
            manifest,
            "--checkpoint",
            checkpoint,
            "--output-dir",
            str(tmp_path / "p2_infer"),
            "--split",
            "all",
            "--device",
            "cpu",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    predictions = json.loads(infer.stdout)["predictions_path"]
    assert Path(predictions).exists()

    eval_result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_catspec_p2.py",
            "--predictions",
            predictions,
            "--manifest",
            manifest,
            "--output-dir",
            str(tmp_path / "p2_eval"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    report = json.loads(eval_result.stdout)

    assert Path(report["report_path"]).exists()
    assert report["gate"]["passed"] is True
    assert report["aggregate"]["spec_correct"]["segment_sequence_accuracy"] > report["aggregate"]["no_spec"]["segment_sequence_accuracy"]
