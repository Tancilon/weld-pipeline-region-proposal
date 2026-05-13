import json
import subprocess
import sys
from pathlib import Path


def test_catspec_mini_cli_help():
    for script in (
        "scripts/train_catspec_mini.py",
        "scripts/infer_catspec_mini.py",
        "scripts/evaluate_catspec_mini.py",
    ):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0
        assert "--output-dir" in result.stdout


def test_catspec_mini_cli_train_infer_eval_smoke(tmp_path):
    autogt = subprocess.run(
        [
            sys.executable,
            "scripts/export_catspec_autogt.py",
            "--output-dir",
            str(tmp_path / "autogt"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    manifest = json.loads(autogt.stdout)["manifest_path"]

    train = subprocess.run(
        [
            sys.executable,
            "scripts/train_catspec_mini.py",
            "--manifest",
            manifest,
            "--output-dir",
            str(tmp_path / "train"),
            "--epochs",
            "120",
            "--batch-size",
            "4",
            "--seed",
            "7",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    checkpoint = json.loads(train.stdout)["checkpoint_path"]
    assert Path(checkpoint).exists()

    infer = subprocess.run(
        [
            sys.executable,
            "scripts/infer_catspec_mini.py",
            "--manifest",
            manifest,
            "--checkpoint",
            checkpoint,
            "--output-dir",
            str(tmp_path / "infer"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    predictions = json.loads(infer.stdout)["predictions_path"]
    infer_report = json.loads(infer.stdout)["report_path"]
    assert Path(predictions).exists()
    assert Path(infer_report).exists()

    eval_result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_catspec_mini.py",
            "--predictions",
            predictions,
            "--output-dir",
            str(tmp_path / "eval"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    report = json.loads(eval_result.stdout)

    assert Path(report["report_path"]).exists()
    assert report["gate"]["passed"] is True
    assert report["modes"]["spec_correct"]["segment_sequence_accuracy"] > report["modes"]["no_spec"]["segment_sequence_accuracy"]
