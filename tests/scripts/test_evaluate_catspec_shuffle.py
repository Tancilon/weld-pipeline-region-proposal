import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_catspec_shuffle_help():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_catspec_shuffle.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--spec" in result.stdout
    assert "--output-dir" in result.stdout


def test_evaluate_catspec_shuffle_runs_from_repo_root(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_catspec_shuffle.py",
            "--spec",
            "specs/categories/square_tube.yaml",
            "--spec",
            "specs/categories/bellmouth.yaml",
            "--output-dir",
            str(tmp_path / "shuffle"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["target_count"] == 2
    assert output["aggregate"]["correct_count"] == 2
    assert output["aggregate"]["shuffle_count"] == 2
    assert Path(output["report_path"]).exists()
