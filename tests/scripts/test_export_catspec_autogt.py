import json
import subprocess
import sys
from pathlib import Path


def test_export_catspec_autogt_help():
    result = subprocess.run(
        [sys.executable, "scripts/export_catspec_autogt.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--spec" in result.stdout
    assert "--output-dir" in result.stdout


def test_export_catspec_autogt_runs_from_repo_root(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_catspec_autogt.py",
            "--spec",
            "specs/categories/square_tube.yaml",
            "--spec",
            "specs/categories/bellmouth.yaml",
            "--output-dir",
            str(tmp_path / "autogt"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["category_count"] == 2
    assert Path(output["jsonl_path"]).exists()
    assert Path(output["manifest_path"]).exists()
