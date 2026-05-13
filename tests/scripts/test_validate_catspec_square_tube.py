import json
import subprocess
import sys
from pathlib import Path


def test_validate_catspec_square_tube_help():
    result = subprocess.run(
        [sys.executable, "scripts/validate_catspec_square_tube.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--spec" in result.stdout
    assert "--output-dir" in result.stdout


def test_validate_catspec_square_tube_runs_from_repo_root(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_catspec_square_tube.py",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert set(output) == {
        "category",
        "topology_match",
        "metrics",
        "report_path",
        "overlay_path",
    }
    assert output["category"] == "square_tube"
    assert Path(output["report_path"]).exists()
    assert Path(output["overlay_path"]).exists()
