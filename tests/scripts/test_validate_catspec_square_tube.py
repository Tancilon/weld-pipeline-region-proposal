import subprocess
import sys


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
