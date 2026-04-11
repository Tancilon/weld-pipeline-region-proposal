import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "visualize_seg_single_agent.sh"


def test_visualize_seg_single_agent_script_passes_expected_arguments(tmp_path):
    log_path = tmp_path / "python_calls.jsonl"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

log_path = Path(os.environ["FAKE_PYTHON_LOG"])
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "argv": sys.argv[1:],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }) + "\\n")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{tmp_path}:{env['PATH']}",
            "FAKE_PYTHON_LOG": str(log_path),
            "NUCLEAR_DATA_PATH": "./data/aiws5.2-dataset-v1-aug",
            "SEG_CKPT": "./results/ckpts/SegNet/best.pth",
            "SPLIT": "train",
            "OUTPUT_DIR": "./results/vis_seg_focus",
            "NUM_VIS": "7",
            "SCORE_THRESHOLD": "0.6",
            "IMAGE_NAMES": "foo.png,bar.png",
            "IMG_SIZE": "224",
            "CUDA_VISIBLE_DEVICES": "3",
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr

    calls = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(calls) == 1

    call = calls[0]
    assert call["cuda_visible_devices"] == "3"
    assert call["argv"] == [
        "runners/visualize_seg_single_agent.py",
        "--nuclear_data_path", "./data/aiws5.2-dataset-v1-aug",
        "--seg_ckpt", "./results/ckpts/SegNet/best.pth",
        "--split", "train",
        "--output_dir", "./results/vis_seg_focus",
        "--num_vis", "7",
        "--score_threshold", "0.6",
        "--image_names", "foo.png,bar.png",
        "--img_size", "224",
    ]
