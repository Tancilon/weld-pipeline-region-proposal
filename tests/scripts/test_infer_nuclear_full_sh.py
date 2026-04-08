import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "infer_nuclear_full.sh"


def test_infer_nuclear_full_script_passes_expected_arguments(tmp_path):
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
            "NUCLEAR_DATA_PATH": "./data/aiws5.2-dataset",
            "SEG_CKPT": "./results/ckpts/SegNet/best.pth",
            "ENERGY_CKPT": "./results/ckpts/EnergyNet/energynet.pth",
            "SCALE_CKPT": "./results/ckpts/ScaleNet/scalenet.pth",
            "SPLIT": "val",
            "OUTPUT_DIR": "./results/full_pipeline_segnet_best",
            "NUM_VIS": "20",
            "SCORE_THRESHOLD": "0.5",
            "REPEAT_NUM": "10",
            "NUM_POINTS": "1024",
            "IMG_SIZE": "224",
            "CUDA_VISIBLE_DEVICES": "2",
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
    assert call["cuda_visible_devices"] == "2"
    assert call["argv"] == [
        "runners/infer_nuclear_full.py",
        "--nuclear_data_path", "./data/aiws5.2-dataset",
        "--seg_ckpt", "./results/ckpts/SegNet/best.pth",
        "--energy_ckpt", "./results/ckpts/EnergyNet/energynet.pth",
        "--scale_ckpt", "./results/ckpts/ScaleNet/scalenet.pth",
        "--split", "val",
        "--output_dir", "./results/full_pipeline_segnet_best",
        "--num_vis", "20",
        "--score_threshold", "0.5",
        "--repeat_num", "10",
        "--num_points", "1024",
        "--img_size", "224",
    ]
