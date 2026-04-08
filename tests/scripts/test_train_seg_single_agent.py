import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "train_seg_single_agent.sh"


def test_train_seg_single_agent_runs_preflight_before_trainer(tmp_path):
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
            "EXP_NAME": "SegNetScriptTest",
            "NUCLEAR_DATA_PATH": "/tmp/nuclear-dataset",
            "POSE_INIT_CKPT": "/tmp/pose-init.pth",
            "WANDB_MODE": "offline",
            "WANDB_PROJECT": "nuclear-seg-single-agent",
            "WANDB_ENTITY": "demo-entity",
            "WANDB_RUN_NAME": "demo-run",
            "BATCH_SIZE": "2",
            "N_EPOCHS": "3",
            "NUM_WORKERS": "0",
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
    assert len(calls) == 2

    preflight_call, trainer_call = calls
    assert preflight_call["argv"] == [
        "scripts/check_nuclear_seg_dataset.py",
        "--nuclear_data_path",
        "/tmp/nuclear-dataset",
    ]
    assert preflight_call["cuda_visible_devices"] == "3"

    trainer_args = trainer_call["argv"]
    assert trainer_args[0] == "runners/trainer.py"
    assert "--dataset_type" in trainer_args
    assert "--nuclear_data_path" in trainer_args
    assert "--pretrained_score_model_path" in trainer_args
    assert "--wandb_mode" in trainer_args
    assert "--wandb_project" in trainer_args
    assert "--wandb_entity" in trainer_args
    assert "--wandb_run_name" in trainer_args
    assert trainer_call["cuda_visible_devices"] == "3"
