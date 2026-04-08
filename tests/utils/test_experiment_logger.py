from argparse import Namespace
from pathlib import Path
import json
import sys
import types

import pytest


def _install_fake_wandb(monkeypatch):
    calls = {}

    class FakeRun:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.logged = []
            self.finished = False

        def log(self, data, step=None, commit=True):
            self.logged.append((data, step, commit))

        def finish(self):
            self.finished = True

    def init(**kwargs):
        calls["init"] = kwargs
        run = FakeRun(**kwargs)
        calls["run"] = run
        return run

    fake_wandb = types.SimpleNamespace(init=init)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    return calls


def test_build_experiment_logger_initializes_wandb_with_expected_mode(monkeypatch, tmp_path):
    calls = _install_fake_wandb(monkeypatch)

    from utils.experiment_logger import WandbLogger, build_experiment_logger

    cfg = Namespace(
        wandb_mode="offline",
        wandb_project="demo-project",
        wandb_entity="demo-entity",
        wandb_run_name="demo-run",
    )

    logger = build_experiment_logger(cfg, tmp_path)

    assert isinstance(logger, WandbLogger)
    assert calls["init"]["mode"] == "offline"
    assert calls["init"]["project"] == "demo-project"
    assert calls["init"]["entity"] == "demo-entity"
    assert calls["init"]["name"] == "demo-run"
    assert calls["init"]["dir"] == str(tmp_path)


def test_build_experiment_logger_disabled_returns_noop(monkeypatch, tmp_path):
    sentinel = object()
    monkeypatch.setitem(sys.modules, "wandb", sentinel)

    from utils.experiment_logger import NoOpLogger, build_experiment_logger

    cfg = Namespace(
        wandb_mode="disabled",
        wandb_project="demo-project",
        wandb_entity="demo-entity",
        wandb_run_name="demo-run",
    )

    logger = build_experiment_logger(cfg, tmp_path)

    assert isinstance(logger, NoOpLogger)


def test_write_config_snapshot_serializes_namespace_to_config_json(tmp_path):
    from utils.experiment_logger import write_config_snapshot

    cfg = Namespace(
        alpha=1,
        beta="two",
        nested=Namespace(gamma=True),
        mapping={"delta": 4},
    )

    write_config_snapshot(cfg, tmp_path)

    config_path = tmp_path / "config.json"
    assert config_path.exists()
    payload = json.loads(config_path.read_text())
    assert payload == {
        "alpha": 1,
        "beta": "two",
        "nested": {"gamma": True},
        "mapping": {"delta": 4},
    }


def test_update_summary_json_merges_best_and_latest_fields(tmp_path):
    from utils.experiment_logger import update_summary_json

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"experiment_name": "existing", "best_epoch": 1}))

    update_summary_json(
        tmp_path,
        {
            "best_epoch": 3,
            "latest_epoch": 4,
            "best_mask_iou": 0.72,
        },
    )

    payload = json.loads(summary_path.read_text())
    assert payload == {
        "experiment_name": "existing",
        "best_epoch": 3,
        "latest_epoch": 4,
        "best_mask_iou": 0.72,
    }


def test_logger_surface_delegates_to_wandb_run(monkeypatch, tmp_path):
    calls = _install_fake_wandb(monkeypatch)

    from utils.experiment_logger import build_experiment_logger

    cfg = Namespace(
        wandb_mode="online",
        wandb_project="demo-project",
        wandb_entity="demo-entity",
        wandb_run_name="demo-run",
    )

    logger = build_experiment_logger(cfg, tmp_path)
    logger.add_scalar("train/loss", 1.25, 7)
    logger.add_scalars("train", {"loss": 1.25, "acc": 0.9}, 7)
    logger.add_image("train/image", "fake-image", 7)
    logger.finish()

    run = calls["run"]
    assert run.logged == [
        ({"train/loss": 1.25}, 7, True),
        ({"train/loss": 1.25, "train/acc": 0.9}, 7, True),
        ({"train/image": "fake-image"}, 7, True),
    ]
    assert run.finished is True
