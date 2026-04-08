import json
from argparse import Namespace
from pathlib import Path
from typing import Any


class NoOpLogger:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_scalars(self, *args, **kwargs):
        return None

    def add_image(self, *args, **kwargs):
        return None

    def finish(self):
        return None


class WandbLogger:
    def __init__(self, run):
        self._run = run

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        payload = {tag: scalar_value}
        self._run.log(payload, step=global_step, commit=True)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, *args, **kwargs):
        payload = {}
        for tag, value in tag_scalar_dict.items():
            if main_tag:
                payload[f"{main_tag}/{tag}"] = value
            else:
                payload[tag] = value
        self._run.log(payload, step=global_step, commit=True)

    def add_image(self, tag, img_tensor, global_step=None, *args, **kwargs):
        self._run.log({tag: img_tensor}, step=global_step, commit=True)

    def finish(self):
        if hasattr(self._run, "finish"):
            self._run.finish()


def _to_jsonable(value: Any):
    if isinstance(value, Namespace):
        return {key: _to_jsonable(val) for key, val in vars(value).items()}
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_config_snapshot(cfg, ckpt_dir):
    ckpt_path = Path(ckpt_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    config_path = ckpt_path / "config.json"
    payload = _to_jsonable(cfg)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return config_path


def update_summary_json(ckpt_dir, summary_update):
    ckpt_path = Path(ckpt_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    summary_path = ckpt_path / "summary.json"

    summary = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            summary.update(loaded)

    summary.update(_to_jsonable(summary_update))

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary_path


def build_experiment_logger(cfg, ckpt_dir):
    mode = getattr(cfg, "wandb_mode", "online")
    if mode == "disabled":
        return NoOpLogger()

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is required unless wandb_mode is disabled") from exc

    init_kwargs = {
        "mode": mode,
        "project": getattr(cfg, "wandb_project", ""),
        "entity": getattr(cfg, "wandb_entity", "") or None,
        "name": getattr(cfg, "wandb_run_name", "") or None,
        "dir": str(ckpt_dir),
        "config": _to_jsonable(cfg),
    }
    init_kwargs = {key: value for key, value in init_kwargs.items() if value not in ("", None)}
    run = wandb.init(**init_kwargs)
    return WandbLogger(run)
