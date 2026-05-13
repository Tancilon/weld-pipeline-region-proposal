"""Training, inference, and evaluation for the CatSpec-Pose mini loop."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from catspec.lightweight_head import (
    SpecConditionedLightweightHead,
    batch_accuracy,
    logits_to_predictions,
    mini_head_loss,
)
from catspec.mini_dataset import (
    PATH_COUNT_CLASSES,
    SEGMENT_SEQUENCE_CLASSES,
    TOPOLOGY_CLASSES,
    CatSpecMiniDataset,
    collate_mini_batch,
)
from catspec.spec_encoder import DEFAULT_VOCAB_SIZE, SpecEncoder
from catspec.validation import _jsonify


DEFAULT_MODES = ["spec_correct", "spec_shuffle", "no_spec"]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _move_targets(targets: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in targets.items()}


def _forward_batch(
    encoder: SpecEncoder,
    head: SpecConditionedLightweightHead,
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    token_ids = batch["model_inputs"]["token_ids"].to(device)
    token_mask = batch["model_inputs"]["token_mask"].to(device)
    spec_embedding = encoder(token_ids, token_mask)
    return head(spec_embedding)


def _evaluate_dataset(
    encoder: SpecEncoder,
    head: SpecConditionedLightweightHead,
    dataset: CatSpecMiniDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_mini_batch)
    totals = {"topology_accuracy": [], "path_count_accuracy": [], "segment_sequence_accuracy": []}
    with torch.no_grad():
        for batch in loader:
            outputs = _forward_batch(encoder, head, batch, device)
            acc = batch_accuracy(outputs, _move_targets(batch["targets"], device))
            for key, value in acc.items():
                totals[key].append(value)
    return {key: float(np.mean(values)) if values else 0.0 for key, values in totals.items()}


def train_mini(
    manifest: str | Path,
    output_dir: str | Path,
    epochs: int = 120,
    batch_size: int = 4,
    seed: int = 7,
    lr: float = 0.03,
    embedding_dim: int = 64,
    hidden_dim: int = 64,
    max_tokens: int = 96,
) -> dict[str, Any]:
    """Train the mini spec-conditioned head on spec-correct prompts."""

    _set_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    dataset = CatSpecMiniDataset(manifest, mode="spec_correct", max_tokens=max_tokens)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_mini_batch)

    encoder = SpecEncoder(embedding_dim=embedding_dim, max_tokens=max_tokens).to(device)
    head = SpecConditionedLightweightHead(embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    log_lines: list[str] = []
    last_loss = 0.0
    for epoch in range(1, epochs + 1):
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = _forward_batch(encoder, head, batch, device)
            loss, _ = mini_head_loss(outputs, _move_targets(batch["targets"], device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses)) if losses else 0.0
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            log_lines.append(f"epoch={epoch} loss={last_loss:.6f}")

    train_accuracy = _evaluate_dataset(encoder, head, dataset, batch_size, device)
    checkpoint_path = output / "catspec_mini_checkpoint.pt"
    metrics_path = output / "catspec_mini_train_metrics.json"
    log_path = output / "catspec_mini_train.log"
    checkpoint = {
        "schema_version": "catspec.mini_checkpoint.v0.1",
        "config": {
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "max_tokens": max_tokens,
            "vocab_size": DEFAULT_VOCAB_SIZE,
        },
        "classes": {
            "topology": TOPOLOGY_CLASSES,
            "path_count": PATH_COUNT_CLASSES,
            "segment_sequence": SEGMENT_SEQUENCE_CLASSES,
        },
        "head_state_dict": head.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)

    metrics = {
        "schema_version": "catspec.mini_train_metrics.v0.1",
        "manifest": str(manifest),
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "final_loss": last_loss,
        "train_accuracy": train_accuracy,
        "checkpoint_path": str(checkpoint_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, default=_jsonify) + "\n", encoding="utf-8")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return {
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
        "metrics": metrics,
    }


def _load_models(checkpoint_path: str | Path) -> tuple[SpecEncoder, SpecConditionedLightweightHead, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    encoder = SpecEncoder(
        embedding_dim=int(config["embedding_dim"]),
        max_tokens=int(config["max_tokens"]),
        vocab_size=int(config["vocab_size"]),
    )
    head = SpecConditionedLightweightHead(
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    head.load_state_dict(checkpoint["head_state_dict"])
    encoder.eval()
    head.eval()
    return encoder, head, checkpoint


def _name(classes: list[Any], label: int) -> Any:
    return classes[int(label)]


def infer_mini(
    manifest: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    modes: list[str] | None = None,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Run mini inference for spec-correct, spec-shuffle, and no-spec modes."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    encoder, head, ckpt = _load_models(checkpoint)
    max_tokens = int(ckpt["config"]["max_tokens"])
    mode_names = modes or list(DEFAULT_MODES)
    predictions: list[dict[str, Any]] = []

    for mode in mode_names:
        dataset = CatSpecMiniDataset(manifest, mode=mode, max_tokens=max_tokens)  # type: ignore[arg-type]
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_mini_batch)
        with torch.no_grad():
            for batch in loader:
                outputs = _forward_batch(encoder, head, batch, torch.device("cpu"))
                pred = logits_to_predictions(outputs)
                for idx, raw in enumerate(batch["raw"]):
                    target = {
                        "topology_label": int(batch["targets"]["topology_label"][idx]),
                        "path_count_label": int(batch["targets"]["path_count_label"][idx]),
                        "segment_sequence_label": int(batch["targets"]["segment_sequence_label"][idx]),
                    }
                    predicted = {
                        "topology_label": int(pred["topology_label"][idx]),
                        "path_count_label": int(pred["path_count_label"][idx]),
                        "segment_sequence_label": int(pred["segment_sequence_label"][idx]),
                    }
                    predictions.append(
                        {
                            "mode": mode,
                            "category": batch["category"][idx],
                            "prompt_category": batch["prompt_category"][idx],
                            "target_topology_name": _name(TOPOLOGY_CLASSES, target["topology_label"]),
                            "predicted_topology_name": _name(TOPOLOGY_CLASSES, predicted["topology_label"]),
                            "target_path_count": _name(PATH_COUNT_CLASSES, target["path_count_label"]),
                            "predicted_path_count": _name(PATH_COUNT_CLASSES, predicted["path_count_label"]),
                            "target_segment_sequence_name": _name(
                                SEGMENT_SEQUENCE_CLASSES,
                                target["segment_sequence_label"],
                            ),
                            "predicted_segment_sequence_name": _name(
                                SEGMENT_SEQUENCE_CLASSES,
                                predicted["segment_sequence_label"],
                            ),
                            "correct": {
                                "topology": predicted["topology_label"] == target["topology_label"],
                                "path_count": predicted["path_count_label"] == target["path_count_label"],
                                "segment_sequence": predicted["segment_sequence_label"] == target["segment_sequence_label"],
                            },
                            "model_inputs": {
                                "token_count": int(batch["model_inputs"]["token_mask"][idx].sum().item()),
                            },
                            "target_labels": target,
                            "predicted_labels": predicted,
                            "static_validation_metrics": raw["metrics"],
                            "raw": raw,
                        }
                    )

    predictions_path = output / "catspec_mini_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as fh:
        for row in predictions:
            fh.write(json.dumps(row, ensure_ascii=False, default=_jsonify) + "\n")
    return {
        "predictions_path": str(predictions_path),
        "prediction_count": len(predictions),
        "modes": mode_names,
    }


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row["static_validation_metrics"].get(key) for row in rows]
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return float(np.mean(finite))


def _mode_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {
            "topology_accuracy": 0.0,
            "path_count_accuracy": 0.0,
            "segment_sequence_accuracy": 0.0,
            "failure_rate": 1.0,
            "correct_contact_edge_recall": 0.0,
            "static_validation_metrics": {},
        }
    topology_accuracy = float(np.mean([row["correct"]["topology"] for row in rows]))
    path_count_accuracy = float(np.mean([row["correct"]["path_count"] for row in rows]))
    segment_sequence_accuracy = float(np.mean([row["correct"]["segment_sequence"] for row in rows]))
    static = {
        "centerline_rmse": _mean_metric(rows, "centerline_rmse"),
        "hausdorff": _mean_metric(rows, "hausdorff"),
        "failure_rate": _mean_metric(rows, "failure_rate"),
        "correct_contact_edge_recall": _mean_metric(rows, "correct_contact_edge_recall"),
    }
    return {
        "topology_accuracy": topology_accuracy,
        "path_count_accuracy": path_count_accuracy,
        "segment_sequence_accuracy": segment_sequence_accuracy,
        "centerline_rmse": static["centerline_rmse"],
        "hausdorff": static["hausdorff"],
        "failure_rate": 1.0 - segment_sequence_accuracy,
        "correct_contact_edge_recall": segment_sequence_accuracy,
        "static_validation_metrics": static,
        "count": count,
    }


def evaluate_predictions(predictions: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Evaluate mini predictions and compare prompt modes."""

    rows = [json.loads(line) for line in Path(predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    mode_metrics = {mode: _mode_metrics(mode_rows) for mode, mode_rows in sorted(by_mode.items())}

    correct = mode_metrics.get("spec_correct", {}).get("segment_sequence_accuracy", 0.0)
    shuffle = mode_metrics.get("spec_shuffle", {}).get("segment_sequence_accuracy", 0.0)
    no_spec = mode_metrics.get("no_spec", {}).get("segment_sequence_accuracy", 0.0)
    passed = bool(correct > shuffle and correct > no_spec)
    if passed:
        reason = "spec_correct segment_sequence_accuracy is higher than spec_shuffle and no_spec"
    else:
        reason = "spec_correct did not beat both spec_shuffle and no_spec on segment_sequence_accuracy"

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "catspec_mini_eval_report.json"
    report = {
        "schema_version": "catspec.mini_eval.v0.1",
        "predictions_path": str(predictions),
        "report_path": str(report_path),
        "modes": mode_metrics,
        "gate": {
            "passed": passed,
            "metric": "segment_sequence_accuracy",
            "reason": reason,
            "spec_correct": correct,
            "spec_shuffle": shuffle,
            "no_spec": no_spec,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, default=_jsonify) + "\n", encoding="utf-8")
    return json.loads(json.dumps(report, default=_jsonify))
