"""Training, inference, and evaluation for CatSpec-Pose P2."""

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
from catspec.mini_dataset import PATH_COUNT_CLASSES, SEGMENT_SEQUENCE_CLASSES, TOPOLOGY_CLASSES
from catspec.p2_dataset import CatSpecP2Dataset, GEOMETRY_DIM, collate_p2_batch
from catspec.spec_encoder import DEFAULT_VOCAB_SIZE, SpecEncoder
from catspec.validation import _jsonify


P2_MODES = ["spec_correct", "spec_shuffle", "no_spec"]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _move_targets(targets: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in targets.items()}


def _forward_batch(
    encoder: SpecEncoder,
    head: SpecConditionedLightweightHead,
    batch: dict[str, Any],
    device: torch.device,
    use_geometry: bool,
) -> dict[str, torch.Tensor]:
    token_ids = batch["model_inputs"]["token_ids"].to(device)
    token_mask = batch["model_inputs"]["token_mask"].to(device)
    spec_embedding = encoder(token_ids, token_mask)
    if use_geometry:
        geometry = batch["model_inputs"]["geometry_features"].to(device)
        spec_embedding = torch.cat([spec_embedding, geometry], dim=-1)
    return head(spec_embedding)


def _evaluate_dataset(
    encoder: SpecEncoder,
    head: SpecConditionedLightweightHead,
    dataset: CatSpecP2Dataset,
    batch_size: int,
    device: torch.device,
    use_geometry: bool,
    num_workers: int = 0,
) -> dict[str, float]:
    if len(dataset) == 0:
        return {"topology_accuracy": 0.0, "path_count_accuracy": 0.0, "segment_sequence_accuracy": 0.0}
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_p2_batch,
    )
    totals = {"topology_accuracy": [], "path_count_accuracy": [], "segment_sequence_accuracy": []}
    with torch.no_grad():
        for batch in loader:
            outputs = _forward_batch(encoder, head, batch, device, use_geometry)
            acc = batch_accuracy(outputs, _move_targets(batch["targets"], device))
            for key, value in acc.items():
                totals[key].append(value)
    return {key: float(np.mean(values)) if values else 0.0 for key, values in totals.items()}


def _checkpoint_payload(
    head: SpecConditionedLightweightHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    embedding_dim: int,
    hidden_dim: int,
    max_tokens: int,
    use_geometry: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "catspec.p2_checkpoint.v0.1",
        "epoch": epoch,
        "config": {
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "max_tokens": max_tokens,
            "vocab_size": DEFAULT_VOCAB_SIZE,
            "geometry_dim": GEOMETRY_DIM if use_geometry else 0,
            "use_geometry": use_geometry,
        },
        "classes": {
            "topology": TOPOLOGY_CLASSES,
            "path_count": PATH_COUNT_CLASSES,
            "segment_sequence": SEGMENT_SEQUENCE_CLASSES,
        },
        "head_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def train_p2(
    manifest: str | Path,
    output_dir: str | Path = "results/catspec/p2_train",
    split: str = "train",
    val_split: str = "val",
    epochs: int = 120,
    batch_size: int = 4,
    lr: float = 0.03,
    seed: int = 7,
    device: str = "auto",
    num_workers: int = 0,
    resume_checkpoint: str | Path | None = None,
    embedding_dim: int = 64,
    hidden_dim: int = 64,
    max_tokens: int = 96,
    use_geometry: bool = False,
) -> dict[str, Any]:
    """Train the P2 spec-conditioned lightweight head."""

    _set_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_device = _device(device)

    train_dataset = CatSpecP2Dataset(manifest, split=split, mode="spec_correct", max_tokens=max_tokens)
    val_dataset = CatSpecP2Dataset(manifest, split=val_split, mode="spec_correct", max_tokens=max_tokens)
    if len(train_dataset) == 0:
        raise ValueError(f"empty P2 train dataset for split {split!r}")

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_p2_batch,
    )
    input_dim = embedding_dim + (GEOMETRY_DIM if use_geometry else 0)
    encoder = SpecEncoder(embedding_dim=embedding_dim, max_tokens=max_tokens).to(run_device)
    head = SpecConditionedLightweightHead(embedding_dim=input_dim, hidden_dim=hidden_dim).to(run_device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    start_epoch = 1
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=run_device, weights_only=False)
        head.load_state_dict(checkpoint["head_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    log_lines: list[str] = []
    history: list[dict[str, Any]] = []
    best_val = -1.0
    latest_checkpoint_path = output / "catspec_p2_latest.pt"
    best_checkpoint_path = output / "catspec_p2_best.pt"
    last_loss = 0.0

    for epoch in range(start_epoch, epochs + 1):
        losses = []
        head.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = _forward_batch(encoder, head, batch, run_device, use_geometry)
            loss, _ = mini_head_loss(outputs, _move_targets(batch["targets"], run_device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses)) if losses else 0.0
        head.eval()
        train_accuracy = _evaluate_dataset(encoder, head, train_dataset, batch_size, run_device, use_geometry, num_workers)
        val_accuracy = _evaluate_dataset(encoder, head, val_dataset, batch_size, run_device, use_geometry, num_workers)
        val_score = val_accuracy["segment_sequence_accuracy"]
        history.append({"epoch": epoch, "loss": last_loss, "train_accuracy": train_accuracy, "val_accuracy": val_accuracy})
        checkpoint = _checkpoint_payload(head, optimizer, epoch, embedding_dim, hidden_dim, max_tokens, use_geometry)
        torch.save(checkpoint, latest_checkpoint_path)
        if val_score >= best_val:
            best_val = val_score
            torch.save(checkpoint, best_checkpoint_path)
        if epoch == start_epoch or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            log_lines.append(
                f"epoch={epoch} loss={last_loss:.6f} "
                f"train_segment={train_accuracy['segment_sequence_accuracy']:.4f} "
                f"val_segment={val_accuracy['segment_sequence_accuracy']:.4f}"
            )

    metrics_path = output / "catspec_p2_train_metrics.json"
    log_path = output / "catspec_p2_train.log"
    metrics = {
        "schema_version": "catspec.p2_train_metrics.v0.1",
        "manifest": str(manifest),
        "split": split,
        "val_split": val_split,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "device": str(run_device),
        "num_workers": num_workers,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "use_geometry": use_geometry,
        "final_loss": last_loss,
        "train_accuracy": history[-1]["train_accuracy"],
        "val_accuracy": history[-1]["val_accuracy"],
        "history": history,
        "latest_checkpoint_path": str(latest_checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, default=_jsonify) + "\n", encoding="utf-8")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return {
        "latest_checkpoint_path": str(latest_checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
        "metrics": metrics,
    }


def _load_models(checkpoint_path: str | Path, device: torch.device) -> tuple[SpecEncoder, SpecConditionedLightweightHead, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    encoder = SpecEncoder(
        embedding_dim=int(config["embedding_dim"]),
        max_tokens=int(config["max_tokens"]),
        vocab_size=int(config["vocab_size"]),
    ).to(device)
    input_dim = int(config["embedding_dim"]) + int(config.get("geometry_dim", 0))
    head = SpecConditionedLightweightHead(
        embedding_dim=input_dim,
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    encoder.eval()
    head.eval()
    return encoder, head, checkpoint


def _name(classes: list[Any], label: int) -> Any:
    return classes[int(label)]


def _logits_summary(outputs: dict[str, torch.Tensor], idx: int) -> dict[str, Any]:
    summary = {}
    for key, classes in (
        ("topology", TOPOLOGY_CLASSES),
        ("path_count", PATH_COUNT_CLASSES),
        ("segment_sequence", SEGMENT_SEQUENCE_CLASSES),
    ):
        logits = outputs[f"{key}_logits"][idx].detach().cpu()
        probs = torch.softmax(logits, dim=-1)
        label = int(torch.argmax(probs).item())
        summary[key] = {
            "label": label,
            "name": _name(classes, label),
            "confidence": float(probs[label].item()),
        }
    return summary


def infer_p2(
    manifest: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path = "results/catspec/p2_infer",
    split: str = "test",
    modes: list[str] | None = None,
    batch_size: int = 4,
    device: str = "auto",
    num_workers: int = 0,
) -> dict[str, Any]:
    """Run P2 inference for spec-correct, spec-shuffle, and no-spec modes."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_device = _device(device)
    encoder, head, ckpt = _load_models(checkpoint, run_device)
    max_tokens = int(ckpt["config"]["max_tokens"])
    use_geometry = bool(ckpt["config"].get("use_geometry", False))
    mode_names = modes or list(P2_MODES)
    predictions: list[dict[str, Any]] = []

    for mode in mode_names:
        dataset = CatSpecP2Dataset(manifest, split=split, mode=mode, max_tokens=max_tokens)  # type: ignore[arg-type]
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_p2_batch,
        )
        with torch.no_grad():
            for batch in loader:
                outputs = _forward_batch(encoder, head, batch, run_device, use_geometry)
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
                            "schema_version": "catspec.p2_prediction.v0.1",
                            "sample_id": batch["sample_id"][idx],
                            "split": split,
                            "mode": mode,
                            "category": batch["category"][idx],
                            "prompt_category": batch["prompt_category"][idx],
                            "source_mesh": batch["source_mesh"][idx],
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
                                "geometry_dim": int(batch["model_inputs"]["geometry_features"][idx].numel()) if use_geometry else 0,
                            },
                            "logits_summary": _logits_summary(outputs, idx),
                            "target_labels": target,
                            "predicted_labels": predicted,
                            "static_validation_metrics": raw["metrics"],
                            "raw": raw,
                        }
                    )

    predictions_path = output / "catspec_p2_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as fh:
        for row in predictions:
            fh.write(json.dumps(row, ensure_ascii=False, default=_jsonify) + "\n")
    return {
        "predictions_path": str(predictions_path),
        "prediction_count": len(predictions),
        "modes": mode_names,
        "split": split,
    }


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row["static_validation_metrics"].get(key) for row in rows]
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return float(np.mean(finite))


def _metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {
            "topology_accuracy": 0.0,
            "path_count_accuracy": 0.0,
            "segment_sequence_accuracy": 0.0,
            "centerline_rmse": None,
            "hausdorff": None,
            "failure_rate": 1.0,
            "correct_contact_edge_recall": 0.0,
            "static_validation_metrics": {},
            "count": 0,
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


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def _read_preprocess_failures(manifest: str | Path | None) -> list[dict[str, Any]]:
    if manifest is None:
        return []
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    failures_path = Path(data["failures_path"])
    if not failures_path.exists():
        failures_path = Path(manifest).parent / data["failures_path"]
    if not failures_path.exists():
        return []
    return [json.loads(line) for line in failures_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_p2_predictions(
    predictions: str | Path,
    output_dir: str | Path = "results/catspec/p2_eval",
    manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate P2 predictions and compare spec modes."""

    rows = [json.loads(line) for line in Path(predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_mode = _group_by(rows, "mode")
    aggregate = {mode: _metrics_for_rows(by_mode.get(mode, [])) for mode in sorted(set(P2_MODES) | set(by_mode))}

    per_category: dict[str, dict[str, Any]] = {}
    for category, category_rows in sorted(_group_by(rows, "category").items()):
        by_category_mode = _group_by(category_rows, "mode")
        per_category[category] = {
            mode: _metrics_for_rows(by_category_mode.get(mode, []))
            for mode in sorted(set(P2_MODES) | set(by_category_mode))
        }

    failure_samples = [
        {
            "sample_id": row["sample_id"],
            "category": row["category"],
            "mode": row["mode"],
            "reason": ",".join(name for name, ok in row["correct"].items() if not ok),
        }
        for row in rows
        if not all(row["correct"].values())
    ]

    correct = aggregate.get("spec_correct", {}).get("segment_sequence_accuracy", 0.0)
    shuffle = aggregate.get("spec_shuffle", {}).get("segment_sequence_accuracy", 0.0)
    no_spec = aggregate.get("no_spec", {}).get("segment_sequence_accuracy", 0.0)
    passed = bool(correct > shuffle and correct > no_spec)
    reason = (
        "spec_correct segment_sequence_accuracy is higher than spec_shuffle and no_spec"
        if passed
        else "spec_correct did not beat both spec_shuffle and no_spec on segment_sequence_accuracy"
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "catspec_p2_eval_report.json"
    report = {
        "schema_version": "catspec.p2_eval.v0.1",
        "predictions_path": str(predictions),
        "manifest": str(manifest) if manifest else None,
        "report_path": str(report_path),
        "aggregate": aggregate,
        "per_category": per_category,
        "failure_samples": failure_samples,
        "preprocess_failures": _read_preprocess_failures(manifest),
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
