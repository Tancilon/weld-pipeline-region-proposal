"""Dataset adapter for the CatSpec-Pose mini train/infer loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

from catspec.spec_encoder import batch_encode_specs


TOPOLOGY_CLASSES = [
    "closed_rounded_rect",
    "open_line_arc_line_arc_line",
    "parallel_open_lines",
]
PATH_COUNT_CLASSES = [1, 2]
SEGMENT_SEQUENCE_CLASSES = [
    "line_arc_line_arc_line_arc_line_arc",
    "line_arc_line_arc_line__line_arc_line_arc_line",
    "line__line",
]

TOPOLOGY_TO_LABEL = {name: idx for idx, name in enumerate(TOPOLOGY_CLASSES)}
PATH_COUNT_TO_LABEL = {count: idx for idx, count in enumerate(PATH_COUNT_CLASSES)}
SEGMENT_SEQUENCE_TO_LABEL = {name: idx for idx, name in enumerate(SEGMENT_SEQUENCE_CLASSES)}

DatasetMode = Literal["spec_correct", "spec_shuffle", "no_spec"]


def load_autogt_rows(manifest_or_jsonl: str | Path) -> list[dict[str, Any]]:
    """Load CatSpec v0.2 auto-GT records from a manifest JSON or JSONL file."""

    path = Path(manifest_or_jsonl)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if "records" in manifest:
        return list(manifest["records"])
    jsonl_path = Path(manifest["jsonl_path"])
    return load_autogt_rows(jsonl_path)


def _path_segment_sequence(locus: dict[str, Any]) -> str:
    return "_".join(segment["type"] for segment in locus["segments"])


def _segment_sequence_name(row: dict[str, Any]) -> str:
    return "__".join(_path_segment_sequence(locus) for locus in row["generated_locus"])


def _topology_name(row: dict[str, Any]) -> str:
    sequence = _segment_sequence_name(row)
    path_count = len(row["generated_locus"])
    if path_count == 1 and sequence == "line_arc_line_arc_line_arc_line_arc":
        return "closed_rounded_rect"
    if path_count == 2 and sequence == "line_arc_line_arc_line__line_arc_line_arc_line":
        return "open_line_arc_line_arc_line"
    if path_count == 2 and sequence == "line__line":
        return "parallel_open_lines"
    raise ValueError(f"unsupported mini topology sequence: {sequence}")


def derive_targets(row: dict[str, Any]) -> dict[str, Any]:
    """Derive lightweight-head targets from one auto-GT row."""

    topology_name = _topology_name(row)
    path_count = len(row["generated_locus"])
    segment_sequence_name = _segment_sequence_name(row)
    return {
        "topology_name": topology_name,
        "path_count": path_count,
        "segment_sequence_name": segment_sequence_name,
        "topology_label": TOPOLOGY_TO_LABEL[topology_name],
        "path_count_label": PATH_COUNT_TO_LABEL[path_count],
        "segment_sequence_label": SEGMENT_SEQUENCE_TO_LABEL[segment_sequence_name],
    }


def _prompt_row_for_mode(rows: list[dict[str, Any]], idx: int, mode: DatasetMode) -> dict[str, Any] | None:
    if mode == "spec_correct":
        return rows[idx]
    if mode == "spec_shuffle":
        return rows[(idx + 1) % len(rows)]
    if mode == "no_spec":
        return None
    raise ValueError(f"unsupported dataset mode: {mode}")


class CatSpecMiniDataset(Dataset):
    """Auto-GT rows plus encoded CatSpec prompt inputs."""

    def __init__(
        self,
        manifest_or_jsonl: str | Path,
        mode: DatasetMode = "spec_correct",
        max_tokens: int = 96,
    ) -> None:
        self.rows = load_autogt_rows(manifest_or_jsonl)
        self.mode = mode
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        prompt = _prompt_row_for_mode(self.rows, idx, self.mode)
        encoded = batch_encode_specs([prompt], max_tokens=self.max_tokens)
        targets = derive_targets(row)
        prompt_category = prompt["category"] if prompt is not None else "<no_spec>"
        model_inputs = {
            "token_ids": encoded["token_ids"][0],
            "token_mask": encoded["token_mask"][0],
        }
        return {
            "category": row["category"],
            "prompt_category": prompt_category,
            "mode": self.mode,
            "model_inputs": model_inputs,
            "targets": targets,
            "raw": row,
        }


def collate_mini_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate mini dataset items for a DataLoader."""

    return {
        "category": [item["category"] for item in items],
        "prompt_category": [item["prompt_category"] for item in items],
        "mode": [item["mode"] for item in items],
        "model_inputs": {
            "token_ids": torch.stack([item["model_inputs"]["token_ids"] for item in items]),
            "token_mask": torch.stack([item["model_inputs"]["token_mask"] for item in items]),
        },
        "targets": {
            "topology_label": torch.tensor([item["targets"]["topology_label"] for item in items], dtype=torch.long),
            "path_count_label": torch.tensor([item["targets"]["path_count_label"] for item in items], dtype=torch.long),
            "segment_sequence_label": torch.tensor(
                [item["targets"]["segment_sequence_label"] for item in items],
                dtype=torch.long,
            ),
        },
        "target_names": {
            "topology_name": [item["targets"]["topology_name"] for item in items],
            "path_count": [item["targets"]["path_count"] for item in items],
            "segment_sequence_name": [item["targets"]["segment_sequence_name"] for item in items],
        },
        "raw": [item["raw"] for item in items],
    }
