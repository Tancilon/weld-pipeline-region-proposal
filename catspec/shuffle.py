"""Offline spec-correct versus spec-shuffle evaluation for CatSpec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from catspec.locus import sample_locus_3d, symmetric_hausdorff, symmetric_rmse
from catspec.schema import load_catspec, resolve_asset_path
from catspec.validation import (
    _expected_types_for_locus,
    _generate_catspec_loci,
    _jsonify,
    _sample_reference_path_3d,
    _write_multi_overlay,
)
from weld.core import load_weld_mesh
from weld.strategies import get_strategy


def _sort_infos(path_infos: list[dict[str, Any]], axis_index: int) -> list[dict[str, Any]]:
    return sorted(path_infos, key=lambda info: float(np.mean(info["points_3d"][:, axis_index])))


def _failure_record(
    target_spec_path: Path,
    prompt_spec_path: Path,
    target_spec: dict[str, Any],
    prompt_spec: dict[str, Any],
    output_dir: Path,
    error: Exception,
) -> dict[str, Any]:
    is_correct = target_spec_path == prompt_spec_path
    return {
        "target_category": target_spec["category"],
        "prompt_category": prompt_spec["category"],
        "mode": "spec_correct" if is_correct else "spec_shuffle",
        "is_spec_correct": is_correct,
        "topology_match": False,
        "metrics": {
            "centerline_rmse": None,
            "hausdorff": None,
            "topology_match": False,
            "failure_rate": 1.0,
            "correct_contact_edge_recall": 0.0,
        },
        "error": str(error),
        "report_path": str(output_dir / "catspec_shuffle_report.json"),
        "overlay_path": None,
    }


def _evaluate_pair(target_spec_path: Path, prompt_spec_path: Path, output_dir: Path) -> dict[str, Any]:
    target_spec = load_catspec(target_spec_path)
    prompt_spec = load_catspec(prompt_spec_path)
    is_correct = target_spec_path == prompt_spec_path
    mode = "spec_correct" if is_correct else "spec_shuffle"

    try:
        source_mesh = resolve_asset_path(target_spec["provenance"]["source_mesh"], target_spec_path)
        reference_weld = resolve_asset_path(target_spec["provenance"]["source_weld_mesh"], target_spec_path)
        generated_loci = _generate_catspec_loci(prompt_spec, source_mesh)
        points_per_segment = int(prompt_spec["welds"][0]["locus"]["params"]["sample_points_per_segment"])
        generated_infos = [
            {
                "locus": locus,
                "points_3d": sample_locus_3d(locus, points_per_segment),
                "segment_types": [segment["type"] for segment in locus["segments"]],
                "closed": bool(locus.get("closed")),
            }
            for locus in generated_loci
        ]

        reference_paths = get_strategy(target_spec["category"]).process(load_weld_mesh(str(reference_weld)))
        reference_infos = [
            {
                "path": path,
                "points_3d": _sample_reference_path_3d(path, points_per_segment),
                "segment_types": [segment["type"] for segment in path["fitted"]],
                "closed": bool(path.get("closed")),
            }
            for path in reference_paths
        ]

        params = prompt_spec["welds"][0]["locus"]["params"]
        sort_axis = params.get("offset_axis", params["plane_axis"])
        axis_index = {"x": 0, "y": 1, "z": 2}[sort_axis]
        generated_infos = _sort_infos(generated_infos, axis_index)
        reference_infos = _sort_infos(reference_infos, axis_index)
        pairs = list(zip(generated_infos, reference_infos))
        expected_types = _expected_types_for_locus(prompt_spec["welds"][0]["locus"]["type"])
        topology_match = (
            len(generated_infos) == len(reference_infos)
            and all(info["segment_types"] == expected_types for info in generated_infos)
            and all(generated["segment_types"] == reference["segment_types"] and generated["closed"] == reference["closed"] for generated, reference in pairs)
        )
        matching_edges = sum(
            generated["segment_types"] == reference["segment_types"] and generated["closed"] == reference["closed"]
            for generated, reference in pairs
        )
        recall = matching_edges / len(reference_infos) if reference_infos else 0.0
        per_pair = [
            {
                "path_index": idx,
                "centerline_rmse": symmetric_rmse(generated["points_3d"], reference["points_3d"]),
                "hausdorff": symmetric_hausdorff(generated["points_3d"], reference["points_3d"]),
            }
            for idx, (generated, reference) in enumerate(pairs)
        ]
        overlay_path = output_dir / f"{target_spec['category']}__prompt_{prompt_spec['category']}.png"
        if pairs:
            _write_multi_overlay(
                [
                    {
                        "generated_3d": generated["points_3d"],
                        "reference_3d": reference["points_3d"],
                    }
                    for generated, reference in pairs
                ],
                tuple(prompt_spec["welds"][0]["locus"]["params"]["profile_axes"]),
                overlay_path,
            )

        return {
            "target_category": target_spec["category"],
            "prompt_category": prompt_spec["category"],
            "mode": mode,
            "is_spec_correct": is_correct,
            "topology_match": topology_match,
            "metrics": {
                "centerline_rmse": float(np.mean([item["centerline_rmse"] for item in per_pair])) if per_pair else None,
                "hausdorff": float(max(item["hausdorff"] for item in per_pair)) if per_pair else None,
                "topology_match": topology_match,
                "failure_rate": 0.0 if topology_match else 1.0,
                "correct_contact_edge_recall": float(recall),
                "per_path": per_pair,
            },
            "report_path": str(output_dir / "catspec_shuffle_report.json"),
            "overlay_path": str(overlay_path) if pairs else None,
        }
    except Exception as exc:
        return _failure_record(target_spec_path, prompt_spec_path, target_spec, prompt_spec, output_dir, exc)


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [record for record in records if record["mode"] == "spec_correct"]
    shuffled = [record for record in records if record["mode"] == "spec_shuffle"]
    finite_correct_rmse = [record["metrics"]["centerline_rmse"] for record in correct if record["metrics"]["centerline_rmse"] is not None]
    finite_shuffle_rmse = [record["metrics"]["centerline_rmse"] for record in shuffled if record["metrics"]["centerline_rmse"] is not None]
    return {
        "correct_count": len(correct),
        "shuffle_count": len(shuffled),
        "failure_rate": float(np.mean([record["metrics"]["failure_rate"] for record in records])) if records else 0.0,
        "correct_contact_edge_recall": float(np.mean([record["metrics"]["correct_contact_edge_recall"] for record in records])) if records else 0.0,
        "spec_correct_mean_rmse": float(np.mean(finite_correct_rmse)) if finite_correct_rmse else None,
        "spec_shuffle_mean_rmse": float(np.mean(finite_shuffle_rmse)) if finite_shuffle_rmse else None,
    }


def evaluate_spec_shuffle(spec_paths: list[str | Path], output_dir: str | Path) -> dict[str, Any]:
    """Evaluate each target spec with its own prompt and one shuffled prompt."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = [Path(path) for path in spec_paths]
    records: list[dict[str, Any]] = []
    for idx, target in enumerate(specs):
        records.append(_evaluate_pair(target, target, output))
        if len(specs) > 1:
            records.append(_evaluate_pair(target, specs[(idx + 1) % len(specs)], output))

    report_path = output / "catspec_shuffle_report.json"
    report = {
        "schema_version": "catspec.shuffle.v0.2",
        "target_count": len(specs),
        "report_path": str(report_path),
        "aggregate": _aggregate(records),
        "records": records,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_jsonify) + "\n",
        encoding="utf-8",
    )
    return json.loads(json.dumps(report, default=_jsonify))
