"""P2 preprocessing for CatSpec-Pose dataset manifests."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml

from catspec.schema import load_catspec
from catspec.validation import _generate_catspec_loci, _jsonify, validate_catspec


P2_SCHEMA_VERSION = "catspec.p2_manifest.v0.1"
P2_SAMPLE_SCHEMA_VERSION = "catspec.p2_sample.v0.1"
DEFAULT_DATASET_ROOT = Path("/home/dq/mnt/localgit/aiws5.2/datasets/obj_share_models")
DEFAULT_SPEC_ROOT = Path("specs/categories")
DEFAULT_CATEGORY_ORDER = ["square_tube", "channel_steel", "H_beam", "bellmouth"]


def _category_sort_key(category: str) -> tuple[int, str]:
    if category in DEFAULT_CATEGORY_ORDER:
        return (DEFAULT_CATEGORY_ORDER.index(category), category)
    return (len(DEFAULT_CATEGORY_ORDER), category)


def scan_obj_pairs(dataset_root: str | Path, categories: list[str] | None = None) -> list[dict[str, Any]]:
    """Scan category folders and pair workpiece OBJ files with their `_weld.obj` references."""

    root = Path(dataset_root)
    selected = set(categories) if categories else None
    pairs: list[dict[str, Any]] = []
    for category_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: _category_sort_key(path.name)):
        category = category_dir.name
        if selected is not None and category not in selected:
            continue
        source_objs = sorted(
            path
            for path in category_dir.glob("*.obj")
            if not path.stem.endswith("_weld")
        )
        for source_obj in source_objs:
            reference_weld = source_obj.with_name(f"{source_obj.stem}_weld.obj")
            pairs.append(
                {
                    "category": category,
                    "source_obj": str(source_obj),
                    "reference_weld_obj": str(reference_weld),
                    "reference_exists": reference_weld.exists(),
                }
            )
    return pairs


def _safe_sample_id(category: str, source_obj: Path) -> str:
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in source_obj.stem)
    return f"{category}__{safe_stem}"


def _geometry_summary(source_obj: Path) -> dict[str, Any]:
    mesh = trimesh.load(source_obj, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    bounds = np.asarray(mesh.bounds, dtype=float)
    extents = bounds[1] - bounds[0]
    return {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "bbox_min": bounds[0].tolist(),
        "bbox_max": bounds[1].tolist(),
        "bbox_extent": extents.tolist(),
        "bbox_volume": float(np.prod(np.maximum(extents, 0.0))),
    }


def _spec_path_for_category(category: str, spec_root: str | Path) -> Path:
    return Path(spec_root) / f"{category}.yaml"


def _write_resolved_spec(
    original_spec_path: Path,
    source_obj: Path,
    reference_weld_obj: Path,
    resolved_spec_path: Path,
) -> dict[str, Any]:
    spec = load_catspec(original_spec_path)
    resolved = copy.deepcopy(spec)
    resolved.setdefault("provenance", {})
    resolved["provenance"]["source_mesh"] = str(source_obj)
    resolved["provenance"]["source_weld_mesh"] = str(reference_weld_obj)
    resolved_spec_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_spec_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return resolved


def _build_sample_record(
    pair: dict[str, Any],
    output_dir: Path,
    spec_root: str | Path,
) -> dict[str, Any]:
    category = pair["category"]
    source_obj = Path(pair["source_obj"])
    reference_weld = Path(pair["reference_weld_obj"])
    if not pair["reference_exists"]:
        raise ValueError(f"missing reference weld OBJ: {reference_weld}")

    original_spec_path = _spec_path_for_category(category, spec_root)
    if not original_spec_path.exists():
        raise ValueError(f"unsupported category without CatSpec: {category}")

    sample_id = _safe_sample_id(category, source_obj)
    resolved_spec_path = output_dir / "cache" / "resolved_specs" / f"{sample_id}.yaml"
    spec = _write_resolved_spec(original_spec_path, source_obj, reference_weld, resolved_spec_path)
    report = validate_catspec(resolved_spec_path, output_dir / "validation" / sample_id)
    generated_locus = _generate_catspec_loci(spec, source_obj)

    return {
        "schema_version": P2_SAMPLE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "category": category,
        "source_mesh": str(source_obj),
        "reference_weld_path": str(reference_weld),
        "original_spec_path": str(original_spec_path),
        "resolved_spec_path": str(resolved_spec_path),
        "generated_locus": generated_locus,
        "weld_meta": spec["welds"][0]["weld_meta"],
        "topology": {
            "topology_match": report["topology_match"],
            "generated": report["generated"],
            "reference": report["reference"],
        },
        "metrics": report["metrics"],
        "source_geometry": _geometry_summary(source_obj),
        "validation_report_path": report["report_path"],
        "overlay_path": report["overlay_path"],
    }


def _make_splits(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    sample_ids = [record["sample_id"] for record in records]
    shuffled = list(sample_ids)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    if n <= 4:
        splits = {"train": list(sample_ids), "val": list(sample_ids), "test": list(sample_ids), "all": list(sample_ids)}
    else:
        train_n = max(1, int(round(n * 0.8)))
        val_n = max(1, int(round(n * 0.1)))
        if train_n + val_n >= n:
            train_n = max(1, n - 2)
            val_n = 1
        splits = {
            "train": shuffled[:train_n],
            "val": shuffled[train_n : train_n + val_n],
            "test": shuffled[train_n + val_n :],
            "all": list(sample_ids),
        }
        if not splits["test"]:
            splits["test"] = splits["val"][-1:]
    return {
        "schema_version": "catspec.p2_splits.v0.1",
        "seed": seed,
        "split_policy": "tiny-dataset-all-splits" if n <= 4 else "deterministic-random-80-10-10",
        "splits": splits,
    }


def _stats(records: list[dict[str, Any]], failures: list[dict[str, Any]], splits: dict[str, Any]) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    for record in records:
        category_counts[record["category"]] = category_counts.get(record["category"], 0) + 1
    return {
        "schema_version": "catspec.p2_stats.v0.1",
        "sample_count": len(records),
        "failure_count": len(failures),
        "category_counts": dict(sorted(category_counts.items(), key=lambda item: _category_sort_key(item[0]))),
        "split_counts": {name: len(values) for name, values in splits["splits"].items()},
        "failed_categories": sorted({failure["category"] for failure in failures}, key=_category_sort_key),
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_jsonify) + "\n", encoding="utf-8")


def _read_cached_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cache_hit"] = True
    return manifest


def preprocess_p2_dataset(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_dir: str | Path = "results/catspec/p2_preprocess",
    spec_root: str | Path = DEFAULT_SPEC_ROOT,
    categories: list[str] | None = None,
    max_samples: int | None = None,
    seed: int = 7,
    force: bool = False,
) -> dict[str, Any]:
    """Scan OBJ assets and write a P2 manifest, sample index, splits, stats, and failures."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "catspec_p2_manifest.json"
    sample_index_path = output / "catspec_p2_sample_index.jsonl"
    split_path = output / "catspec_p2_splits.json"
    stats_path = output / "catspec_p2_stats.json"
    failures_path = output / "catspec_p2_failures.jsonl"

    if manifest_path.exists() and sample_index_path.exists() and split_path.exists() and stats_path.exists() and not force:
        return _read_cached_manifest(manifest_path)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    pairs = scan_obj_pairs(dataset_root, categories=categories)
    if max_samples is not None:
        pairs = pairs[:max_samples]

    for pair in pairs:
        try:
            records.append(_build_sample_record(pair, output, spec_root))
        except Exception as exc:  # noqa: BLE001 - failure rows are part of the preprocessing contract.
            failures.append(
                {
                    "category": pair["category"],
                    "source_obj": pair["source_obj"],
                    "reference_weld_obj": pair["reference_weld_obj"],
                    "reason": str(exc),
                }
            )

    splits = _make_splits(records, seed)
    stats = _stats(records, failures, splits)

    with sample_index_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=_jsonify) + "\n")
    with failures_path.open("w", encoding="utf-8") as fh:
        for failure in failures:
            fh.write(json.dumps(failure, ensure_ascii=False, default=_jsonify) + "\n")

    manifest = {
        "schema_version": P2_SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "spec_root": str(spec_root),
        "output_dir": str(output),
        "manifest_path": str(manifest_path),
        "sample_index_path": str(sample_index_path),
        "split_path": str(split_path),
        "stats_path": str(stats_path),
        "failures_path": str(failures_path),
        "sample_count": len(records),
        "failure_count": len(failures),
        "categories": [category for category in DEFAULT_CATEGORY_ORDER if any(row["category"] == category for row in records)],
        "records": records,
        "splits": splits["splits"],
        "stats": stats,
        "cache_hit": False,
    }
    _write_json(split_path, splits)
    _write_json(stats_path, stats)
    _write_json(manifest_path, manifest)
    return json.loads(json.dumps(manifest, default=_jsonify))
