"""Auto-GT manifest export for CatSpec-derived weld features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from catspec.schema import load_catspec, resolve_asset_path
from catspec.validation import _generate_catspec_loci, _jsonify, validate_catspec


def _record_from_spec(spec_path: Path, output_dir: Path) -> dict[str, Any]:
    spec = load_catspec(spec_path)
    category = spec["category"]
    report = validate_catspec(spec_path, output_dir / category)
    source_mesh = resolve_asset_path(spec["provenance"]["source_mesh"], spec_path)
    reference_weld = resolve_asset_path(spec["provenance"]["source_weld_mesh"], spec_path)
    generated_locus = _generate_catspec_loci(spec, source_mesh)

    return {
        "category": category,
        "schema_version": spec["schema_version"],
        "spec_path": str(spec_path),
        "source_mesh": str(source_mesh),
        "reference_weld_path": str(reference_weld),
        "generated_locus": generated_locus,
        "weld_meta": spec["welds"][0]["weld_meta"],
        "topology": {
            "topology_match": report["topology_match"],
            "generated": report["generated"],
            "reference": report["reference"],
        },
        "metrics": report["metrics"],
        "validation_report_path": report["report_path"],
        "overlay_path": report["overlay_path"],
    }


def export_autogt_manifest(spec_paths: list[str | Path], output_dir: str | Path) -> dict[str, Any]:
    """Export one JSONL row per CatSpec category plus a JSON manifest."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = [Path(path) for path in spec_paths]
    records = [_record_from_spec(spec_path, output) for spec_path in specs]

    jsonl_path = output / "catspec_autogt.jsonl"
    manifest_path = output / "catspec_autogt_manifest.json"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=_jsonify) + "\n")

    manifest = {
        "schema_version": "catspec.autogt.v0.2",
        "category_count": len(records),
        "jsonl_path": str(jsonl_path),
        "manifest_path": str(manifest_path),
        "records": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_jsonify) + "\n",
        encoding="utf-8",
    )
    return json.loads(json.dumps(manifest, default=_jsonify))
