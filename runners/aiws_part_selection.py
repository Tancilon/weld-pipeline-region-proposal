from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class PartSelectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SelectedPartMask:
    part_name: str
    mask_path: Path
    candidate: dict


def parse_part_mask_overrides(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise PartSelectionError("--part-mask values must use part=mask")
        part, mask = value.split("=", 1)
        part = part.strip()
        mask = mask.strip()
        if not part or not mask:
            raise PartSelectionError("--part-mask values must use non-empty part=mask")
        result[part] = mask
    return result


def _resolve_path(path_value: str, output_root: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = output_root / path
    return path.resolve()


def _candidate_records(candidates_payload: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for candidate in candidates_payload.get("candidates", []):
        mask_id = str(candidate.get("mask_id", ""))
        mask_path = str(candidate.get("mask_path", ""))
        if mask_id and mask_path:
            result[mask_id] = candidate
            result[mask_path] = candidate
    return result


def _load_selected_parts(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    focused_parts = payload.get("focused_parts", {})
    if not isinstance(focused_parts, dict):
        raise PartSelectionError(f"{path} focused_parts must be a mapping")
    return {str(part): str(mask) for part, mask in focused_parts.items() if str(mask)}


def resolve_selected_part_mask_records(
    candidates_path: str | Path,
    selected_parts_path: str | Path,
    cli_overrides: dict[str, str],
    weld_focus: list[str],
    output_root: str | Path,
) -> dict[str, SelectedPartMask]:
    output_root = Path(output_root).resolve()
    candidates_path = Path(candidates_path)
    candidates_payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates_by_key = _candidate_records(candidates_payload)
    selected = _load_selected_parts(Path(selected_parts_path))
    selected.update(cli_overrides)

    result: dict[str, SelectedPartMask] = {}
    missing: list[str] = []
    for part in weld_focus:
        raw_value = selected.get(part, "")
        if not raw_value:
            missing.append(part)
            continue
        candidate = candidates_by_key.get(raw_value)
        if candidate is None:
            mask_path_value = raw_value
            candidate = {"mask_id": Path(raw_value).stem, "mask_path": raw_value}
        else:
            mask_path_value = str(candidate["mask_path"])
        mask_path = _resolve_path(mask_path_value, output_root)
        if not mask_path.exists():
            raise PartSelectionError(f"selected mask for {part} does not exist: {mask_path}")
        result[part] = SelectedPartMask(
            part_name=part,
            mask_path=mask_path,
            candidate=candidate,
        )
    if missing:
        raise PartSelectionError(f"missing selected masks for focused parts: {', '.join(missing)}")
    return result


def resolve_selected_part_masks(
    candidates_path: str | Path,
    selected_parts_path: str | Path,
    cli_overrides: dict[str, str],
    weld_focus: list[str],
    output_root: str | Path,
) -> dict[str, Path]:
    records = resolve_selected_part_mask_records(
        candidates_path=candidates_path,
        selected_parts_path=selected_parts_path,
        cli_overrides=cli_overrides,
        weld_focus=weld_focus,
        output_root=output_root,
    )
    return {part: record.mask_path for part, record in records.items()}
