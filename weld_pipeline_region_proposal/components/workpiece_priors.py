from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from weld_pipeline_region_proposal.components.repo_paths import PROJECT_ROOT


class WorkpiecePriorError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkpiecePrior:
    category: str
    component_assembly: dict[str, Path]
    weld_focus: list[str]
    part_sizes: dict[str, list[list[float]]]


class WorkpiecePriorRegistry:
    def __init__(self, info_path: str | Path, repo_root: str | Path | None = None):
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else PROJECT_ROOT
        self.info_path = Path(info_path)
        if not self.info_path.is_absolute():
            self.info_path = self.repo_root / self.info_path
        self.info_path = self.info_path.resolve()
        self._entries = self._load_entries()

    def _load_entries(self) -> dict[str, WorkpiecePrior]:
        try:
            payload = yaml.safe_load(self.info_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise WorkpiecePriorError(
                f"workpiece prior file not found: {self.info_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise WorkpiecePriorError("workpiece prior payload must be a mapping")

        entries: dict[str, WorkpiecePrior] = {}
        for category, raw_entry in payload.items():
            if not isinstance(raw_entry, dict):
                raise WorkpiecePriorError(f"{category} prior must be a mapping")
            component_assembly = self._parse_component_assembly(category, raw_entry)
            weld_focus = self._parse_weld_focus(category, raw_entry)
            part_sizes = self._parse_part_sizes(category, raw_entry)
            for part in weld_focus:
                if part not in component_assembly:
                    raise WorkpiecePriorError(
                        f"{category}.{part} missing component_assembly entry"
                    )
                if part not in part_sizes:
                    raise WorkpiecePriorError(f"{category}.{part} missing candidate sizes")
            entries[category] = WorkpiecePrior(
                category=category,
                component_assembly=component_assembly,
                weld_focus=weld_focus,
                part_sizes=part_sizes,
            )
        return entries

    def _parse_component_assembly(
        self, category: str, raw_entry: dict[str, Any]
    ) -> dict[str, Path]:
        raw = raw_entry.get("component_assembly", {})
        if not isinstance(raw, dict):
            raise WorkpiecePriorError(f"{category}.component_assembly must be a mapping")
        result: dict[str, Path] = {}
        for part, value in raw.items():
            if not isinstance(part, str) or not isinstance(value, str) or not value:
                raise WorkpiecePriorError(
                    f"{category}.component_assembly entries must be non-empty strings"
                )
            path = Path(value)
            if not path.is_absolute():
                path = self.repo_root / path
            result[part] = path.resolve()
        return result

    def _parse_weld_focus(self, category: str, raw_entry: dict[str, Any]) -> list[str]:
        raw = raw_entry.get("weld_focus", [])
        if raw is None:
            raw = []
        if not isinstance(raw, list) or not all(
            isinstance(item, str) and item for item in raw
        ):
            raise WorkpiecePriorError(f"{category}.weld_focus must be a list of part names")
        return list(raw)

    def _parse_part_sizes(
        self, category: str, raw_entry: dict[str, Any]
    ) -> dict[str, list[list[float]]]:
        raw_parts = raw_entry.get("parts", {})
        if not isinstance(raw_parts, dict):
            raise WorkpiecePriorError(f"{category}.parts must be a mapping")
        parsed: dict[str, list[list[float]]] = {}
        for part, raw_part in raw_parts.items():
            if not isinstance(raw_part, dict):
                raise WorkpiecePriorError(f"{category}.{part} part entry must be a mapping")
            raw_sizes = raw_part.get("size", [])
            if not isinstance(raw_sizes, list) or not raw_sizes:
                raise WorkpiecePriorError(
                    f"{category}.{part}.size must be a non-empty list"
                )
            sizes = [self._parse_size(category, str(part), size) for size in raw_sizes]
            parsed[str(part)] = sizes
        return parsed

    def _parse_size(self, category: str, part: str, size: Any) -> list[float]:
        if not isinstance(size, list) or len(size) != 3:
            raise WorkpiecePriorError(f"{category}.{part}.size entries must have 3 values")
        values = [float(value) for value in size]
        if any(value <= 0 for value in values):
            raise WorkpiecePriorError(f"{category}.{part}.size values must be positive")
        return values

    def get(self, category: str) -> WorkpiecePrior:
        try:
            return self._entries[category]
        except KeyError as exc:
            raise WorkpiecePriorError(f"{category} prior missing") from exc

    def component_template(self, category: str, part: str) -> Path:
        entry = self.get(category)
        try:
            return entry.component_assembly[part]
        except KeyError as exc:
            raise WorkpiecePriorError(f"{category}.{part} component template missing") from exc

    @staticmethod
    def _generic_match_part_size(
        candidates: list[list[float]],
        raw: tuple[float, ...],
        sigma: float,
        fallback_reason: str | None = None,
    ) -> dict[str, Any]:
        best_candidate: list[float] | None = None
        best_error: float | None = None
        for candidate in candidates:
            error = sum(abs(r - c) / c for r, c in zip(raw, candidate)) / 3.0
            if best_error is None or error < best_error:
                best_error = error
                best_candidate = candidate

        if best_candidate is None or best_error is None:
            raise WorkpiecePriorError("size match failed")
        result: dict[str, Any] = {
            "matched_size_xyz_mm": [float(value) for value in best_candidate],
            "size_match_error": float(best_error),
            "match_confidence": float(math.exp(-best_error / float(sigma))),
            "size_match_method": "per_axis_relative_error_v1",
        }
        if fallback_reason:
            result["size_match_fallback_reason"] = fallback_reason
        return result

    @staticmethod
    def _is_positive_finite_xyz(values: tuple[float, ...]) -> bool:
        return len(values) == 3 and all(
            math.isfinite(float(value)) and float(value) > 0.0 for value in values
        )

    @classmethod
    def _match_square_tube_tube_size(
        cls,
        candidates: list[list[float]],
        raw: tuple[float, ...],
        sigma: float,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not cls._is_positive_finite_xyz(raw):
            return None, "invalid_raw_size"
        for candidate in candidates:
            if len(candidate) != 3 or not math.isclose(
                float(candidate[0]),
                float(candidate[2]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                return None, "invalid_square_tube_candidate"

        raw_x, raw_y, raw_z = raw
        section_estimate = min(raw_x, raw_z)
        length_estimate = raw_y
        xz_symmetry_gap = abs(raw_x - raw_z) / max(raw_x, raw_z)
        candidate_errors: list[dict[str, Any]] = []
        best_candidate: list[float] | None = None
        best_error: float | None = None

        for candidate in candidates:
            section = float(candidate[0])
            length = float(candidate[1])
            section_error = abs(section_estimate - section) / section
            length_error = abs(length_estimate - length) / length
            score = (
                section_error * 0.75
                + length_error * 0.20
                + xz_symmetry_gap * 0.05
            )
            candidate_error = {
                "candidate_size_xyz_mm": [float(value) for value in candidate],
                "section_error": float(section_error),
                "length_error": float(length_error),
                "xz_symmetry_gap": float(xz_symmetry_gap),
                "score": float(score),
            }
            candidate_errors.append(candidate_error)
            if best_error is None or score < best_error:
                best_error = score
                best_candidate = candidate

        if best_candidate is None or best_error is None:
            return None, "no_square_tube_candidate_scored"
        return (
            {
                "matched_size_xyz_mm": [float(value) for value in best_candidate],
                "size_match_error": float(best_error),
                "match_confidence": float(math.exp(-best_error / float(sigma))),
                "size_match_method": "square_tube_section_min_xz_v1",
                "size_match_diagnostics": {
                    "raw_size_xyz_mm": [float(value) for value in raw],
                    "section_estimate_mm": float(section_estimate),
                    "section_source": "min_xz",
                    "length_estimate_mm": float(length_estimate),
                    "xz_symmetry_gap": float(xz_symmetry_gap),
                    "candidate_errors": candidate_errors,
                },
            },
            None,
        )

    def match_part_size(
        self,
        category: str,
        part: str,
        raw_size_xyz_mm: Iterable[float],
        sigma: float = 0.05,
    ) -> dict[str, Any]:
        if sigma <= 0 or not math.isfinite(float(sigma)):
            raise WorkpiecePriorError("sigma must be positive")
        raw = tuple(float(value) for value in raw_size_xyz_mm)
        if len(raw) != 3:
            raise WorkpiecePriorError("raw_size_xyz_mm must have 3 values")
        entry = self.get(category)
        try:
            candidates = entry.part_sizes[part]
        except KeyError as exc:
            raise WorkpiecePriorError(f"{category}.{part} candidate sizes missing") from exc

        if category == "square_tube" and part == "tube":
            square_tube_match, fallback_reason = self._match_square_tube_tube_size(
                candidates,
                raw,
                sigma,
            )
            if square_tube_match is not None:
                return square_tube_match
            return self._generic_match_part_size(
                candidates,
                raw,
                sigma,
                fallback_reason=fallback_reason,
            )

        return self._generic_match_part_size(candidates, raw, sigma)
