from __future__ import annotations

import contextlib
import io
import json
import logging
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REGION_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = REGION_ROOT.parent
for path in (PROJECT_ROOT, REGION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from weld_pipeline_region_proposal.adapter.types import CoarseLocalizationResult
from weld_pipeline_region_proposal.components.aiws_pipeline_contracts import validate_region_proposal
from weld_pipeline_region_proposal.components.rgbd_size_review import review_standard_size_match
from weld_pipeline_region_proposal.components.workpiece_priors import WorkpiecePriorRegistry
from weld_pipeline_region_proposal.utils.depth_compat import load_depth
from runners.aiws_auto_part_selection import select_weld_focus_masks
from runners.aiws_genpose_part import GenPosePartEstimator
from runners.aiws_object_roi import (
    ObjectRoiError,
    ObjectRoiEstimator,
    build_object_roi_from_mask,
)
from runners.aiws_part_selection import (
    PartSelectionError,
    parse_part_mask_overrides,
    resolve_selected_part_mask_records,
)
from runners.aiws_region_proposal_contract import (
    genpose_result_to_part_payload,
    write_region_proposal,
)
from runners.aiws_semantic_sam_candidates import (
    build_mask_candidates,
    write_selected_parts_template,
)


@dataclass(frozen=True)
class VisualCoarseRuntimeConfig:
    workpiece_info_path: str = str(PROJECT_ROOT / "workpiece_priors/workpiece_info.yaml")
    camera_path: str = str(PROJECT_ROOT / "workpiece_priors/camera.json")
    seg_ckpt: str = str(
        PROJECT_ROOT
        / "weld-pipeline-region-proposal/results/ckpts/SegNet_lora_rank4_e30/best.pth"
    )
    energy_ckpt: str = str(
        PROJECT_ROOT / "weld-pipeline-region-proposal/results/ckpts/EnergyNet/energynet.pth"
    )
    scale_ckpt: str = str(
        PROJECT_ROOT / "weld-pipeline-region-proposal/results/ckpts/ScaleNet/scalenet.pth"
    )
    semantic_sam_python: str = sys.executable
    semantic_sam_script: str = str(REGION_ROOT / "Semantic-SAM/run_auto_masks.py")
    semantic_sam_ckpt: str = (
        "/media/bakhda4/tancilon/Semantic-SAM/ckpts/swint_only_sam_many2many.pth"
    )
    semantic_sam_model_type: str = "T"
    semantic_sam_levels: tuple[str, ...] = ("4",)
    object_score_threshold: float = 0.5
    auto_selection_score_threshold: float = 0.65
    auto_selection_margin: float = 0.10
    allow_low_confidence_auto_selection: bool = False
    skip_semantic_sam: bool = False
    force: bool = False


def _sample_id(rgb_path: str | Path) -> str:
    stem = Path(rgb_path).stem
    return stem.removesuffix("_color").removesuffix("_rgb")


def _copy_inputs(rgb_path: Path, depth_path: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_rgb = output_dir / "rgb.png"
    out_depth = output_dir / depth_path.name
    shutil.copy2(rgb_path, out_rgb)
    shutil.copy2(depth_path, out_depth)
    return out_rgb, out_depth


def _camera_intrinsics(camera_path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(camera_path).read_text(encoding="utf-8"))
    return dict(payload.get("intrinsics", payload))


def _size_match_fields(payload: dict) -> dict:
    result = {
        "matched_size_xyz_mm": payload["matched_size_xyz_mm"],
        "size_match_error": payload["size_match_error"],
        "match_confidence": payload["match_confidence"],
        "size_match_method": payload.get("size_match_method", ""),
    }
    if "size_match_diagnostics" in payload:
        result["size_match_diagnostics"] = payload["size_match_diagnostics"]
    if "size_match_fallback_reason" in payload:
        result["size_match_fallback_reason"] = payload["size_match_fallback_reason"]
    return result


def _apply_rgbd_size_review(
    *,
    registry: WorkpiecePriorRegistry,
    category: str,
    part_name: str,
    part_payload: dict,
    depth: np.ndarray,
    mask_path: Path,
    intrinsics: dict[str, float],
) -> dict:
    try:
        candidates = registry.get(category).part_sizes[part_name]
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        reviewed = review_standard_size_match(
            base_match=_size_match_fields(part_payload),
            candidate_sizes_mm=candidates,
            depth=depth,
            mask=mask,
            intrinsics=intrinsics,
        )
    except Exception as exc:
        if float(part_payload.get("match_confidence", 0.0)) < 0.02:
            part_payload["size_match_review"] = {
                "review_method": "rgbd_pca_extents_v1",
                "decision": "review_failed",
                "error": str(exc),
            }
        return part_payload

    part_payload.update(reviewed)
    return part_payload


def _selected_mask_to_original(
    source: Path,
    target_shape: tuple[int, int],
) -> Image.Image:
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    target_height, target_width = target_shape
    mask = Image.open(source).convert("L")
    if mask.size != (target_width, target_height):
        mask = mask.resize((target_width, target_height), resample=nearest)
    return mask


def _copy_selected_masks(
    selected,
    part_masks_dir: Path,
    target_shape: tuple[int, int],
) -> dict[str, Path]:
    part_masks_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for part, record in selected.items():
        target = part_masks_dir / f"{part}.png"
        mask = _selected_mask_to_original(
            source=record.mask_path,
            target_shape=target_shape,
        )
        mask.save(target)
        result[part] = target
    return result


def _write_object_mask_auto_selection(
    *,
    output_dir: Path,
    sample_id: str,
    workpiece_type: str,
    weld_focus: list[str],
) -> Path:
    payload = {
        "schema_version": 1,
        "sample_id": sample_id,
        "workpiece_type": workpiece_type,
        "selection_source": "object_mask",
        "focused_parts": {part: "object_mask" for part in weld_focus},
        "scores": {
            part: {
                "selected_mask_id": "object_mask",
                "score": 1.0,
                "decision": "accepted",
            }
            for part in weld_focus
        },
    }
    output_path = output_dir / "selected_parts.auto.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _copy_object_mask_to_focus_parts(
    *,
    object_mask_path: Path,
    weld_focus: list[str],
    part_masks_dir: Path,
    target_shape: tuple[int, int],
) -> dict[str, Path]:
    part_masks_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for part in weld_focus:
        target = part_masks_dir / f"{part}.png"
        mask = _selected_mask_to_original(
            source=object_mask_path,
            target_shape=target_shape,
        )
        mask.save(target)
        result[part] = target
    return result


def _semantic_sam_command(
    config: VisualCoarseRuntimeConfig,
    rgb_path: Path,
    output_dir: Path,
    sample_id: str,
) -> list[str]:
    return [
        *config.semantic_sam_python.split(),
        config.semantic_sam_script,
        str(rgb_path),
        "--model-type",
        config.semantic_sam_model_type,
        "--ckpt",
        config.semantic_sam_ckpt,
        "--levels",
        *[str(level) for level in config.semantic_sam_levels],
        "--output-dir",
        str(output_dir / "semantic_sam"),
        "--output-prefix",
        sample_id,
    ]


@contextlib.contextmanager
def _runtime_output_scope(*, quiet: bool):
    if not quiet:
        yield
        return

    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                yield
    finally:
        logging.disable(previous_disable_level)


def _ensure_semantic_sam(
    config: VisualCoarseRuntimeConfig,
    rgb_path: Path,
    output_dir: Path,
    sample_id: str,
    *,
    quiet: bool = False,
) -> Path:
    metadata_path = output_dir / "semantic_sam" / f"{sample_id}_metadata.json"
    if metadata_path.exists() and not config.force:
        return metadata_path
    if config.skip_semantic_sam:
        if not metadata_path.exists():
            raise FileNotFoundError(f"Semantic-SAM metadata not found: {metadata_path}")
        return metadata_path
    command = _semantic_sam_command(config, rgb_path, output_dir, sample_id)
    if quiet:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout or "").strip()
            if len(details) > 1000:
                details = details[-1000:]
            raise RuntimeError(
                f"Semantic-SAM failed with exit code {completed.returncode}: {details}"
            )
    else:
        subprocess.run(command, check=True)
    return metadata_path


def _resolve_object_roi(
    *,
    config: VisualCoarseRuntimeConfig,
    out_rgb: Path,
    out_depth: Path,
    output_dir: Path,
    sample_id: str,
    workpiece_type: str | None,
    object_mask_path: str | None,
):
    if object_mask_path:
        if not workpiece_type:
            raise RuntimeError("--workpiece-type is required when --object-mask is supplied")
        return build_object_roi_from_mask(
            rgb_path=out_rgb,
            mask_path=object_mask_path,
            output_dir=output_dir,
            sample_id=sample_id,
            workpiece_type=workpiece_type,
            class_id=-1,
            class_confidence=1.0,
            source_depth_path=out_depth,
        )
    estimator = ObjectRoiEstimator(
        seg_ckpt=config.seg_ckpt,
        device="cuda",
    )
    result = estimator.estimate(
        rgb_path=out_rgb,
        output_dir=output_dir,
        sample_id=sample_id,
        score_threshold=config.object_score_threshold,
        source_depth_path=out_depth,
    )
    if workpiece_type and workpiece_type != result.workpiece_type:
        raise RuntimeError(
            f"workpiece type mismatch: expected {workpiece_type}, "
            f"predicted {result.workpiece_type}"
        )
    return result


def _failure_result(
    status: str,
    *,
    workpiece_type: str | None,
    error: str,
) -> CoarseLocalizationResult:
    return CoarseLocalizationResult(
        status=status,
        workpiece_type=workpiece_type,
        class_confidence=None,
        matched_size_xyz_mm=None,
        size_match_confidence=None,
        part_masks={},
        region_proposal_path=None,
        diagnostics={"error": error},
    )


def visual_coarse_localize(
    rgb_path: str,
    depth_path: str,
    output_dir: str,
    *,
    workpiece_type: str | None = None,
    object_mask_path: str | None = None,
    verbose: bool = False,
) -> CoarseLocalizationResult:
    return visual_coarse_localize_with_config(
        rgb_path=rgb_path,
        depth_path=depth_path,
        output_dir=output_dir,
        workpiece_type=workpiece_type,
        object_mask_path=object_mask_path,
        verbose=verbose,
        config=VisualCoarseRuntimeConfig(),
    )


def visual_coarse_localize_with_config(
    rgb_path: str,
    depth_path: str,
    output_dir: str,
    *,
    workpiece_type: str | None = None,
    object_mask_path: str | None = None,
    verbose: bool = False,
    config: VisualCoarseRuntimeConfig | None = None,
) -> CoarseLocalizationResult:
    with _runtime_output_scope(quiet=not verbose):
        return _visual_coarse_localize_with_config_impl(
            rgb_path=rgb_path,
            depth_path=depth_path,
            output_dir=output_dir,
            workpiece_type=workpiece_type,
            object_mask_path=object_mask_path,
            verbose=verbose,
            config=config,
        )


def _visual_coarse_localize_with_config_impl(
    rgb_path: str,
    depth_path: str,
    output_dir: str,
    *,
    workpiece_type: str | None = None,
    object_mask_path: str | None = None,
    verbose: bool = False,
    config: VisualCoarseRuntimeConfig | None = None,
) -> CoarseLocalizationResult:
    config = config or VisualCoarseRuntimeConfig()
    output = Path(output_dir)
    sample_id = _sample_id(rgb_path)
    try:
        registry = WorkpiecePriorRegistry(config.workpiece_info_path, repo_root=PROJECT_ROOT)
        out_rgb, out_depth = _copy_inputs(Path(rgb_path), Path(depth_path), output)
        depth = load_depth(out_depth)
        object_roi = _resolve_object_roi(
            config=config,
            out_rgb=out_rgb,
            out_depth=out_depth,
            output_dir=output,
            sample_id=sample_id,
            workpiece_type=workpiece_type,
            object_mask_path=object_mask_path,
        )
    except ObjectRoiError as exc:
        return _failure_result("unrecognized", workpiece_type=workpiece_type, error=str(exc))
    except Exception as exc:
        return _failure_result("failed", workpiece_type=workpiece_type, error=str(exc))

    resolved_workpiece_type = object_roi.workpiece_type
    try:
        prior = registry.get(resolved_workpiece_type)
        object_mask = object_roi.object_mask
        resolved_object_mask_path = object_roi.object_mask_path
        auto_selected_parts_path = output / "selected_parts.auto.json"

        if resolved_workpiece_type == "cover_plate":
            auto_selected_parts_path = _write_object_mask_auto_selection(
                output_dir=output,
                sample_id=sample_id,
                workpiece_type=resolved_workpiece_type,
                weld_focus=prior.weld_focus,
            )
            part_masks = _copy_object_mask_to_focus_parts(
                object_mask_path=resolved_object_mask_path,
                weld_focus=prior.weld_focus,
                part_masks_dir=output / "part_masks",
                target_shape=tuple(int(value) for value in depth.shape),
            )
        else:
            metadata_path = _ensure_semantic_sam(
                config,
                object_roi.masked_full_rgb_path,
                output,
                sample_id,
                quiet=not verbose,
            )
            candidates_path = build_mask_candidates(
                metadata_path=metadata_path,
                output_path=output / "mask_candidates.json",
                sample_id=sample_id,
                workpiece_type=resolved_workpiece_type,
                depth=depth,
                object_mask=object_mask,
                output_root=output,
            )
            write_selected_parts_template(
                output_path=output / "selected_parts.template.json",
                sample_id=sample_id,
                workpiece_type=resolved_workpiece_type,
                weld_focus=prior.weld_focus,
            )
            cli_overrides = parse_part_mask_overrides([])
            try:
                selected = resolve_selected_part_mask_records(
                    candidates_path=candidates_path,
                    selected_parts_path=output / "selected_parts.json",
                    auto_selected_parts_path=auto_selected_parts_path,
                    cli_overrides=cli_overrides,
                    weld_focus=prior.weld_focus,
                    output_root=output,
                )
            except PartSelectionError:
                auto_selection = select_weld_focus_masks(
                    candidates_path=candidates_path,
                    output_root=output,
                    weld_focus=prior.weld_focus,
                    object_mask=object_mask,
                    depth=depth,
                    accept_score_threshold=config.auto_selection_score_threshold,
                    min_score_margin=config.auto_selection_margin,
                    allow_low_confidence=config.allow_low_confidence_auto_selection,
                )
                if not auto_selection.accepted:
                    return CoarseLocalizationResult(
                        status="segmentation_required",
                        workpiece_type=resolved_workpiece_type,
                        class_confidence=float(object_roi.class_confidence),
                        matched_size_xyz_mm=None,
                        size_match_confidence=None,
                        part_masks={},
                        region_proposal_path=None,
                        object_mask_path=str(resolved_object_mask_path),
                        diagnostics={
                            "reason": auto_selection.reason,
                            "candidates_path": str(candidates_path),
                            "auto_selection_path": str(auto_selection.diagnostics_path),
                        },
                    )
                selected = resolve_selected_part_mask_records(
                    candidates_path=candidates_path,
                    selected_parts_path=output / "selected_parts.json",
                    auto_selected_parts_path=auto_selection.selected_parts_path,
                    cli_overrides=cli_overrides,
                    weld_focus=prior.weld_focus,
                    output_root=output,
                )
            part_masks = _copy_selected_masks(
                selected,
                output / "part_masks",
                target_shape=tuple(int(value) for value in depth.shape),
            )

        estimator = GenPosePartEstimator(
            seg_ckpt=config.seg_ckpt,
            energy_ckpt=config.energy_ckpt,
            scale_ckpt=config.scale_ckpt,
            device="cuda",
        )
        intrinsics = _camera_intrinsics(config.camera_path)
        focused_parts = {}
        for part_name, mask_path in part_masks.items():
            genpose_result = estimator.estimate_part(
                rgb_path=out_rgb,
                depth=depth,
                mask_path=mask_path,
                intrinsics=intrinsics,
            )
            focused_parts[part_name] = genpose_result_to_part_payload(
                registry=registry,
                category=resolved_workpiece_type,
                part_name=part_name,
                mask_path=mask_path,
                genpose_result=genpose_result,
            )
            focused_parts[part_name] = _apply_rgbd_size_review(
                registry=registry,
                category=resolved_workpiece_type,
                part_name=part_name,
                part_payload=focused_parts[part_name],
                depth=depth,
                mask_path=mask_path,
                intrinsics=intrinsics,
            )

        region_path = write_region_proposal(
            output_dir=output,
            sample_id=sample_id,
            workpiece_type=resolved_workpiece_type,
            camera_path=config.camera_path,
            rgb_path=out_rgb,
            depth_path=out_depth,
            object_mask_path=resolved_object_mask_path,
            focused_parts=focused_parts,
        )
        validate_region_proposal(json.loads(region_path.read_text(encoding="utf-8")))
        primary_part = prior.weld_focus[0]
        primary_payload = focused_parts[primary_part]
        diagnostics = {}
        if verbose:
            diagnostics = {
                "selected_parts_path": str(auto_selected_parts_path),
                "object_roi_path": str(object_roi.roi_metadata_path),
            }
        return CoarseLocalizationResult(
            status="ok",
            workpiece_type=resolved_workpiece_type,
            class_confidence=float(object_roi.class_confidence),
            matched_size_xyz_mm=[
                float(value) for value in primary_payload["matched_size_xyz_mm"]
            ],
            size_match_confidence=float(primary_payload["match_confidence"]),
            part_masks={part: str(path) for part, path in part_masks.items()},
            region_proposal_path=str(region_path),
            object_mask_path=str(resolved_object_mask_path),
            raw_size_xyz_mm=[
                float(value) for value in primary_payload["raw_size_xyz_mm"]
            ],
            coarse_pose_cam_4x4=primary_payload.get("coarse_pose_cam_4x4"),
            selected_parts_path=str(auto_selected_parts_path)
            if auto_selected_parts_path.exists()
            else None,
            diagnostics=diagnostics if verbose else None,
        )
    except Exception as exc:
        return _failure_result(
            "failed",
            workpiece_type=resolved_workpiece_type,
            error=str(exc),
        )
