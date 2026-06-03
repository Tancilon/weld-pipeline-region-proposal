from __future__ import annotations

from pathlib import Path

import numpy as np


_SUPPORTED_NON_EXR_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _read_exr(path: Path) -> np.ndarray:
    try:
        import OpenEXR
        import Imath
    except ImportError as exc:
        raise RuntimeError("OpenEXR support is not available") from exc

    try:
        exr = OpenEXR.InputFile(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read EXR file: {path}") from exc

    try:
        try:
            header = exr.header()
            channels = header["channels"]
            channel_name = _select_exr_channel(channels)
            data_window = header.get("dataWindow")
            if data_window is None:
                raise RuntimeError(f"EXR file missing dataWindow: {path}")

            width = int(data_window.max.x - data_window.min.x + 1)
            height = int(data_window.max.y - data_window.min.y + 1)
            pixel_type = Imath.PixelType(getattr(Imath.PixelType, "FLOAT", 0))
            raw = exr.channel(channel_name, pixel_type)
            depth = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
            return depth
        except Exception as exc:
            raise RuntimeError(f"Failed to read EXR file: {path}") from exc
    finally:
        close = getattr(exr, "close", None)
        if callable(close):
            close()


def _select_exr_channel(channels: dict[str, object]) -> str:
    if "Y" in channels:
        return "Y"
    if "R" in channels:
        return "R"
    return next(iter(channels))


def _valid_depth_values(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    return depth[valid]


def _normalize_depth_to_meters(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = _valid_depth_values(depth)
    if valid.size == 0:
        return np.zeros(depth.shape, dtype=np.float32)

    valid_max = float(valid.max())
    valid_mean = float(valid.mean())
    if valid_max > 10.0 or valid_mean > 1000.0:
        depth = depth / 1000.0
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32, copy=False)
    return depth


def _load_non_exr(path: Path) -> np.ndarray:
    import cv2

    depth = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(f"Depth image not found: {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return np.asarray(depth, dtype=np.float32)


def load_depth(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Depth image not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".exr":
        depth = _read_exr(path)
    elif suffix in _SUPPORTED_NON_EXR_SUFFIXES:
        depth = _load_non_exr(path)
    else:
        raise ValueError(f"Unsupported depth image suffix: {path.suffix}")

    return _normalize_depth_to_meters(depth)
