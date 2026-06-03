from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

REGION_ROOT = Path(__file__).resolve().parents[2]
if str(REGION_ROOT) in sys.path:
    sys.path.remove(str(REGION_ROOT))
sys.path.insert(0, str(REGION_ROOT))
for name in [
    module_name
    for module_name in sys.modules
    if module_name == "utils" or module_name.startswith("utils.")
]:
    sys.modules.pop(name, None)

from utils.depth_compat import load_depth


class _FakeInputFile:
    def __init__(self, path: str, channels: dict[str, np.ndarray]):
        self.path = path
        self._channels = channels

    def header(self):
        first = next(iter(self._channels.values()))
        height, width = first.shape

        class _Point:
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

        class _Window:
            def __init__(self, width: int, height: int):
                self.min = _Point(0, 0)
                self.max = _Point(width - 1, height - 1)

        return {
            "channels": {name: object() for name in self._channels},
            "dataWindow": _Window(width, height),
        }

    def channel(self, name: str, _pixel_type):
        return self._channels[name].astype(np.float32, copy=False).tobytes()


def _install_fake_exr(monkeypatch, channels: dict[str, np.ndarray]):
    fake_module = types.SimpleNamespace()
    fake_module.InputFile = lambda path: _FakeInputFile(path, channels)
    fake_imath = types.SimpleNamespace(PixelType=lambda *_args, **_kwargs: object(), FLOAT=0)
    monkeypatch.setitem(sys.modules, "OpenEXR", fake_module)
    monkeypatch.setitem(sys.modules, "Imath", fake_imath)


def _install_fake_exr_input_failure(monkeypatch, exc: Exception):
    fake_module = types.SimpleNamespace()

    def _raise(_path: str):
        raise exc

    fake_module.InputFile = _raise
    fake_imath = types.SimpleNamespace(PixelType=lambda *_args, **_kwargs: object(), FLOAT=0)
    monkeypatch.setitem(sys.modules, "OpenEXR", fake_module)
    monkeypatch.setitem(sys.modules, "Imath", fake_imath)


def _install_fake_exr_header_failure(monkeypatch, exc: Exception):
    class _BrokenInputFile:
        def __init__(self, path: str):
            self.path = path

        def header(self):
            raise exc

        def close(self):
            pass

    fake_module = types.SimpleNamespace()
    fake_module.InputFile = lambda path: _BrokenInputFile(path)
    fake_imath = types.SimpleNamespace(PixelType=lambda *_args, **_kwargs: object(), FLOAT=0)
    monkeypatch.setitem(sys.modules, "OpenEXR", fake_module)
    monkeypatch.setitem(sys.modules, "Imath", fake_imath)


def _install_fake_cv2(monkeypatch, image: np.ndarray):
    fake_cv2 = types.SimpleNamespace()
    fake_cv2.IMREAD_ANYCOLOR = 1
    fake_cv2.IMREAD_ANYDEPTH = 2
    fake_cv2.imread = lambda *_args, **_kwargs: image
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)


def test_load_depth_prefers_y_then_r(monkeypatch, tmp_path):
    path = tmp_path / "depth.exr"
    path.write_bytes(b"exr")
    channels = {
        "Z": np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float32),
        "Y": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "R": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    }
    _install_fake_exr(monkeypatch, channels)

    depth = load_depth(path)

    assert depth.dtype == np.float32
    assert depth.shape == (2, 2)
    np.testing.assert_allclose(depth, channels["Y"])


def test_load_depth_uses_r_when_y_missing(monkeypatch, tmp_path):
    path = tmp_path / "depth.exr"
    path.write_bytes(b"exr")
    channels = {
        "Z": np.array([[9.0, 9.0], [9.0, 9.0]], dtype=np.float32),
        "R": np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32),
    }
    _install_fake_exr(monkeypatch, channels)

    depth = load_depth(path)

    np.testing.assert_allclose(depth, channels["R"])


def test_load_depth_uses_first_header_channel_when_y_and_r_missing(monkeypatch, tmp_path):
    path = tmp_path / "depth.exr"
    path.write_bytes(b"exr")
    channels = {
        "G": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "B": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    }
    _install_fake_exr(monkeypatch, channels)

    depth = load_depth(path)

    np.testing.assert_allclose(depth, channels["G"])


def test_load_depth_normalizes_millimeters_to_meters(monkeypatch, tmp_path):
    path = tmp_path / "depth.exr"
    path.write_bytes(b"exr")
    channels = {
        "Z": np.array([[500.0, 1500.0], [2500.0, 3500.0]], dtype=np.float32),
    }
    _install_fake_exr(monkeypatch, channels)

    depth = load_depth(path)

    np.testing.assert_allclose(depth, channels["Z"] / 1000.0)


def test_load_depth_keeps_meter_scale_for_small_values(monkeypatch, tmp_path):
    path = tmp_path / "depth.exr"
    path.write_bytes(b"exr")
    channels = {
        "Z": np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32),
    }
    _install_fake_exr(monkeypatch, channels)

    depth = load_depth(path)

    np.testing.assert_allclose(depth, channels["Z"])


def test_load_depth_returns_zero_array_for_all_zero_exr(monkeypatch, tmp_path):
    path = tmp_path / "depth.exr"
    path.write_bytes(b"exr")
    channels = {
        "Z": np.zeros((2, 3), dtype=np.float32),
    }
    _install_fake_exr(monkeypatch, channels)

    depth = load_depth(path)

    np.testing.assert_array_equal(depth, np.zeros((2, 3), dtype=np.float32))


def test_load_depth_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        load_depth(Path("/tmp/definitely-missing-depth.exr"))


def test_load_depth_collapses_non_exr_to_first_channel(monkeypatch, tmp_path):
    path = tmp_path / "depth.png"
    path.write_bytes(b"png")
    image = np.array(
        [
            [[1, 9, 9], [2, 9, 9]],
            [[3, 9, 9], [4, 9, 9]],
        ],
        dtype=np.uint16,
    )
    _install_fake_cv2(monkeypatch, image)

    depth = load_depth(path)

    assert depth.dtype == np.float32
    np.testing.assert_array_equal(depth, image[:, :, 0].astype(np.float32))


def test_load_depth_returns_zero_for_all_invalid_exr(monkeypatch, tmp_path):
    path = tmp_path / "depth.exr"
    path.write_bytes(b"exr")
    channels = {
        "Z": np.array([[np.nan, -1.0], [np.inf, -2.0]], dtype=np.float32),
    }
    _install_fake_exr(monkeypatch, channels)

    depth = load_depth(path)

    np.testing.assert_array_equal(depth, np.zeros((2, 2), dtype=np.float32))


def test_load_depth_raises_value_error_for_unsupported_existing_suffix(monkeypatch, tmp_path):
    path = tmp_path / "depth.txt"
    path.write_bytes(b"depth")

    fake_cv2 = types.SimpleNamespace()
    fake_cv2.IMREAD_ANYCOLOR = 1
    fake_cv2.IMREAD_ANYDEPTH = 2
    fake_cv2.imread = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    with pytest.raises(ValueError):
        load_depth(path)


def test_load_depth_zeroes_invalid_values_and_scales_valid_mm_exr(monkeypatch, tmp_path):
    path = tmp_path / "depth.exr"
    path.write_bytes(b"exr")
    channels = {
        "Z": np.array([[500.0, np.inf], [1500.0, -1.0]], dtype=np.float32),
    }
    _install_fake_exr(monkeypatch, channels)

    depth = load_depth(path)

    expected = np.array([[0.5, 0.0], [1.5, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(depth, expected)


def test_load_depth_wraps_exr_open_failure_with_path_context(monkeypatch, tmp_path):
    path = tmp_path / "depth.exr"
    path.write_bytes(b"exr")
    _install_fake_exr_input_failure(monkeypatch, ValueError("boom"))

    with pytest.raises(RuntimeError, match=str(path)):
        load_depth(path)


def test_load_depth_wraps_exr_parse_failure_with_path_context(monkeypatch, tmp_path):
    path = tmp_path / "depth.exr"
    path.write_bytes(b"exr")
    _install_fake_exr_header_failure(monkeypatch, ValueError("boom"))

    with pytest.raises(RuntimeError, match=str(path)):
        load_depth(path)
