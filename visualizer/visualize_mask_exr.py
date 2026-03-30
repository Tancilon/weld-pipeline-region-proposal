#!/usr/bin/env python3
"""Visualize a uint mask stored in EXR format."""

from __future__ import annotations

import argparse
import colorsys
from dataclasses import dataclass
from pathlib import Path

import Imath
import numpy as np
import OpenEXR
from PIL import Image, ImageDraw

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class MaskObjectInfo:
    mask_id: int
    pixel_count: int
    coverage: float
    bbox_xyxy: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and visualize a mask EXR file."
    )
    parser.add_argument("mask_path", type=Path, help="Path to *_mask.exr")
    parser.add_argument(
        "--color",
        type=Path,
        default=None,
        help="Optional color image for overlay. Defaults to sibling *_color.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <mask_path stem>_vis.png",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay alpha when a color image is available",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1800,
        help="Resize preview if it is wider than this value",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the generated preview with OpenCV",
    )
    return parser.parse_args()


def load_mask_exr(path: Path) -> np.ndarray:
    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    data_window = header["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    channels = header["channels"]
    if "Y" in channels:
        channel_name = "Y"
    elif "R" in channels:
        channel_name = "R"
    else:
        channel_name = next(iter(channels))

    channel_type_name = str(channels[channel_name].type)
    if channel_type_name == "UINT":
        pixel_type = Imath.PixelType(Imath.PixelType.UINT)
        dtype = np.uint32
    elif channel_type_name == "FLOAT":
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        dtype = np.float32
    elif channel_type_name == "HALF":
        pixel_type = Imath.PixelType(Imath.PixelType.HALF)
        dtype = np.float16
    else:
        raise ValueError(f"Unsupported EXR channel type: {channel_type_name}")

    buffer = exr.channel(channel_name, pixel_type)
    mask = np.frombuffer(buffer, dtype=dtype).reshape(height, width)

    if mask.dtype != np.uint32:
        mask = np.rint(mask).astype(np.uint32)
    return mask


def infer_color_path(mask_path: Path) -> Path | None:
    if mask_path.name.endswith("_mask.exr"):
        candidate = mask_path.with_name(mask_path.name.replace("_mask.exr", "_color.png"))
        if candidate.exists():
            return candidate
    return None


def summarize_mask(mask: np.ndarray) -> list[MaskObjectInfo]:
    total_pixels = int(mask.size)
    object_ids, counts = np.unique(mask, return_counts=True)
    summary: list[MaskObjectInfo] = []

    for mask_id, count in zip(object_ids.tolist(), counts.tolist()):
        if mask_id == 0:
            continue
        ys, xs = np.where(mask == mask_id)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        summary.append(
            MaskObjectInfo(
                mask_id=int(mask_id),
                pixel_count=int(count),
                coverage=float(count / total_pixels),
                bbox_xyxy=(x1, y1, x2, y2),
            )
        )
    return summary


def color_for_id(mask_id: int) -> tuple[int, int, int]:
    if mask_id == 0:
        return (0, 0, 0)
    hue = (mask_id * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return (int(blue * 255), int(green * 255), int(red * 255))


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for mask_id in np.unique(mask):
        vis[mask == mask_id] = color_for_id(int(mask_id))
    return vis


def draw_annotations(image: np.ndarray, summary: list[MaskObjectInfo]) -> np.ndarray:
    if cv2 is None:
        pil_image = Image.fromarray(image[:, :, ::-1])
        draw = ImageDraw.Draw(pil_image)
        for obj in summary:
            color = color_for_id(obj.mask_id)
            x1, y1, x2, y2 = obj.bbox_xyxy
            draw.rectangle([(x1, y1), (x2, y2)], outline=color[::-1], width=3)
            label = f"id={obj.mask_id} px={obj.pixel_count}"
            draw.text((x1, max(0, y1 - 18)), label, fill=color[::-1])
        return np.array(pil_image)[:, :, ::-1]

    annotated = image.copy()
    for obj in summary:
        color = color_for_id(obj.mask_id)
        x1, y1, x2, y2 = obj.bbox_xyxy
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = f"id={obj.mask_id} px={obj.pixel_count}"
        cv2.putText(
            annotated,
            label,
            (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    if cv2 is None:
        pil_image = Image.fromarray(image[:, :, ::-1])
        draw = ImageDraw.Draw(pil_image)
        draw.rectangle([(0, 0), (pil_image.width, 42)], fill=(24, 24, 24))
        draw.text((16, 12), title, fill=(255, 255, 255))
        return np.array(pil_image)[:, :, ::-1]

    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (24, 24, 24), -1)
    cv2.putText(
        canvas,
        title,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def resize_if_needed(image: np.ndarray, max_width: int) -> np.ndarray:
    if image.shape[1] <= max_width:
        return image
    scale = max_width / image.shape[1]
    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    if cv2 is not None:
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    pil_image = Image.fromarray(image[:, :, ::-1])
    pil_image = pil_image.resize(new_size, resample=Image.Resampling.LANCZOS)
    return np.array(pil_image)[:, :, ::-1]


def build_preview(
    mask: np.ndarray,
    summary: list[MaskObjectInfo],
    color_path: Path | None,
    alpha: float,
) -> np.ndarray:
    colored_mask = colorize_mask(mask)
    annotated_mask = draw_annotations(colored_mask, summary)
    panels = [add_title(annotated_mask, "Mask Labels")]

    if color_path is not None and color_path.exists():
        if cv2 is not None:
            color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        else:
            color = np.array(Image.open(color_path).convert("RGB"))[:, :, ::-1]
        if color is not None and color.shape[:2] == mask.shape[:2]:
            if cv2 is not None:
                overlay = cv2.addWeighted(color, 1.0 - alpha, colored_mask, alpha, 0.0)
            else:
                overlay = np.clip(
                    color.astype(np.float32) * (1.0 - alpha)
                    + colored_mask.astype(np.float32) * alpha,
                    0,
                    255,
                ).astype(np.uint8)
            overlay = draw_annotations(overlay, summary)
            panels.insert(0, add_title(color, "Color"))
            panels.insert(1, add_title(overlay, "Overlay"))

    return np.concatenate(panels, axis=1)


def default_output_path(mask_path: Path) -> Path:
    stem = mask_path.name
    if stem.endswith(".exr"):
        stem = stem[:-4]
    return mask_path.with_name(f"{stem}_vis.png")


def print_summary(mask_path: Path, mask: np.ndarray, summary: list[MaskObjectInfo]) -> None:
    unique_ids = np.unique(mask)
    foreground_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)

    print(f"mask_path: {mask_path}")
    print(f"shape: {mask.shape[0]}x{mask.shape[1]}")
    print(f"dtype: {mask.dtype}")
    print(f"unique_ids: {unique_ids.tolist()}")
    print(f"foreground_pixels: {foreground_pixels} / {total_pixels} ({foreground_pixels / total_pixels:.2%})")

    if not summary:
        print("objects: none")
        return

    print("objects:")
    for obj in summary:
        x1, y1, x2, y2 = obj.bbox_xyxy
        print(
            f"  id={obj.mask_id} pixels={obj.pixel_count} "
            f"coverage={obj.coverage:.2%} bbox=({x1}, {y1})-({x2}, {y2})"
        )


def main() -> None:
    args = parse_args()
    mask_path = args.mask_path
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    color_path = args.color or infer_color_path(mask_path)
    output_path = args.output or default_output_path(mask_path)

    mask = load_mask_exr(mask_path)
    summary = summarize_mask(mask)
    preview = build_preview(mask, summary, color_path, alpha=args.alpha)
    preview = resize_if_needed(preview, max_width=args.max_width)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        ok = cv2.imwrite(str(output_path), preview)
        if not ok:
            raise RuntimeError(f"Failed to write preview image: {output_path}")
    else:
        Image.fromarray(preview[:, :, ::-1]).save(output_path)

    print_summary(mask_path, mask, summary)
    print(f"saved_preview: {output_path}")

    if args.show:
        if cv2 is not None:
            cv2.imshow("mask_preview", preview)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            Image.fromarray(preview[:, :, ::-1]).show()


if __name__ == "__main__":
    main()
