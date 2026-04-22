"""Command-line entry: python -m weld ..."""

from __future__ import annotations

import argparse

from weld.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Extract weld seam centerlines from OBJ mesh and fit as line/arc segments."
    )
    parser.add_argument("--workpiece", required=True, help="Path to workpiece OBJ file")
    parser.add_argument("--weld", required=True, help="Path to weld seam OBJ file")
    parser.add_argument("--output", default=None, help="JSON output path (default: auto)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--category", default=None,
                        help="Category name (bellmouth, channel_steel, h_beam, "
                             "square_tube, cover_plate). Unknown or omitted = generic.")
    args = parser.parse_args()
    run_pipeline(args.workpiece, args.weld, args.output, args.no_viz, args.category)


if __name__ == "__main__":
    main()
