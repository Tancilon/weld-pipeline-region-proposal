#!/usr/bin/env python
"""Validate square_tube CatSpec v0 against the existing weld OBJ."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        default="specs/categories/square_tube.yaml",
        help="Path to square_tube CatSpec YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/catspec/square_tube",
        help="Directory for JSON report and overlay PNG.",
    )
    args = parser.parse_args()

    from catspec.validation import validate_square_tube

    report = validate_square_tube(args.spec, args.output_dir)
    print(
        json.dumps(
            {
                "category": report["category"],
                "topology_match": report["topology_match"],
                "metrics": report["metrics"],
                "report_path": report["report_path"],
                "overlay_path": report["overlay_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
