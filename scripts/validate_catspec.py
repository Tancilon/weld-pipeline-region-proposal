#!/usr/bin/env python
"""Validate a CatSpec YAML against its existing weld OBJ reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        default="specs/categories/square_tube.yaml",
        help="Path to CatSpec YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/catspec",
        help="Directory for JSON report and overlay PNG.",
    )
    args = parser.parse_args()

    from catspec.validation import validate_catspec

    report = validate_catspec(args.spec, args.output_dir)
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
