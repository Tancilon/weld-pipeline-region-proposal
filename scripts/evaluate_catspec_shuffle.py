#!/usr/bin/env python
"""Run CatSpec spec-correct versus spec-shuffle offline evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SPECS = [
    "specs/categories/square_tube.yaml",
    "specs/categories/channel_steel.yaml",
    "specs/categories/H_beam.yaml",
    "specs/categories/bellmouth.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        action="append",
        dest="specs",
        help="Path to a CatSpec YAML. May be repeated. Defaults to all supported v0.2 specs.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/catspec/shuffle",
        help="Directory for the shuffle JSON report and overlays.",
    )
    args = parser.parse_args()

    from catspec.shuffle import evaluate_spec_shuffle

    report = evaluate_spec_shuffle(args.specs or DEFAULT_SPECS, args.output_dir)
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "target_count": report["target_count"],
                "aggregate": report["aggregate"],
                "report_path": report["report_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
