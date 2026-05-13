#!/usr/bin/env python
"""Evaluate CatSpec-Pose mini predictions across prompt modes."""

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
    parser.add_argument("--predictions", required=True, help="Path to mini prediction JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory for evaluation report.")
    args = parser.parse_args()

    from catspec.mini_pipeline import evaluate_predictions

    report = evaluate_predictions(args.predictions, args.output_dir)
    print(
        json.dumps(
            {
                "report_path": report["report_path"],
                "gate": report["gate"],
                "modes": report["modes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
