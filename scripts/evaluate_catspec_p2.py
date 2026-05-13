#!/usr/bin/env python
"""Evaluate CatSpec-Pose P2 predictions."""

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
    parser.add_argument("--predictions", required=True, help="P2 prediction JSONL path.")
    parser.add_argument("--manifest", default=None, help="Optional P2 manifest path for preprocess failure rows.")
    parser.add_argument("--output-dir", default="results/catspec/p2_eval", help="Evaluation output directory.")
    args = parser.parse_args()

    from catspec.p2_pipeline import evaluate_p2_predictions

    report = evaluate_p2_predictions(args.predictions, args.output_dir, manifest=args.manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
