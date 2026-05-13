#!/usr/bin/env python
"""Run CatSpec-Pose mini inference for spec-correct, spec-shuffle, and no-spec modes."""

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
    parser.add_argument("--manifest", required=True, help="Path to CatSpec auto-GT manifest JSON or JSONL.")
    parser.add_argument("--checkpoint", required=True, help="Path to mini head checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory for prediction JSONL.")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")
    args = parser.parse_args()

    from catspec.mini_pipeline import evaluate_predictions, infer_mini

    result = infer_mini(
        args.manifest,
        args.checkpoint,
        args.output_dir,
        batch_size=args.batch_size,
    )
    report = evaluate_predictions(result["predictions_path"], Path(args.output_dir) / "eval")
    result["report_path"] = report["report_path"]
    result["gate"] = report["gate"]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
