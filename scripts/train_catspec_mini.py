#!/usr/bin/env python
"""Train the CatSpec-Pose mini spec-conditioned lightweight head."""

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
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoint, train metrics, and log.")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic random seed.")
    parser.add_argument("--lr", type=float, default=0.03, help="Adam learning rate.")
    args = parser.parse_args()

    from catspec.mini_pipeline import train_mini

    result = train_mini(
        args.manifest,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        lr=args.lr,
    )
    print(
        json.dumps(
            {
                "checkpoint_path": result["checkpoint_path"],
                "metrics_path": result["metrics_path"],
                "log_path": result["log_path"],
                "train_accuracy": result["metrics"]["train_accuracy"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
