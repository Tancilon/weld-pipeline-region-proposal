#!/usr/bin/env python
"""Run CatSpec-Pose P2 inference."""

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
    parser.add_argument("--manifest", required=True, help="P2 manifest path.")
    parser.add_argument("--checkpoint", required=True, help="P2 checkpoint path.")
    parser.add_argument("--output-dir", default="results/catspec/p2_infer", help="Inference output directory.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="Split to infer.")
    parser.add_argument(
        "--mode",
        action="append",
        dest="modes",
        choices=["spec_correct", "spec_shuffle", "no_spec"],
        help="Inference mode. May be repeated. Defaults to all modes.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    args = parser.parse_args()

    from catspec.p2_pipeline import infer_p2

    result = infer_p2(
        args.manifest,
        args.checkpoint,
        args.output_dir,
        split=args.split,
        modes=args.modes,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
