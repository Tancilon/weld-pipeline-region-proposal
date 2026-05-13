#!/usr/bin/env python
"""Train the CatSpec-Pose P2 lightweight head."""

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
    parser.add_argument("--manifest", default=None, help="P2 manifest path. If omitted, preprocess from --dataset-root.")
    parser.add_argument(
        "--dataset-root",
        default="/home/dq/mnt/localgit/aiws5.2/datasets/obj_share_models",
        help="Dataset root used to create/reuse a P2 manifest when --manifest is omitted.",
    )
    parser.add_argument(
        "--preprocess-output-dir",
        default="results/catspec/p2_preprocess",
        help="Preprocess output directory used when --manifest is omitted.",
    )
    parser.add_argument("--output-dir", default="results/catspec/p2_train", help="Training output directory.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"], help="Training split.")
    parser.add_argument("--val-split", default="val", choices=["train", "val", "test", "all"], help="Validation split.")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--resume-checkpoint", default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--use-geometry", action="store_true", help="Concatenate source geometry summary to spec embedding.")
    args = parser.parse_args()

    from catspec.p2_pipeline import train_p2
    from catspec.p2_preprocess import preprocess_p2_dataset

    manifest = args.manifest
    if manifest is None:
        preprocess = preprocess_p2_dataset(args.dataset_root, args.preprocess_output_dir, seed=args.seed, force=False)
        manifest = preprocess["manifest_path"]

    result = train_p2(
        manifest,
        args.output_dir,
        split=args.split,
        val_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        resume_checkpoint=args.resume_checkpoint,
        use_geometry=args.use_geometry,
    )
    print(
        json.dumps(
            {
                "latest_checkpoint_path": result["latest_checkpoint_path"],
                "best_checkpoint_path": result["best_checkpoint_path"],
                "metrics_path": result["metrics_path"],
                "log_path": result["log_path"],
                "manifest_path": manifest,
                "train_accuracy": result["metrics"]["train_accuracy"],
                "val_accuracy": result["metrics"]["val_accuracy"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
