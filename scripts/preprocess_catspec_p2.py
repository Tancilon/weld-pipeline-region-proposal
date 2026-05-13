#!/usr/bin/env python
"""Preprocess CatSpec-Pose P2 OBJ assets into a manifest and splits."""

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
        "--dataset-root",
        default="/home/dq/mnt/localgit/aiws5.2/datasets/obj_share_models",
        help="Root containing category OBJ folders.",
    )
    parser.add_argument("--output-dir", default="results/catspec/p2_preprocess", help="Preprocess output directory.")
    parser.add_argument("--spec-root", default="specs/categories", help="Directory containing CatSpec YAML files.")
    parser.add_argument("--category", action="append", dest="categories", help="Category to include. May be repeated.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional maximum scanned OBJ pairs.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic split seed.")
    parser.add_argument("--force", action="store_true", help="Regenerate outputs instead of reusing cached manifest.")
    args = parser.parse_args()

    from catspec.p2_preprocess import preprocess_p2_dataset

    manifest = preprocess_p2_dataset(
        args.dataset_root,
        args.output_dir,
        spec_root=args.spec_root,
        categories=args.categories,
        max_samples=args.max_samples,
        seed=args.seed,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "sample_count": manifest["sample_count"],
                "failure_count": manifest["failure_count"],
                "manifest_path": manifest["manifest_path"],
                "sample_index_path": manifest["sample_index_path"],
                "split_path": manifest["split_path"],
                "stats_path": manifest["stats_path"],
                "failures_path": manifest["failures_path"],
                "cache_hit": manifest["cache_hit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
