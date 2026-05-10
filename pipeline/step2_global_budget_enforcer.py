"""Step 2: run HUMBE global budget enforcer for train/val/test splits.

This is a thin wrapper over:
    src.downstream_task.preprocess.humbe_preprocess_pipeline
"""

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.downstream_task.preprocess.humbe_preprocess_pipeline import main as humbe_main


def run_split(
    images_dir: str,
    output_dir: str,
    budget: float,
    score: str,
    force: bool,
    verbose: bool,
) -> None:
    args = Namespace(
        images_dir=images_dir,
        output_dir=output_dir,
        budget=budget,
        score=score,
        force=force,
        verbose=verbose,
    )
    humbe_main(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HUMBE preprocessing across train/val/test splits"
    )
    parser.add_argument(
        "--images-root",
        default="/Volumes/Xbox_HD/Data/med_img",
        help="Root containing train/val/test .svs folders",
    )
    parser.add_argument(
        "--output-root",
        default="/Volumes/Xbox_HD/Data/extracted/humbe",
        help="Root output directory for HUMBE .pt files",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated split names",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.8,
        help="HUMBE budget ratio",
    )
    parser.add_argument(
        "--score",
        default="text_align_score",
        help="Patch score module key",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite outputs")
    parser.add_argument("--verbose", action="store_true", help="Verbose errors")
    args = parser.parse_args()

    images_root = os.path.abspath(args.images_root)
    output_root = os.path.abspath(args.output_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        split_images = os.path.join(images_root, split)
        split_output = os.path.join(output_root, split)
        if not os.path.isdir(split_images):
            print(f"[WARN] Missing split dir: {split_images} (skipping)")
            continue
        print(f"\n[STEP2] split={split} images={split_images} out={split_output}")
        run_split(
            images_dir=split_images,
            output_dir=split_output,
            budget=args.budget,
            score=args.score,
            force=args.force,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
