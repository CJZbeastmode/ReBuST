"""Step 4: run A2C inference from HUMBE outputs for train/val/test splits.

This is a thin wrapper over:
    src.downstream_task.preprocess.full_preprocess_pipeline
"""

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.downstream_task.preprocess.full_preprocess_pipeline import main as full_main


def run_split(
    images_dir: str,
    humbe_dir: str,
    output_dir: str,
    model: str,
    stochastic: bool,
    force: bool,
    verbose: bool,
) -> None:
    args = Namespace(
        images_dir=images_dir,
        humbe_dir=humbe_dir,
        output_dir=output_dir,
        model=model,
        stochastic=stochastic,
        force=force,
        verbose=verbose,
    )
    full_main(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run A2C inference on top of HUMBE outputs across splits"
    )
    parser.add_argument(
        "--images-root",
        default="/Volumes/Xbox_HD/Data/med_img",
        help="Root containing train/val/test .svs folders",
    )
    parser.add_argument(
        "--humbe-root",
        default="/Volumes/Xbox_HD/Data/extracted/humbe",
        help="Root containing HUMBE .pt outputs per split",
    )
    parser.add_argument(
        "--output-root",
        default="/Volumes/Xbox_HD/Data/extracted/a2c",
        help="Root output directory for A2C-refined .pt files",
    )
    parser.add_argument(
        "--model",
        default="data/models/rl/a2c/a2c.pt",
        help="A2C checkpoint path",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated split names",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite outputs")
    parser.add_argument("--verbose", action="store_true", help="Verbose errors")
    args = parser.parse_args()

    images_root = os.path.abspath(args.images_root)
    humbe_root = os.path.abspath(args.humbe_root)
    output_root = os.path.abspath(args.output_root)
    model = os.path.abspath(args.model)

    if not os.path.isfile(model):
        raise FileNotFoundError(f"A2C checkpoint not found: {model}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        split_images = os.path.join(images_root, split)
        split_humbe = os.path.join(humbe_root, split)
        split_output = os.path.join(output_root, split)

        if not os.path.isdir(split_images):
            print(f"[WARN] Missing image split dir: {split_images} (skipping)")
            continue
        if not os.path.isdir(split_humbe):
            print(f"[WARN] Missing HUMBE split dir: {split_humbe} (skipping)")
            continue

        print(
            f"\n[STEP4] split={split} images={split_images} humbe={split_humbe} out={split_output}"
        )
        run_split(
            images_dir=split_images,
            humbe_dir=split_humbe,
            output_dir=split_output,
            model=model,
            stochastic=args.stochastic,
            force=args.force,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
