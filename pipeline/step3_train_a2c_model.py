"""Step 3: train A2C model from .svs images.

This is a thin wrapper over:
    src.training.rl.a2c.train_a2c.train_a2c
"""

import argparse
import os
import sys
from pathlib import Path

import torch

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.training.rl.a2c.train_a2c import train_a2c


def main() -> None:
    parser = argparse.ArgumentParser(description="Train A2C model from .svs slides")
    parser.add_argument(
        "--images-dir",
        default="/Volumes/Xbox_HD/Data/med_img/train",
        help="Directory containing training .svs images",
    )
    parser.add_argument(
        "--patch-score",
        default="text_align_score",
        help="Patch score module key",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--episodes-per-image",
        type=int,
        default=5,
        help="Episodes per image per epoch",
    )
    parser.add_argument(
        "--output-dir",
        default="/Volumes/Xbox_HD/Data/models/rl/a2c_lvl4",
        help="Output directory for checkpoints/final model",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override: cuda | mps | cpu",
    )
    parser.add_argument(
        "--redundancy-penalty",
        type=float,
        default=0.2,
        help="Redundancy penalty weight",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.3,
        help="Spatial overlap threshold",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda",
    )
    args = parser.parse_args()

    images_dir = os.path.abspath(args.images_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Training images directory not found: {images_dir}")

    device = torch.device(args.device) if args.device else None

    train_a2c(
        images_dir=images_dir,
        patch_score=args.patch_score,
        num_epochs=args.epochs,
        episodes_per_image=args.episodes_per_image,
        output_dir=output_dir,
        device=device,
        redundancy_penalty=args.redundancy_penalty,
        overlap_threshold=args.overlap_threshold,
        gae_lambda=args.gae_lambda,
    )


if __name__ == "__main__":
    main()
