"""Supervised zoom-regressor preprocessing pipeline."""

import argparse
import os
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.downstream_task.preprocess.supervised_preprocess_pipeline import main

DEFAULT_IMAGES_DIR = "/Volumes/Xbox_HD/Data/med_img/test"
DEFAULT_OUTPUT_DIR = "/Volumes/Xbox_HD/Data/extracted/supervised_zoom/test"
DEFAULT_MODEL = "data/models/supervised/zoom_classifier.pth"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Infer using supervised zoom regressor")
    p.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--level-mode",
        choices=["root_only", "finest_only"],
        default="finest_only",
    )
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()
    args.supervised_model_type = "zoom_regressor"
    args.images_dir = os.path.abspath(args.images_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    main(args)
