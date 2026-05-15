"""Step 0: download TCGA SVS slides.

Thin wrapper over src/downloader/image_downloader.py.
"""

import argparse
import os
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.downloader.image_downloader import ImageDownloader


def _split_projects(value: str) -> list[str]:
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TCGA SVS slides (one per case)."
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=1000,
        help="Target total number of slides to download",
    )
    parser.add_argument(
        "--data-dir",
        default="/Volumes/Xbox_HD/data/med_img",
        help="Output directory for downloaded .svs files",
    )
    parser.add_argument(
        "--projects",
        default="",
        help="Comma-separated TCGA project IDs (empty = defaults in downloader)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output for selected case IDs",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    projects = _split_projects(args.projects)

    if projects:
        downloader = ImageDownloader(
            target_total=args.target_total,
            PROJECTS=projects,
            DATA_DIR=data_dir,
            fetched_images_output_json=args.output_json,
        )
    else:
        downloader = ImageDownloader(
            target_total=args.target_total,
            DATA_DIR=data_dir,
            fetched_images_output_json=args.output_json,
        )
    downloader.run()


if __name__ == "__main__":
    main()
