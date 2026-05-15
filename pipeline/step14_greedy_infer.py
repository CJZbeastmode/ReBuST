"""Step 14: greedy inference baseline using patch score modules."""

import argparse
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.inference.greedy.greedy_infer import greedy_infer_wsi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy zoom inference based on a patch score module"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="./data/to_test_image/test_img_1.svs",
        help="Path to .svs image",
    )
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--score-module", type=str, default="text_align_score")
    parser.add_argument("--viz-title", type=str, default=None)
    args = parser.parse_args()

    greedy_infer_wsi(
        args.image,
        max_depth=args.max_depth,
        output_dir=args.output_dir,
        score_module=args.score_module,
        viz_title=args.viz_title,
    )


if __name__ == "__main__":
    main()
