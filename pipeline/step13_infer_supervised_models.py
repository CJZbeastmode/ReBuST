"""Step 13: infer supervised baselines (score regressor or zoom classifier)."""

import argparse
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.inference.supervised.supervised_score_regressor_infer import (
    greedy_infer_wsi_regressor,
)
from src.inference.supervised.supervised_zoom_classifier_infer import (
    greedy_infer_wsi_regressor as greedy_zoom_classifier,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer supervised baselines (score regressor or zoom classifier)"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["score_regressor", "zoom_classifier"],
        help="Which supervised model to run",
    )
    parser.add_argument("--image", required=True, help="Path to .svs slide")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument(
        "--output-viz-path",
        required=True,
        help="Output HTML visualization path",
    )
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()

    if args.model == "score_regressor":
        greedy_infer_wsi_regressor(
            image_path=args.image,
            model_path=args.checkpoint,
            output_viz_path=args.output_viz_path,
            max_depth=args.max_depth,
        )
        return

    greedy_zoom_classifier(
        image_path=args.image,
        model_path=args.checkpoint,
        output_viz_path=args.output_viz_path,
        max_depth=args.max_depth,
        output_dir=None,
    )


if __name__ == "__main__":
    main()
