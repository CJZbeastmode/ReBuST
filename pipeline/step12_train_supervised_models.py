"""Step 12: train supervised baselines (score regressor or zoom classifier)."""

import argparse
import os
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.training.supervised.score_regressor import train as train_score_regressor
from src.training.supervised.zoom_classifier import train as train_zoom_classifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train supervised baselines (score regressor or zoom classifier)"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["score_regressor", "zoom_classifier"],
        help="Which supervised model to train",
    )
    parser.add_argument("--data-npz", required=True, help="Path to .npz dataset")
    parser.add_argument("--model-out", required=True, help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)

    if args.model == "score_regressor":
        train_score_regressor(
            device=args.device,
            epochs=args.epochs,
            data_npz=args.data_npz,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            model_out=args.model_out,
            random_seed=args.seed,
        )
        return

    train_zoom_classifier(
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        data_npz=args.data_npz,
        model_out=args.model_out,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
