"""End-to-end Raza dual attention pipeline for PT embeddings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

from src.ablation_patch_selector.DualAttention.infer_pipeline import run_inference
from src.ablation_patch_selector.DualAttention.train_pipeline import train_raza


def run_end_to_end(args: argparse.Namespace) -> Dict[str, object]:
    os.makedirs(args.out_dir, exist_ok=True)

    train_args = argparse.Namespace(
        **{
            "train_embeddings_dir": args.train_dir,
            "val_embeddings_dir": args.val_dir,
            "out_dir": str(Path(args.out_dir) / "train"),
            "input_format": "pt",
            "embed_dim": args.embed_dim,
            "soft_hidden_dim": args.soft_hidden_dim,
            "hard_hidden_dim": args.hard_hidden_dim,
            "coord_dim": args.coord_dim,
            "num_tiles": args.num_tiles,
            "num_glimpses": args.num_glimpses,
            "pool_multiplier": args.pool_multiplier,
            "noise_low": args.noise_low,
            "noise_high": args.noise_high,
            "min_tile_dist": args.min_tile_dist,
            "min_glimpse_dist": args.min_glimpse_dist,
            "soft_entropy_beta": args.soft_entropy_beta,
            "policy_entropy_weight": args.policy_entropy_weight,
            "soft_weight": args.soft_weight,
            "soft_decay": args.soft_decay,
            "max_patches_per_wsi": args.max_patches_per_wsi,
            "patch_sample_mode": args.patch_sample_mode,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "log_every": args.log_every,
        }
    )

    train_result = train_raza(train_args)
    checkpoint = train_result.get("best_path")
    if not checkpoint:
        raise RuntimeError("Training did not produce a checkpoint.")

    infer_args = argparse.Namespace(
        **{
            "embeddings_dir": args.test_dir,
            "checkpoint": checkpoint,
            "out_dir": str(Path(args.out_dir) / "infer"),
            "num_tiles": args.num_tiles,
            "num_glimpses": args.num_glimpses,
            "pool_multiplier": args.pool_multiplier,
            "min_tile_dist": args.min_tile_dist,
        }
    )

    infer_result = run_inference(infer_args)
    return {
        "train": train_result,
        "infer": infer_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end Raza dual attention pipeline"
    )
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--out-dir", default="data/ablation/raza_dual_attention")

    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--soft-hidden-dim", type=int, default=256)
    parser.add_argument("--hard-hidden-dim", type=int, default=256)
    parser.add_argument("--coord-dim", type=int, default=2)

    parser.add_argument("--num-tiles", type=int, default=12)
    parser.add_argument("--num-glimpses", type=int, default=6)
    parser.add_argument("--pool-multiplier", type=int, default=2)
    parser.add_argument("--noise-low", type=float, default=0.0)
    parser.add_argument("--noise-high", type=float, default=0.1)
    parser.add_argument("--min-tile-dist", type=float, default=0.0)
    parser.add_argument("--min-glimpse-dist", type=float, default=0.0)

    parser.add_argument("--soft-entropy-beta", type=float, default=0.1)
    parser.add_argument("--policy-entropy-weight", type=float, default=0.01)
    parser.add_argument("--soft-weight", type=float, default=1.0)
    parser.add_argument("--soft-decay", type=float, default=0.98)

    parser.add_argument("--max-patches-per-wsi", type=int, default=0)
    parser.add_argument(
        "--patch-sample-mode",
        type=str,
        default="uniform",
        choices=["uniform", "random", "head"],
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_end_to_end(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
