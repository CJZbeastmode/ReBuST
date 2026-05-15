"""Train aggregation transformer model on per-WSI embedding bags."""

import argparse
import json
import math
import os
import random
import time
from typing import Dict
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from data import (
    WSIEmbeddingDataset,
    build_items_from_pt_labels,
    build_items_from_pt_labels_with_map,
    collate_wsi_batch,
)
from engine import evaluate, train_one_epoch
from model import AggregationTransformer, PureMIL


# Set seed for reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Helper functions for monitoring and reporting
def _metrics_brief(metrics: Dict) -> str:
    return (
        f"loss={metrics['loss']:.4f} "
        f"acc={metrics['accuracy']:.4f} "
        f"bacc={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"auc={metrics['auc']:.4f}"
    )


# Resolve monitor mode from metric name when mode is 'auto'
def _resolve_monitor_mode(metric_name: str, mode: str) -> str:
    if mode in {"min", "max"}:
        return mode
    if metric_name == "loss":
        return "min"
    return "max"


# Check if current score is strictly better than best within min_delta tolerance
def _is_improved(
    current: float, best: float | None, mode: str, min_delta: float
) -> bool:
    if not math.isfinite(current):
        return False
    if best is None:
        return True
    if mode == "min":
        return current < (best - min_delta)
    return current > (best + min_delta)


def train(args) -> None:
    set_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    method = str(args.method).upper()
    pure = method in {"ABMIL_PURE", "CLAM_PURE"}
    pooling_type = "clam" if method in {"CLAM", "CLAM_PURE"} else "abmil"

    # ---------------- Load data ----------------
    train_items, label_map = build_items_from_pt_labels(args.train_embeddings_dir)
    val_items = build_items_from_pt_labels_with_map(
        args.val_embeddings_dir,
        label_map,
        strict=not args.allow_unknown_eval_labels,
    )
    test_items = build_items_from_pt_labels_with_map(
        args.test_embeddings_dir,
        label_map,
        strict=not args.allow_unknown_eval_labels,
    )

    train_ds = WSIEmbeddingDataset(
        train_items,
        args.train_embeddings_dir,
        max_patches_per_wsi=args.max_patches_per_wsi,
        patch_sample_mode=args.patch_sample_mode,
        sample_seed=args.seed,
    )
    val_ds = WSIEmbeddingDataset(
        val_items,
        args.val_embeddings_dir,
        max_patches_per_wsi=args.max_patches_per_wsi,
        patch_sample_mode="uniform",
        sample_seed=args.seed,
    )
    test_ds = WSIEmbeddingDataset(
        test_items,
        args.test_embeddings_dir,
        max_patches_per_wsi=args.max_patches_per_wsi,
        patch_sample_mode="uniform",
        sample_seed=args.seed,
    )

    # === Imbalance: WeightedRandomSampler + logit adjustment ===
    train_labels = [item.label for item in train_items]
    class_counts = np.bincount(train_labels, minlength=len(label_map)).astype(
        np.float32
    )
    class_counts[class_counts == 0.0] = 1.0
    class_weights = (len(train_labels) / (len(label_map) * class_counts)).astype(
        np.float32
    )
    sample_weights = torch.tensor(
        [class_weights[lbl] for lbl in train_labels], dtype=torch.float32
    )
    train_sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    print(f"[CLASS WEIGHTS] {class_weights.tolist()}")

    # ---------------- Data & loaders ----------------
    train_loader = DataLoader(
        train_ds,
        batch_size=args.wsi_batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_wsi_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.wsi_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wsi_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.wsi_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wsi_batch,
    )

    # ---------------- Model ----------------
    if pure:
        model = PureMIL(
            embed_dim=args.embed_dim,
            model_dim=args.model_dim,
            num_classes=len(label_map),
            dropout=args.dropout,
            use_coords=not args.no_coords,
            attn_dim=args.attn_dim,
            pooling_type=pooling_type,
        ).to(device)
    else:
        model = AggregationTransformer(
            embed_dim=args.embed_dim,
            model_dim=args.model_dim,
            num_classes=len(label_map),
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_coords=not args.no_coords,
            attn_dim=args.attn_dim,
            pooling_type=pooling_type,
            wikg_k=args.wikg_k,
            wikg_steps=args.wikg_steps,
            wikg_temperature=args.wikg_temperature,
        ).to(device)

    # ---------------- Optimization ----------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    model.set_logit_adjustment(
        torch.tensor(class_counts, device=device), tau=args.logit_adjust_tau
    )
    print(f"[LOGIT ADJUST] tau={args.logit_adjust_tau}")

    criterion = nn.CrossEntropyLoss()
    monitor_mode = _resolve_monitor_mode(args.monitor_metric, args.monitor_mode)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=monitor_mode,
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        threshold=args.lr_scheduler_threshold,
        min_lr=args.lr_scheduler_min_lr,
    )

    print(
        f"[DATA] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"classes={len(label_map)} batch={args.wsi_batch_size} max_patches_per_wsi={args.max_patches_per_wsi}"
    )

    best_val_score = None
    best_path = os.path.join(
        args.out_dir,
        f"best_aggregation_transformer_{method.lower()}.pt",
    )
    epochs_no_improve = 0
    history = []

    # ---------------- Training loop ----------------
    for epoch in range(1, args.epochs + 1):
        print(f"\n[EPOCH {epoch:03d}] Starting epoch {epoch}...")
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            progress_prefix=f"TRAIN EPOCH {epoch:03d}",
            progress_interval=args.train_log_interval,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train": {
                    "loss": train_metrics["loss"],
                    "accuracy": train_metrics["accuracy"],
                    "balanced_accuracy": train_metrics["balanced_accuracy"],
                    "macro_f1": train_metrics["macro_f1"],
                    "f1": train_metrics["f1"],
                    "auc": train_metrics["auc"],
                },
                "val": {
                    "loss": val_metrics["loss"],
                    "accuracy": val_metrics["accuracy"],
                    "balanced_accuracy": val_metrics["balanced_accuracy"],
                    "macro_f1": val_metrics["macro_f1"],
                    "f1": val_metrics["f1"],
                    "auc": val_metrics["auc"],
                },
            }
        )

        print(
            f"[EPOCH {epoch:03d}] "
            f"train({_metrics_brief(train_metrics)}) "
            f"val({_metrics_brief(val_metrics)}) "
            f"time={elapsed:.2f}s"
        )

        current_val_score = float(val_metrics[args.monitor_metric])
        current_lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"[MONITOR] metric=val_{args.monitor_metric} "
            f"value={current_val_score:.6f} mode={monitor_mode} lr={current_lr:.8f}"
        )

        improved = _is_improved(
            current=current_val_score,
            best=best_val_score,
            mode=monitor_mode,
            min_delta=args.early_stopping_min_delta,
        )
        if improved:
            best_val_score = current_val_score
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_map": label_map,
                    "config": {
                        "model_type": "pure_mil" if pure else "aggregation_transformer",
                        "embed_dim": args.embed_dim,
                        "model_dim": args.model_dim,
                        "num_heads": args.num_heads,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "attn_dim": args.attn_dim,
                        "pooling_type": pooling_type,
                        "wikg_k": args.wikg_k,
                        "wikg_steps": args.wikg_steps,
                        "wikg_temperature": args.wikg_temperature,
                        "use_coords": not args.no_coords,
                        "max_patches_per_wsi": args.max_patches_per_wsi,
                        "patch_sample_mode": args.patch_sample_mode,
                        "monitor_metric": args.monitor_metric,
                        "monitor_mode": monitor_mode,
                        "logit_adjust_tau": args.logit_adjust_tau,
                    },
                    "best_epoch": epoch,
                    "best_val_score": best_val_score,
                },
                best_path,
            )
            print(
                f"[BEST] Saved new best checkpoint at epoch {epoch} "
                f"(val_{args.monitor_metric}={best_val_score:.6f})"
            )
        else:
            epochs_no_improve += 1

        if math.isfinite(current_val_score):
            scheduler.step(current_val_score)

        if (
            args.early_stopping_patience > 0
            and epochs_no_improve >= args.early_stopping_patience
        ):
            print(
                f"[EARLY STOP] No val_{args.monitor_metric} improvement for {epochs_no_improve} epochs."
            )
            break

    if not os.path.exists(best_path):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "label_map": label_map,
                "config": {
                    "embed_dim": args.embed_dim,
                    "model_dim": args.model_dim,
                    "num_heads": args.num_heads,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "attn_dim": args.attn_dim,
                    "pooling_type": pooling_type,
                    "wikg_k": args.wikg_k,
                    "wikg_steps": args.wikg_steps,
                    "wikg_temperature": args.wikg_temperature,
                    "use_coords": not args.no_coords,
                    "max_patches_per_wsi": args.max_patches_per_wsi,
                    "patch_sample_mode": args.patch_sample_mode,
                    "monitor_metric": args.monitor_metric,
                    "monitor_mode": monitor_mode,
                },
                "best_epoch": None,
                "best_val_score": None,
            },
            best_path,
        )

    # ---------------- Test & report ----------------
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"[TEST] {_metrics_brief(test_metrics)}")

    result = {
        "split": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "label_map": label_map,
        "best_epoch": ckpt.get("best_epoch"),
        "best_val_metric": args.monitor_metric,
        "best_val_score": ckpt.get("best_val_score"),
        "test": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "f1": test_metrics["f1"],
            "auc": test_metrics["auc"],
            "per_class_f1": test_metrics["per_class_f1"].tolist(),
            "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        },
        "logit_adjust_tau": args.logit_adjust_tau,
        "method": method,
        "history": history,
    }

    result_path = os.path.join(
        args.out_dir,
        f"train_result_aggregation_transformer_{method.lower()}.json",
    )
    with open(result_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"[SAVED] best checkpoint: {best_path}")
    print(f"[SAVED] train result: {result_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train aggregation transformer (ABMIL/CLAM pooling)"
    )
    p.add_argument(
        "--train-embeddings-dir",
        default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/a2c/train",
    )
    p.add_argument(
        "--val-embeddings-dir",
        default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/a2c/val",
    )
    p.add_argument(
        "--test-embeddings-dir",
        default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/a2c/test",
    )
    p.add_argument(
        "--out-dir", default="data/models/ablation_classifier/aggregation_transformer"
    )

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--wsi-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--train-log-interval",
        type=int,
        default=10,
        help="Batch interval for train progress logging",
    )

    p.add_argument(
        "--method",
        choices=["CLAM", "ABMIL", "CLAM_PURE", "ABMIL_PURE"],
        required=True,
        help="CLAM/ABMIL = transformer encoder + pooling; CLAM_PURE/ABMIL_PURE = pooling only (true baselines)",
    )

    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--model-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--attn-dim", type=int, default=128)
    p.add_argument("--wikg-k", type=int, default=16)
    p.add_argument("--wikg-steps", type=int, default=2)
    p.add_argument("--wikg-temperature", type=float, default=1.0)
    p.add_argument("--no-coords", action="store_true")

    p.add_argument("--max-patches-per-wsi", type=int, default=2048)
    p.add_argument(
        "--patch-sample-mode",
        choices=["uniform", "random", "head"],
        default="uniform",
    )

    p.add_argument(
        "--logit-adjust-tau",
        type=float,
        default=1.0,
        help="Logit adjustment tau for class prior correction (0 = disabled)",
    )

    p.add_argument(
        "--monitor-metric",
        choices=["loss", "accuracy", "balanced_accuracy", "macro_f1", "f1", "auc"],
        default="macro_f1",
    )
    p.add_argument("--monitor-mode", choices=["auto", "min", "max"], default="auto")
    p.add_argument("--early-stopping-patience", type=int, default=8)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)

    p.add_argument("--lr-scheduler-patience", type=int, default=2)
    p.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    p.add_argument("--lr-scheduler-threshold", type=float, default=1e-4)
    p.add_argument("--lr-scheduler-min-lr", type=float, default=1e-6)

    p.add_argument("--allow-unknown-eval-labels", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
