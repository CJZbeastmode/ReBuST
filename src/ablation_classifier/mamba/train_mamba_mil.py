"""Train Mamba-style MIL model on per-WSI embedding bags."""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.ablation_classifier.mamba.data import (
    WSIEmbeddingDataset,
    build_items_from_pt_labels,
    build_items_from_pt_labels_with_map,
    collate_wsi_batch,
)
from src.ablation_classifier.mamba.engine import evaluate, train_one_epoch
from src.ablation_classifier.mamba.model import MambaMIL


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _metrics_brief(metrics: Dict) -> str:
    return (
        f"loss={metrics['loss']:.4f} "
        f"acc={metrics['accuracy']:.4f} "
        f"bacc={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"auc={metrics['auc']:.4f}"
    )


def _resolve_monitor_mode(metric_name: str, mode: str) -> str:
    if mode in {"min", "max"}:
        return mode
    if metric_name == "loss":
        return "min"
    return "max"


def _is_improved(current: float, best: float | None, mode: str, min_delta: float) -> bool:
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

    train_labels = [item.label for item in train_items]
    class_counts = np.bincount(train_labels, minlength=len(label_map)).astype(np.float32)
    class_counts[class_counts == 0.0] = 1.0
    class_weights_arr = (len(train_labels) / (len(label_map) * class_counts)).astype(np.float32)
    sample_weights = torch.tensor([class_weights_arr[lbl] for lbl in train_labels], dtype=torch.float32)
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

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

    model = MambaMIL(
        embed_dim=args.embed_dim,
        model_dim=args.model_dim,
        num_classes=len(label_map),
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_coords=not args.no_coords,
        attn_dim=args.attn_dim,
        expand_factor=args.expand_factor,
        conv_kernel_size=args.conv_kernel_size,
        d_state=args.d_state,
    ).to(device)

    model.set_logit_adjustment(torch.tensor(class_counts, device=device), tau=args.logit_adjust_tau)
    print(f"[LOGIT ADJUST] tau={args.logit_adjust_tau} class_counts={class_counts.tolist()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    best_path = os.path.join(args.out_dir, "best_mamba_mil.pt")
    epochs_no_improve = 0
    history = []

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
            grad_accum_steps=args.grad_accum_steps,
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
                        "embed_dim": args.embed_dim,
                        "model_dim": args.model_dim,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "attn_dim": args.attn_dim,
                        "use_coords": not args.no_coords,
                        "expand_factor": args.expand_factor,
                        "conv_kernel_size": args.conv_kernel_size,
                        "d_state": args.d_state,
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

        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
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
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "attn_dim": args.attn_dim,
                    "use_coords": not args.no_coords,
                    "expand_factor": args.expand_factor,
                    "conv_kernel_size": args.conv_kernel_size,
                    "d_state": args.d_state,
                    "max_patches_per_wsi": args.max_patches_per_wsi,
                    "patch_sample_mode": args.patch_sample_mode,
                    "monitor_metric": args.monitor_metric,
                    "monitor_mode": monitor_mode,
                    "logit_adjust_tau": args.logit_adjust_tau,
                },
                "best_epoch": None,
                "best_val_score": None,
            },
            best_path,
        )

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
        "history": history,
    }

    result_path = os.path.join(args.out_dir, "train_result_mamba_mil.json")
    with open(result_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"[SAVED] best checkpoint: {best_path}")
    print(f"[SAVED] train result: {result_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mamba-style MIL model")
    parser.add_argument(
        "--train-embeddings-dir",
        default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/a2c/train",
    )
    parser.add_argument(
        "--val-embeddings-dir",
        default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/a2c/val",
    )
    parser.add_argument(
        "--test-embeddings-dir",
        default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/a2c/test",
    )
    parser.add_argument("--out-dir", default="data/models/ablation_classifier/mamba")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--wsi-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-log-interval", type=int, default=10)
    parser.add_argument("--grad-accum-steps", type=int, default=1)

    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn-dim", type=int, default=128)
    parser.add_argument("--expand-factor", type=int, default=2)
    parser.add_argument("--conv-kernel-size", type=int, default=3)
    parser.add_argument("--d-state", type=int, default=8,
                        help="SSM state dimension (d_state in S6 selective scan)")
    parser.add_argument("--no-coords", action="store_true")

    parser.add_argument("--max-patches-per-wsi", type=int, default=512)
    parser.add_argument(
        "--patch-sample-mode",
        choices=["uniform", "random", "head"],
        default="uniform",
    )
    parser.add_argument("--logit-adjust-tau", type=float, default=1.0,
                        help="Logit adjustment tau (0 = disabled)")

    parser.add_argument(
        "--monitor-metric",
        choices=["loss", "accuracy", "balanced_accuracy", "macro_f1", "f1", "auc"],
        default="macro_f1",
    )
    parser.add_argument("--monitor-mode", choices=["auto", "min", "max"], default="auto")
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)

    parser.add_argument("--lr-scheduler-patience", type=int, default=2)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-threshold", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-6)

    parser.add_argument("--allow-unknown-eval-labels", action="store_true")
    parser.add_argument(
        "--device",
        default="mps" if __import__("torch").backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
