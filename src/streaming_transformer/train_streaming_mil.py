"""Module for train streaming mil."""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import (
    WSIEmbeddingDataset,
    build_items_from_labels_json,
    build_items_from_pt_labels,
    build_items_from_pt_labels_with_map,
    collate_wsi_batch,
    split_items,
)
from engine import evaluate, train_one_epoch
from model import StreamingMILTransformer


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


def _is_improved(
    current: float,
    best: float | None,
    mode: str,
    min_delta: float,
) -> bool:
    if not math.isfinite(current):
        return False
    if best is None:
        return True
    if mode == "min":
        return current < (best - min_delta)
    return current > (best + min_delta)


def _remap_state_dict(state_dict: dict) -> dict:
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("local_layers."):
            new_key = "local_encoder.layers." + key[len("local_layers."):]
            remapped[new_key] = value
        else:
            remapped[key] = value
    return remapped


def _invert_label_map(label_map: Dict[str, int]) -> Dict[int, str]:
    return {idx: name for name, idx in label_map.items()}


def _summarize_confusions(
    conf_mat: np.ndarray,
    idx_to_label: Dict[int, str],
    top_k: int = 5,
) -> str:
    confusions = []
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            if i == j:
                continue
            count = int(conf_mat[i, j])
            if count > 0:
                confusions.append((count, i, j))
    confusions.sort(reverse=True)
    parts = []
    for count, i, j in confusions[:top_k]:
        parts.append(
            f"{idx_to_label.get(i, str(i))}->{idx_to_label.get(j, str(j))}:{count}"
        )
    return ", ".join(parts) if parts else "none"


def _summarize_rare_classes(
    train_class_counts: np.ndarray,
    val_conf_mat: np.ndarray,
    val_per_class_f1: np.ndarray,
    idx_to_label: Dict[int, str],
    max_items: int = 8,
) -> str:
    rare_idx = [i for i, c in enumerate(train_class_counts) if int(c) <= 20]
    if not rare_idx:
        return "none"
    val_support = val_conf_mat.sum(axis=1)
    diag = np.diag(val_conf_mat)
    parts = []
    for i in sorted(rare_idx, key=lambda k: (train_class_counts[k], idx_to_label.get(k, ""))):
        support = int(val_support[i])
        recall = float(diag[i] / support) if support > 0 else float("nan")
        f1 = float(val_per_class_f1[i]) if i < len(val_per_class_f1) else float("nan")
        parts.append(
            f"{idx_to_label.get(i, str(i))}:train={int(train_class_counts[i])},val={support},recall={recall:.3f},f1={f1:.3f}"
        )
    return "; ".join(parts[:max_items])


def _summarize_pred_distribution(
    preds: np.ndarray,
    idx_to_label: Dict[int, str],
    top_k: int = 5,
) -> str:
    if preds.size == 0:
        return "none"
    unique, counts = np.unique(preds, return_counts=True)
    order = np.argsort(counts)[::-1]
    parts = []
    for idx in order[:top_k]:
        label_idx = int(unique[idx])
        parts.append(f"{idx_to_label.get(label_idx, str(label_idx))}:{int(counts[idx])}")
    return ", ".join(parts)


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

    # ---------------- Data & loaders ----------------
    train_ds = WSIEmbeddingDataset(
        train_items,
        args.train_embeddings_dir,
        images_dir=args.train_images_dir,
    )
    val_ds = WSIEmbeddingDataset(
        val_items,
        args.val_embeddings_dir,
        images_dir=args.val_images_dir,
    )
    test_ds = WSIEmbeddingDataset(
        test_items,
        args.test_embeddings_dir,
        images_dir=args.test_images_dir,
    )

    # === Imbalance: WeightedRandomSampler ===
    train_labels = [item.label for item in train_items]
    class_counts = np.bincount(train_labels, minlength=len(label_map)).astype(
        np.float32
    )
    class_counts[class_counts == 0.0] = 1.0
    class_weights = (len(train_labels) / (len(label_map) * class_counts)).astype(
        np.float32
    )
    sample_weights = torch.tensor(
        [class_weights[label] for label in train_labels],
        dtype=torch.float32,
    )
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

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
    model = StreamingMILTransformer(
        embed_dim=args.embed_dim,
        model_dim=args.model_dim,
        num_classes=len(label_map),
        patch_chunk_size=args.patch_batch_size,
        local_num_heads=args.local_heads,
        local_num_layers=args.local_layers,
        local_dropout=args.local_dropout,
    ).to(device)
    model.set_logit_adjustment(
        torch.tensor(class_counts, device=device),
        tau=args.logit_adjust_tau,
    )
    print(f"[LOGIT ADJUST] tau={args.logit_adjust_tau}")

    # ---------------- Optimization ----------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # === Imbalance: weighted sampler only (plain CE) ===
    print(f"[CLASS WEIGHTS] {class_weights.tolist()}")
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
        f"classes={len(label_map)} wsi_batch_size={args.wsi_batch_size} patch_batch_size={args.patch_batch_size}"
    )
    # === Logging: class distribution ===
    idx_to_label = _invert_label_map(label_map)
    class_summary = sorted(
        ((idx_to_label.get(i, str(i)), int(c)) for i, c in enumerate(class_counts)),
        key=lambda x: x[1],
        reverse=True,
    )
    print(f"[CLASS COUNTS] {class_summary}")

    best_val_score = None
    best_path = os.path.join(args.out_dir, "best_streaming_mil.pt")
    epochs_no_improve = 0
    history = []

    # ---------------- Training loop ----------------
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            progress_prefix=None,
            progress_interval=args.train_log_interval,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_elapsed_sec = time.perf_counter() - epoch_start

        row = {
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
        history.append(row)

        # === Logging: epoch summary ===
        train_val_f1_gap = train_metrics["f1"] - val_metrics["f1"]
        train_val_loss_gap = train_metrics["loss"] - val_metrics["loss"]
        val_confusions = _summarize_confusions(
            val_metrics["confusion_matrix"],
            idx_to_label,
        )
        rare_class_report = _summarize_rare_classes(
            class_counts,
            val_metrics["confusion_matrix"],
            val_metrics["per_class_f1"],
            idx_to_label,
        )
        pred_distribution = _summarize_pred_distribution(
            val_metrics["preds"],
            idx_to_label,
        )
        print(
            f"[EPOCH {epoch:03d}] "
            f"train({_metrics_brief(train_metrics)}) "
            f"val({_metrics_brief(val_metrics)}) "
            f"gap_f1={train_val_f1_gap:.4f} gap_loss={train_val_loss_gap:.4f} "
            f"time={epoch_elapsed_sec:.2f}s"
        )
        print(f"[VAL CONFUSIONS] {val_confusions}")
        print(f"[VAL PRED DISTRIB] {pred_distribution}")
        print(f"[RARE CLASS REPORT] {rare_class_report}")

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
                        "patch_batch_size": args.patch_batch_size,
                        "memory_slots": args.memory_slots,
                        "local_heads": args.local_heads,
                        "local_layers": args.local_layers,
                        "local_dropout": args.local_dropout,
                        "monitor_metric": args.monitor_metric,
                        "monitor_mode": monitor_mode,
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
        else:
            print(
                f"[WARN] val_{args.monitor_metric} is non-finite; skipping LR scheduler step."
            )

        if (
            args.early_stopping_patience > 0
            and epochs_no_improve >= args.early_stopping_patience
        ):
            print(
                f"[EARLY STOP] No val_{args.monitor_metric} improvement for "
                f"{epochs_no_improve} epochs."
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
                    "patch_batch_size": args.patch_batch_size,
                    "memory_slots": args.memory_slots,
                    "local_heads": args.local_heads,
                    "local_layers": args.local_layers,
                    "local_dropout": args.local_dropout,
                    "monitor_metric": args.monitor_metric,
                    "monitor_mode": monitor_mode,
                },
                "best_epoch": None,
                "best_val_score": None,
            },
            best_path,
        )
        print(
            "[WARN] No valid best checkpoint found during training; saved final model."
        )

    # ---------------- Test & report ----------------
    ckpt = torch.load(best_path, map_location=device)
    state_dict = _remap_state_dict(ckpt["model_state_dict"])
    model.load_state_dict(state_dict, strict=False)
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"[TEST] {_metrics_brief(test_metrics)}")

    result = {
        "split": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
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
        "history": history,
    }

    result_path = os.path.join(args.out_dir, "train_result_streaming_mil.json")
    with open(result_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"[SAVED] best checkpoint: {best_path}")
    print(f"[SAVED] train result: {result_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train streaming MIL transformer with per-WSI CLS streaming"
    )
    parser.add_argument(
        "--train-embeddings-dir",
        default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/full/train",
        help="Directory with per-WSI .pt files containing train embeddings",
    )
    parser.add_argument(
        "--val-embeddings-dir",
        default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/full/val",
        help="Directory with per-WSI .pt files containing validation embeddings",
    )
    parser.add_argument(
        "--test-embeddings-dir",
        default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/full/test",
        help="Directory with per-WSI .pt files containing test embeddings",
    )

    parser.add_argument(
        "--labels-json",
        default=None,
        help="Optional JSON mapping case_id -> class name. If omitted, labels are read from PT files.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/models/downstream_tasks/streaming_mil_v1_3_2",
        help="Output directory for checkpoint and metrics",
    )

    parser.add_argument(
        "--train-images-dir",
        default=None,
        help="Optional image directory fallback for PT files that contain active_patches but missing img_path",
    )
    parser.add_argument(
        "--val-images-dir",
        default=None,
        help="Optional image directory fallback for PT files that contain active_patches but missing img_path",
    )
    parser.add_argument(
        "--test-images-dir",
        default=None,
        help="Optional image directory fallback for PT files that contain active_patches but missing img_path",
    )

    parser.add_argument("--no-stratified-split", action="store_true")

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--wsi-batch-size", type=int, default=16)
    parser.add_argument("--patch-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=5000000000)
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum improvement required on monitored validation metric.",
    )
    parser.add_argument(
        "--monitor-metric",
        choices=["loss", "accuracy", "balanced_accuracy", "macro_f1", "f1", "auc"],
        default="f1",
        help="Validation metric used for checkpointing, LR scheduling, and early stopping.",
    )
    parser.add_argument(
        "--monitor-mode",
        choices=["auto", "min", "max"],
        default="auto",
        help="Optimization direction for monitor metric. 'auto': min for loss, max otherwise.",
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=2,
        help="Epochs without monitor improvement before reducing learning rate.",
    )
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=0.5,
        help="Multiplicative LR decay factor when plateau is detected.",
    )
    parser.add_argument(
        "--lr-scheduler-threshold",
        type=float,
        default=1e-4,
        help="Minimum monitor change to qualify as LR scheduler improvement.",
    )
    parser.add_argument(
        "--lr-scheduler-min-lr",
        type=float,
        default=1e-6,
        help="Lower bound for learning rate during ReduceLROnPlateau.",
    )

    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument(
        "--memory-slots",
        type=int,
        default=4,
        help="Number of persistent memory tokens carried across chunks.",
    )
    parser.add_argument(
        "--allow-unknown-eval-labels",
        action="store_true",
        help="Allow and skip validation/test labels not present in train label_map.",
    )

    parser.add_argument("--local-heads", type=int, default=8)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--local-dropout", type=float, default=0.1)

    parser.add_argument(
        "--logit-adjust-tau",
        type=float,
        default=1.0,
        help="Logit adjustment strength based on class priors",
    )

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--train-log-interval",
        type=int,
        default=10,
        help="Batch interval for train progress logging",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
