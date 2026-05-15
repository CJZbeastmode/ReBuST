"""EvoPS training pipeline (classifier only)."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.ablation_patch_selector.SASHA.data import (
    PTEmbeddingDirDataset,
    build_items_and_label_map,
    build_items_with_label_map,
    collate_samples,
)
from src.ablation_patch_selector.SASHA.models import HAFEDClassifier


# Set seed for reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Evaluate HAFED on a loader, returning loss and accuracy
def evaluate_hafed(
    model: HAFEDClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            for sample in batch:
                emb = sample.embeddings.to(device)
                mask = torch.ones(1, emb.shape[0], dtype=torch.bool, device=device)
                logits, _, _ = model(emb.unsqueeze(0), mask)
                target = torch.tensor(
                    [sample.label_idx], dtype=torch.long, device=device
                )
                loss = F.cross_entropy(logits, target)

                total += 1
                total_loss += float(loss.item())
                pred = int(torch.argmax(logits, dim=-1).item())
                correct += int(pred == sample.label_idx)

    if total == 0:
        return {"loss": float("inf"), "acc": 0.0}

    return {"loss": total_loss / total, "acc": correct / total}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EvoPS classifier")

    parser.add_argument(
        "--train-embeddings-dir", default="/Volumes/Xbox_HD/Data/med_img/train"
    )
    parser.add_argument(
        "--val-embeddings-dir", default="/Volumes/Xbox_HD/Data/med_img/val"
    )
    parser.add_argument(
        "--out-dir", default="data/models/ablation_patch_selector/evops"
    )
    parser.add_argument(
        "--input-format",
        type=str,
        default="auto",
        choices=["pt", "svs", "auto"],
        help="Input file type expected in train/val directories.",
    )
    parser.add_argument(
        "--svs-level-mode",
        type=str,
        default="root_only",
        choices=["root_only", "finest_only"],
        help="WSI level used when input-format includes .svs.",
    )
    parser.add_argument(
        "--svs-embed-backend",
        type=str,
        default="plip",
        choices=["plip", "conch"],
        help="Embedding backend for direct .svs loading.",
    )

    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)

    parser.add_argument("--max-patches-per-wsi", type=int, default=0)
    parser.add_argument(
        "--patch-sample-mode",
        type=str,
        default="uniform",
        choices=["uniform", "head", "random"],
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--logit-adjust-tau",
        type=float,
        default=1.0,
        help="Logit adjustment tau (0 = disabled)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_evops(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Load data ----------------
    train_items, label_map = build_items_and_label_map(
        args.train_embeddings_dir,
        input_format=args.input_format,
    )
    val_items = build_items_with_label_map(
        args.val_embeddings_dir,
        label_map=label_map,
        strict=False,
        input_format=args.input_format,
    )

    train_ds = PTEmbeddingDirDataset(
        items=train_items,
        max_patches_per_wsi=args.max_patches_per_wsi,
        sample_mode=args.patch_sample_mode,
        sample_seed=args.seed,
        svs_level_mode=args.svs_level_mode,
        svs_embed_backend=args.svs_embed_backend,
    )
    val_ds = PTEmbeddingDirDataset(
        items=val_items,
        max_patches_per_wsi=args.max_patches_per_wsi,
        sample_mode="head",
        sample_seed=args.seed,
        svs_level_mode=args.svs_level_mode,
        svs_embed_backend=args.svs_embed_backend,
    )

    # === Imbalance: WeightedRandomSampler + logit adjustment ===
    train_labels = [int(item[2]) for item in train_items]
    class_counts = np.bincount(train_labels, minlength=len(label_map)).astype(
        np.float32
    )
    class_counts[class_counts == 0.0] = 1.0
    class_weights_arr = (len(train_labels) / (len(label_map) * class_counts)).astype(
        np.float32
    )
    sample_weights = torch.tensor(
        [class_weights_arr[int(item[2])] for item in train_items], dtype=torch.float32
    )
    train_sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # ---------------- Data & loaders ----------------
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=collate_samples
    )

    print("[EvoPS] ================================================")
    print(f"[EvoPS] device={device} seed={args.seed} out_dir={args.out_dir}")
    print(
        f"[EvoPS] input_format={args.input_format} "
        f"svs_level_mode={args.svs_level_mode} svs_embed_backend={args.svs_embed_backend}"
    )
    print(
        f"[EvoPS] train_items={len(train_ds)} val_items={len(val_ds)} "
        f"num_classes={len(label_map)}"
    )
    print(f"[EvoPS] label_map={label_map}")
    print("[EvoPS] ================================================")

    # ---------------- Model ----------------
    model = HAFEDClassifier(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(label_map),
        num_heads=args.num_heads,
    ).to(device)
    model.set_logit_adjustment(
        torch.tensor(class_counts, device=device), tau=args.logit_adjust_tau
    )
    print(
        f"[EvoPS] logit_adjust_tau={args.logit_adjust_tau} class_counts={class_counts.tolist()}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_model_path = os.path.join(args.out_dir, "best_hafed.pt")

    # ---------------- Training loop ----------------
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        seen = 0
        total_samples = max(1, len(train_ds))
        log_interval = max(1, total_samples // 5)

        print(
            f"[EvoPS][HAFED][E{epoch:03d}] ENTER "
            f"samples={total_samples} lr={args.lr:.2e}"
        )

        for batch in train_loader:
            for sample in batch:
                emb = sample.embeddings.to(device)
                mask = torch.ones(1, emb.shape[0], dtype=torch.bool, device=device)
                target = torch.tensor(
                    [sample.label_idx], dtype=torch.long, device=device
                )

                logits, _, _ = model(emb.unsqueeze(0), mask)
                loss = F.cross_entropy(logits, target)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                running_loss += float(loss.item())
                seen += 1

                if seen == 1 or seen % log_interval == 0 or seen == total_samples:
                    avg_loss = running_loss / max(1, seen)
                    print(
                        f"[EvoPS][HAFED][E{epoch:03d}] progress "
                        f"{seen}/{total_samples} avg_loss={avg_loss:.4f}"
                    )

        val_metrics = evaluate_hafed(model, val_loader, device=device)
        train_loss = running_loss / max(1, seen)
        improved = val_metrics["acc"] > best_val_acc
        epoch_secs = time.perf_counter() - epoch_start
        print(
            f"[EvoPS][HAFED] epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        )
        print(
            f"[EvoPS][HAFED][E{epoch:03d}] EXIT "
            f"improved={improved} best_val_acc={max(best_val_acc, val_metrics['acc']):.4f} "
            f"time={epoch_secs:.2f}s"
        )

        if improved:
            best_val_acc = val_metrics["acc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_map": label_map,
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_heads": args.num_heads,
                    "num_classes": len(label_map),
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                best_model_path,
            )
            print(f"[EvoPS][HAFED] saved best checkpoint: {best_model_path}")

    return {
        "label_map": label_map,
        "best_hafed_path": best_model_path,
        "best_val_acc": best_val_acc,
        "out_dir": args.out_dir,
    }


def main() -> None:
    args = parse_args()
    train_evops(args)


if __name__ == "__main__":
    main()
