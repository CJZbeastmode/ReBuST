"""End-to-end EvoPS pipeline for SVS/PT directories.

Workflow
--------
1) Train HAFED-like classifier on train/val.
2) Run EvoPS selector + classifier on test split.
3) Save per-case classification outputs and aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.ablation_patch_selector.EvoPS.evo import evo_select_subset
from src.ablation_patch_selector.EvoPS.train_pipeline import train_evops
from src.ablation_patch_selector.SASHA.data import (
    PTEmbeddingDirDataset,
    build_items_with_label_map,
)
from src.ablation_patch_selector.SASHA.models import HAFEDClassifier


# Safely coerce to float, returning nan on failure
def _safe_float(v: float) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


# Load HAFED checkpoint and build model
def _load_hafed(
    checkpoint_path: str, device: torch.device
) -> Tuple[HAFEDClassifier, Dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = HAFEDClassifier(
        embed_dim=int(ckpt.get("embed_dim", 512)),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        num_classes=int(ckpt.get("num_classes", 2)),
        num_heads=int(ckpt.get("num_heads", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# Compute aggregate classification metrics
def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    metrics = {
        "num_cases": int(len(y_true)),
        "accuracy": _safe_float((y_true == y_pred).mean() if len(y_true) > 0 else 0.0),
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
        "weighted_f1": float("nan"),
        "auc_ovr_weighted": float("nan"),
    }

    if len(y_true) == 0:
        return metrics

    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

        metrics["balanced_accuracy"] = _safe_float(
            balanced_accuracy_score(y_true, y_pred)
        )
        metrics["macro_f1"] = _safe_float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["weighted_f1"] = _safe_float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        num_classes = (
            int(y_prob.shape[1]) if y_prob.ndim == 2 and y_prob.size > 0 else 0
        )
        if num_classes == 2:
            metrics["auc_ovr_weighted"] = _safe_float(
                roc_auc_score(y_true, y_prob[:, 1])
            )
        elif num_classes > 2:
            metrics["auc_ovr_weighted"] = _safe_float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            )
    except Exception:
        pass

    return metrics


# Map end-to-end args to train_evops args
def _build_train_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        train_embeddings_dir=args.train_dir,
        val_embeddings_dir=args.val_dir,
        out_dir=args.out_dir,
        input_format=args.input_format,
        svs_level_mode=args.svs_level_mode,
        svs_embed_backend=args.svs_embed_backend,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        max_patches_per_wsi=args.max_patches_per_wsi,
        patch_sample_mode=args.patch_sample_mode,
        epochs=args.hafed_epochs,
        lr=args.hafed_lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )


def run_end_to_end(args: argparse.Namespace) -> Dict[str, object]:
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Train ----------------
    train_args = _build_train_args(args)
    train_result = train_evops(train_args)

    label_map = dict(train_result["label_map"])
    int_to_label = {int(v): str(k) for k, v in label_map.items()}

    hafed, _ = _load_hafed(str(train_result["best_hafed_path"]), device=device)

    # ---------------- Test ----------------
    test_items = build_items_with_label_map(
        args.test_dir,
        label_map=label_map,
        strict=False,
        input_format=args.input_format,
    )

    test_ds = PTEmbeddingDirDataset(
        items=test_items,
        max_patches_per_wsi=args.max_patches_per_wsi,
        sample_mode="head",
        sample_seed=args.seed,
        svs_level_mode=args.svs_level_mode,
        svs_embed_backend=args.svs_embed_backend,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, collate_fn=lambda x: x
    )

    predictions: List[Dict] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    kept_ratios: List[float] = []

    for batch in test_loader:
        for sample in batch:
            emb = sample.embeddings.detach().cpu().float()
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)

            scores = emb.norm(dim=-1)
            selected, pred_idx, best_score = evo_select_subset(
                candidate_embeddings=emb,
                hafed=hafed,
                device=device,
                budget=args.selection_budget,
                population_size=args.population_size,
                generations=args.generations,
                elite_fraction=args.elite_fraction,
                mutation_rate=args.mutation_rate,
                crossover_rate=args.crossover_rate,
                seed=args.seed,
                scores=scores,
                target_idx=None,
            )

            if not selected:
                selected = [int(torch.argmax(scores).item())]

            subset = emb[torch.tensor(selected, dtype=torch.long)]
            with torch.no_grad():
                mask = torch.ones(1, subset.shape[0], dtype=torch.bool, device=device)
                logits, _, _ = hafed(subset.unsqueeze(0).to(device), mask)
                probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()

            pred_idx = int(np.argmax(probs))
            true_idx = int(sample.label_idx)

            y_true.append(true_idx)
            y_pred.append(pred_idx)
            y_prob.append(probs)
            kept_ratios.append(float(len(selected) / max(1, emb.shape[0])))

            predictions.append(
                {
                    "case_id": sample.case_id,
                    "true_label": str(sample.label_str),
                    "true_index": true_idx,
                    "pred_label": int_to_label.get(pred_idx, str(pred_idx)),
                    "pred_index": pred_idx,
                    "probs": [float(x) for x in probs.tolist()],
                    "source_patch_count": int(emb.shape[0]),
                    "selected_patch_count": int(len(selected)),
                    "selected_ratio": float(len(selected) / max(1, emb.shape[0])),
                    "selection_score": float(best_score),
                    "source_path": str(sample.source_pt_path),
                }
            )

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    y_prob_np = (
        np.vstack(y_prob) if y_prob else np.zeros((0, len(label_map)), dtype=np.float32)
    )

    metrics = _compute_metrics(y_true_np, y_pred_np, y_prob_np)
    metrics["mean_selected_ratio"] = _safe_float(
        np.mean(kept_ratios) if kept_ratios else 0.0
    )
    metrics["label_map"] = label_map

    # ---------------- Save & report ----------------
    predictions_path = os.path.join(args.out_dir, "evops_test_predictions.json")
    metrics_path = os.path.join(args.out_dir, "evops_test_metrics.json")

    with open(predictions_path, "w", encoding="utf-8") as fh:
        json.dump({"predictions": predictions}, fh, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"[SAVED] predictions={predictions_path}")
    print(f"[SAVED] metrics={metrics_path}")
    print(
        "[TEST] "
        f"n={metrics['num_cases']} "
        f"acc={metrics['accuracy']:.4f} "
        f"balanced_acc={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f} "
        f"auc={metrics['auc_ovr_weighted']:.4f}"
    )

    return {"predictions": predictions, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EvoPS end-to-end pipeline")

    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--out-dir", default="data/models/ablation/evops")
    parser.add_argument(
        "--input-format",
        type=str,
        default="auto",
        choices=["pt", "svs", "auto"],
        help="Input file type expected in train/val/test directories.",
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

    parser.add_argument("--hafed-epochs", type=int, default=10)
    parser.add_argument("--hafed-lr", type=float, default=1e-4)

    parser.add_argument("--selection-budget", type=int, default=32)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--crossover-rate", type=float, default=0.7)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_end_to_end(args)


if __name__ == "__main__":
    main()
