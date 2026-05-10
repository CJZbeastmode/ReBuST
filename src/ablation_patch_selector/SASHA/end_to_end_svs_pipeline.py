"""End-to-end SASHA pipeline for SVS directories.

Workflow
--------
1) Train SASHA (HAFED + selector) on train/val splits.
2) Run deterministic selector + classifier on test split.
3) Save per-case classification outputs and aggregate metrics.

Expected data layout
--------------------
/Volumes/Xbox_HD/Data/med_img/
    train/*.svs
    val/*.svs
    test/*.svs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.ablation_patch_selector.SASHA.data import (
    PTEmbeddingDirDataset,
    build_items_with_label_map,
)
from src.ablation_patch_selector.SASHA.models import HAFEDClassifier, SashaPolicyValue
from src.ablation_patch_selector.SASHA.train_pipeline import (
    pick_candidate_pool,
    run_selector_episode,
    train_sasha,
)


def _safe_float(v: float) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _load_hafed(checkpoint_path: str, device: torch.device) -> tuple[HAFEDClassifier, Dict]:
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


def _load_selector(checkpoint_path: str, device: torch.device) -> tuple[SashaPolicyValue, Dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SashaPolicyValue(
        embed_dim=int(ckpt.get("embed_dim", 512)),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
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

        num_classes = int(y_prob.shape[1]) if y_prob.ndim == 2 and y_prob.size > 0 else 0
        if num_classes == 2:
            metrics["auc_ovr_weighted"] = _safe_float(roc_auc_score(y_true, y_prob[:, 1]))
        elif num_classes > 2:
            metrics["auc_ovr_weighted"] = _safe_float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            )
    except Exception:
        pass

    return metrics


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
        hafed_epochs=args.hafed_epochs,
        hafed_lr=args.hafed_lr,
        selector_epochs=args.selector_epochs,
        selector_lr=args.selector_lr,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        gamma=args.gamma,
        max_candidates=args.max_candidates,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )


def run_end_to_end(args: argparse.Namespace) -> Dict[str, object]:
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_args = _build_train_args(args)
    train_result = train_sasha(train_args)

    label_map = dict(train_result["label_map"])
    int_to_label = {int(v): str(k) for k, v in label_map.items()}

    hafed, _ = _load_hafed(str(train_result["best_hafed_path"]), device=device)
    selector, _ = _load_selector(str(train_result["best_selector_path"]), device=device)

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
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=lambda x: x)

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

            candidates, _, _ = pick_candidate_pool(
                embeddings=emb,
                hafed=hafed,
                max_candidates=args.max_candidates,
                device=device,
            )

            if candidates.shape[0] <= 0:
                continue

            rollout, final_prob, kept = run_selector_episode(
                sample=sample,
                candidate_embeddings=candidates,
                policy=selector,
                hafed_frozen=hafed,
                device=device,
                max_steps=args.max_steps,
                gamma=1.0,
                deterministic=True,
            )

            if kept <= 0:
                kept = 1
                top_idx = int(torch.argmax(candidates.norm(dim=-1)).item())
                rollout.selected_indices = [top_idx]

            chosen_idx = torch.tensor(rollout.selected_indices, dtype=torch.long)
            subset = candidates[chosen_idx]

            with torch.no_grad():
                mask = torch.ones(1, subset.shape[0], dtype=torch.bool, device=device)
                logits, _, _ = hafed(subset.unsqueeze(0).to(device), mask)
                probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()

            pred_idx = int(np.argmax(probs))
            true_idx = int(sample.label_idx)

            y_true.append(true_idx)
            y_pred.append(pred_idx)
            y_prob.append(probs)
            kept_ratios.append(float(kept / max(1, emb.shape[0])))

            predictions.append(
                {
                    "case_id": sample.case_id,
                    "true_label": str(sample.label_str),
                    "true_index": true_idx,
                    "pred_label": int_to_label.get(pred_idx, str(pred_idx)),
                    "pred_index": pred_idx,
                    "probs": [float(x) for x in probs.tolist()],
                    "source_patch_count": int(emb.shape[0]),
                    "selected_patch_count": int(kept),
                    "selected_ratio": float(kept / max(1, emb.shape[0])),
                    "selector_final_prob": float(final_prob),
                    "source_path": str(sample.source_pt_path),
                }
            )

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    y_prob_np = np.vstack(y_prob) if y_prob else np.zeros((0, len(label_map)), dtype=np.float32)

    metrics = _compute_metrics(y_true_np, y_pred_np, y_prob_np)
    metrics["mean_selected_ratio"] = _safe_float(np.mean(kept_ratios) if kept_ratios else 0.0)
    metrics["label_map"] = label_map

    predictions_path = os.path.join(args.out_dir, "sasha_test_predictions.json")
    metrics_path = os.path.join(args.out_dir, "sasha_test_metrics.json")

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
        f"bacc={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"auc={metrics['auc_ovr_weighted']:.4f} "
        f"mean_selected_ratio={metrics['mean_selected_ratio']:.4f}"
    )

    return {
        "train": train_result,
        "metrics": metrics,
        "predictions_path": predictions_path,
        "metrics_path": metrics_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end SASHA train+test pipeline")

    parser.add_argument("--data-root", default="/Volumes/Xbox_HD/Data/med_img")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--out-dir", default="data/models/ablation/sasha_e2e")

    parser.add_argument("--input-format", type=str, default="svs", choices=["pt", "svs", "auto"])
    parser.add_argument("--svs-level-mode", type=str, default="root_only", choices=["root_only", "finest_only"])
    parser.add_argument("--svs-embed-backend", type=str, default="plip", choices=["plip", "conch"])

    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)

    parser.add_argument("--max-patches-per-wsi", type=int, default=0)
    parser.add_argument("--patch-sample-mode", type=str, default="uniform", choices=["uniform", "head", "random"])

    parser.add_argument("--hafed-epochs", type=int, default=10)
    parser.add_argument("--hafed-lr", type=float, default=1e-4)

    parser.add_argument("--selector-epochs", type=int, default=8)
    parser.add_argument("--selector-lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=24)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    root = os.path.abspath(args.data_root)
    args.train_dir = os.path.abspath(args.train_dir or os.path.join(root, "train"))
    args.val_dir = os.path.abspath(args.val_dir or os.path.join(root, "val"))
    args.test_dir = os.path.abspath(args.test_dir or os.path.join(root, "test"))
    args.out_dir = os.path.abspath(args.out_dir)

    return args


def main() -> None:
    args = parse_args()
    run_end_to_end(args)


if __name__ == "__main__":
    main()
