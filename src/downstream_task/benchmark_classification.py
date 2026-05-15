"""
Stratified K-fold TransformerMIL benchmark for pre-extracted WSI embeddings.

Given a directory of ``{case_id}.pt`` files (each containing a dict with key
``"embeddings": Tensor[N, 512]``) and a JSON label file, this script:

1. Builds a case list filtered to cases that have a ``.pt`` file.
2. Runs Stratified K-fold cross-validation.
3. In each fold:  trains TransformerMIL for ``--epochs`` epochs on the train
   split and evaluates on the val split.
4. Reports per-fold and aggregated Accuracy / F1 / AUC.
5. Optionally saves the full result dict to ``--out-json``.

Usage
-----
python src/downstream_task/benchmark_classification.py \\
    --embeddings-dir data/extracted_embeddings/humbe_a2c \\
    --labels-json    data/labels_main.json \\
    --k 5 --epochs 20 --device cpu

Programmatic API
----------------
from src.downstream_task.benchmark_classification import run_kfold_benchmark
results = run_kfold_benchmark("data/extracted_embeddings/humbe_a2c",
...                               "data/labels_main.json", k=5, epochs=20)
"""

import sys
import os
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.downstream_task.wsi_classification_plip import TransformerMIL

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for benchmarking. "
        "Install it with: pip install scikit-learn"
    ) from exc


# ============================================================
# Dataset
# ============================================================


class PreextractedWSIDataset(Dataset):
    """Loads pre-extracted ``.pt`` embedding files on demand."""

    def __init__(self, items: List[Dict], embeddings_dir: str):
        self.items = items
        self.embeddings_dir = embeddings_dir

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        case_id = item["case_id"]
        label = item["label"]
        pt_path = os.path.join(self.embeddings_dir, f"{case_id}.pt")

        loaded = torch.load(pt_path, map_location="cpu")
        if isinstance(loaded, dict) and "embeddings" in loaded:
            embeddings = loaded["embeddings"].float()
            patch_count = int(loaded.get("patch_count", embeddings.shape[0]))
        elif isinstance(loaded, torch.Tensor):
            embeddings = loaded.float()
            patch_count = embeddings.shape[0]
        else:
            embeddings = torch.zeros(1, 512)
            patch_count = 0

        # Sanity-guard: ensure 2-D [N, 512]
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        return {
            "embeddings": embeddings,
            "label": label,
            "case_id": case_id,
            "patch_count": patch_count,
        }


def _collate_single(batch):
    """Pass-through collate for batch_size=1 (one WSI per step)."""
    return batch


# ============================================================
# Training / evaluation helpers
# ============================================================


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        item = batch[0]
        X = item["embeddings"].to(device)  # [N, 512]
        y = torch.tensor([item["label"]], device=device)

        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits.unsqueeze(0), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> Dict:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for batch in loader:
        item = batch[0]
        X = item["embeddings"].to(device)
        logits, _ = model(X)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [num_classes]
        pred = int(probs.argmax())

        all_labels.append(item["label"])
        all_preds.append(pred)
        all_probs.append(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.vstack(all_probs)  # [N_val, num_classes]

    acc = float(accuracy_score(all_labels, all_preds))
    f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))

    try:
        if num_classes == 2:
            auc = float(roc_auc_score(all_labels, all_probs[:, 1]))
        else:
            auc = float(
                roc_auc_score(
                    all_labels,
                    all_probs,
                    multi_class="ovr",
                    average="weighted",
                )
            )
    except Exception:
        auc = float("nan")

    return {"accuracy": acc, "f1": f1, "auc": auc}


# ============================================================
# K-fold benchmark (public API)
# ============================================================


def run_kfold_benchmark(
    embeddings_dir: str,
    labels_json: str,
    k: int = 5,
    epochs: int = 20,
    lr: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
) -> Dict:
    """Run stratified K-fold TransformerMIL classification benchmark.

    Parameters
    ----------
    embeddings_dir : str
        Directory containing ``{case_id}.pt`` files.
    labels_json : str
        JSON mapping ``case_id`` → ``project_label`` (string).
    k : int
        Number of folds (default 5).
    epochs : int
        Training epochs per fold (default 20).
    lr : float
        Adam learning rate (default 1e-4).
    device : str
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Summary dict with keys:
        ``accuracy_mean``, ``accuracy_std``, ``f1_mean``, ``f1_std``,
        ``auc_mean``, ``auc_std``, ``avg_patches_per_wsi``,
        ``num_cases``, ``k``, ``epochs``, ``folds`` (list of per-fold dicts).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Build case list — only cases with both a label and a .pt file
    # ------------------------------------------------------------------
    with open(labels_json, "r") as f:
        raw_labels = json.load(f)

    available = [
        cid
        for cid in raw_labels
        if raw_labels[cid] and os.path.exists(os.path.join(embeddings_dir, f"{cid}.pt"))
    ]

    if not available:
        raise FileNotFoundError(
            f"No matching .pt files found in {embeddings_dir}. "
            "Run extract_method_embeddings.py first."
        )

    projects = sorted({raw_labels[c] for c in available})
    proj2int = {p: i for i, p in enumerate(projects)}
    num_classes = len(projects)

    items = [{"case_id": c, "label": proj2int[raw_labels[c]]} for c in available]

    print(
        f"[BENCHMARK] dir={embeddings_dir}  cases={len(items)}  "
        f"classes={num_classes}  k={k}  epochs={epochs}  device={device}"
    )
    print(f"[BENCHMARK] Label map: {proj2int}")

    # ------------------------------------------------------------------
    # Stratified K-fold
    # ------------------------------------------------------------------
    y_strat = np.array([it["label"] for it in items])

    # Guard: if any class has fewer samples than k, reduce k automatically.
    min_class_count = int(np.bincount(y_strat).min())
    if min_class_count < k:
        print(
            f"[WARNING] Smallest class has {min_class_count} samples; "
            f"reducing k from {k} to {min_class_count}."
        )
        k = max(2, min_class_count)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    dev = torch.device(device)
    fold_metrics: List[Dict] = []
    all_patch_counts: List[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(items, y_strat)):
        print(
            f"\n{'='*55}\n"
            f"[FOLD {fold_idx+1}/{k}]  "
            f"train={len(train_idx)}  val={len(val_idx)}\n"
            f"{'='*55}"
        )

        train_items = [items[i] for i in train_idx]
        val_items = [items[i] for i in val_idx]

        train_ds = PreextractedWSIDataset(train_items, embeddings_dir)
        val_ds = PreextractedWSIDataset(val_items, embeddings_dir)

        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, collate_fn=_collate_single
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, collate_fn=_collate_single
        )

        model = TransformerMIL(
            embed_dim=512,
            hidden_dim=256,
            num_classes=num_classes,
            num_heads=8,
            num_layers=2,
            dropout=0.1,
        ).to(dev)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        pbar = tqdm(range(epochs), desc=f"Fold {fold_idx+1}", unit="epoch")
        for epoch in pbar:
            loss = _train_one_epoch(model, train_loader, optimizer, criterion, dev)
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        fold_result = _evaluate(model, val_loader, num_classes, dev)
        fold_result["fold"] = fold_idx + 1

        # Collect patch counts from the validation set
        for it in val_items:
            pt_path = os.path.join(embeddings_dir, f"{it['case_id']}.pt")
            try:
                loaded = torch.load(pt_path, map_location="cpu")
                if isinstance(loaded, dict):
                    cnt = loaded.get(
                        "patch_count",
                        loaded["embeddings"].shape[0],
                    )
                elif isinstance(loaded, torch.Tensor):
                    cnt = loaded.shape[0]
                else:
                    cnt = 0
                all_patch_counts.append(int(cnt))
            except Exception:
                pass

        fold_metrics.append(fold_result)
        print(
            f"  [FOLD {fold_idx+1}] "
            f"acc={fold_result['accuracy']:.4f}  "
            f"f1={fold_result['f1']:.4f}  "
            f"auc={fold_result['auc']:.4f}"
        )

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["f1"] for m in fold_metrics]
    aucs = [m["auc"] for m in fold_metrics if not np.isnan(m["auc"])]

    summary = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "avg_patches_per_wsi": (
            float(np.mean(all_patch_counts)) if all_patch_counts else float("nan")
        ),
        "num_classes": num_classes,
        "num_cases": len(items),
        "k": k,
        "epochs": epochs,
        "label_map": proj2int,
        "folds": fold_metrics,
    }

    print(
        f"\n{'='*55}\n"
        f"  K-FOLD SUMMARY ({k} folds)\n"
        f"{'='*55}\n"
        f"  ACC : {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}\n"
        f"  F1  : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}\n"
        f"  AUC : {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}\n"
        f"  Avg patches/WSI : {summary['avg_patches_per_wsi']:.1f}\n"
        f"{'='*55}"
    )

    return summary


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Stratified K-fold MIL benchmark on pre-extracted embeddings"
    )
    p.add_argument(
        "--embeddings-dir",
        required=True,
        help="Directory with {case_id}.pt embedding files",
    )
    p.add_argument(
        "--labels-json",
        default="data/labels_main.json",
        help="JSON mapping case_id → project label",
    )
    p.add_argument("--k", type=int, default=5, help="Number of CV folds (default 5)")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs per fold")
    p.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate")
    p.add_argument(
        "--device",
        default="cpu",
        help='Torch device ("cpu", "cuda", "mps")',
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-json",
        default=None,
        help="Path to save full results as JSON",
    )
    args = p.parse_args()

    results = run_kfold_benchmark(
        embeddings_dir=args.embeddings_dir,
        labels_json=args.labels_json,
        k=args.k,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Results saved → {args.out_json}")
