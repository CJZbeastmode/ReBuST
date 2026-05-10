"""EvoPS inference pipeline.

Loads trained HAFED and uses evolutionary selection to produce classification outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.ablation_patch_selector.EvoPS.evo import evo_select_subset
from src.ablation_patch_selector.SASHA.data import read_label_from_path
from src.ablation_patch_selector.SASHA.models import HAFEDClassifier


def _safe_float(value: float) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels, preds):
        if 0 <= label < num_classes and 0 <= pred < num_classes:
            mat[label, pred] += 1
    return mat


def _classification_metrics(
    labels: np.ndarray, preds: np.ndarray, probs: np.ndarray, num_classes: int
) -> Dict[str, object]:
    if labels.size == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
            "per_class_f1": [float("nan")] * num_classes,
            "per_class_accuracy_summary": [],
            "support": [0] * num_classes,
            "confusion_matrix": np.zeros((num_classes, num_classes), dtype=np.int64).tolist(),
        }

    conf = _confusion_matrix(labels, preds, num_classes)
    support = conf.sum(axis=1)
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(tp),
            where=(precision + recall) > 0,
        )
    accuracy = float(np.mean(labels == preds))
    balanced_accuracy = float(np.mean(recall)) if num_classes > 0 else float("nan")
    macro_f1 = float(np.mean(f1)) if num_classes > 0 else float("nan")
    weighted_f1 = (
        float(np.sum(f1 * support) / np.sum(support))
        if np.sum(support) > 0
        else float("nan")
    )
    per_class_accuracy_summary = []
    for class_idx in range(num_classes):
        cls_support = int(support[class_idx])
        cls_correct = int(conf[class_idx, class_idx])
        cls_acc = float(cls_correct / cls_support) if cls_support > 0 else float("nan")
        per_class_accuracy_summary.append(
            {
                "class_index": class_idx,
                "support": cls_support,
                "correct": cls_correct,
                "accuracy": cls_acc,
            }
        )

    auc = float("nan")
    if labels.size > 0 and probs.size > 0 and probs.shape[0] == labels.shape[0]:
        try:
            from sklearn.metrics import roc_auc_score

            if num_classes == 2 and probs.shape[1] >= 2:
                auc = float(roc_auc_score(labels, probs[:, 1]))
            elif num_classes > 2 and probs.shape[1] == num_classes:
                auc = float(
                    roc_auc_score(
                        labels,
                        probs,
                        multi_class="ovr",
                        average="weighted",
                    )
                )
        except Exception:
            auc = float("nan")

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "f1": weighted_f1,
        "auc": auc,
        "per_class_f1": f1.tolist(),
        "per_class_accuracy_summary": per_class_accuracy_summary,
        "support": support.tolist(),
        "confusion_matrix": conf.tolist(),
    }


def _bootstrap_metric_std(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, float]:
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "f1", "auc"]
    if labels.size == 0 or bootstrap_samples <= 1:
        return {f"{k}_std": float("nan") for k in keys}

    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    idx = np.arange(n)
    collected = {k: [] for k in keys}

    for _ in range(bootstrap_samples):
        sample_idx = rng.choice(idx, size=n, replace=True)
        sample_metrics = _classification_metrics(
            labels[sample_idx],
            preds[sample_idx],
            probs[sample_idx],
            num_classes,
        )
        for k in keys:
            v = sample_metrics.get(k, float("nan"))
            try:
                collected[k].append(float(v))
            except Exception:
                collected[k].append(float("nan"))

    out = {}
    for k in keys:
        vals = np.array(collected[k], dtype=np.float64)
        valid = vals[np.isfinite(vals)]
        out[f"{k}_std"] = (
            float(np.std(valid, ddof=1)) if valid.size > 1 else float("nan")
        )
    return out


def list_pt_files(input_dir: str) -> List[Path]:
    base = Path(input_dir)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = [
        p
        for p in sorted(base.iterdir())
        if p.is_file() and p.suffix.lower() == ".pt" and not p.name.startswith(".") and not p.name.startswith("._")
    ]
    if not files:
        raise FileNotFoundError(f"No .pt files found in {input_dir}")
    return files


def load_hafed(hafed_path: str, device: torch.device) -> Tuple[HAFEDClassifier, Dict]:
    ckpt = torch.load(hafed_path, map_location=device)
    model = HAFEDClassifier(
        embed_dim=int(ckpt.get("embed_dim", 512)),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        num_classes=int(ckpt.get("num_classes", 2)),
        num_heads=int(ckpt.get("num_heads", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _load_embeddings(pt_path: str) -> torch.Tensor:
    loaded = torch.load(pt_path, map_location="cpu")
    if isinstance(loaded, dict):
        embeddings = loaded.get("embeddings")
    else:
        embeddings = loaded

    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().float()
    else:
        embeddings = torch.tensor(embeddings, dtype=torch.float32)

    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)

    if embeddings.numel() == 0:
        embeddings = torch.zeros(1, 512, dtype=torch.float32)

    return embeddings


def predict_one(
    hafed: HAFEDClassifier,
    embeddings: torch.Tensor,
    device: torch.device,
    selection_budget: int,
    population_size: int,
    generations: int,
    elite_fraction: float,
    mutation_rate: float,
    crossover_rate: float,
    seed: int,
) -> Dict:
    scores = embeddings.norm(dim=-1)
    selected, _, best_score = evo_select_subset(
        candidate_embeddings=embeddings,
        hafed=hafed,
        device=device,
        budget=selection_budget,
        population_size=population_size,
        generations=generations,
        elite_fraction=elite_fraction,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        seed=seed,
        scores=scores,
        target_idx=None,
    )

    if not selected:
        selected = [int(torch.argmax(scores).item())]

    subset = embeddings[torch.tensor(selected, dtype=torch.long)]
    with torch.no_grad():
        mask = torch.ones(1, subset.shape[0], dtype=torch.bool, device=device)
        logits, _, _ = hafed(subset.unsqueeze(0).to(device), mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        pred = int(np.argmax(probs))

    return {
        "pred": pred,
        "probs": probs.tolist(),
        "selected": [int(x) for x in selected],
        "selection_score": float(best_score),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EvoPS inference")
    parser.add_argument("--input-dir", default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/full/test", help="Directory with per-WSI .pt files containing test embeddings")
    parser.add_argument("--hafed-checkpoint", default="data/models/ablation_patch_selector/evops/best_hafed.pt")
    parser.add_argument("--out-json", default="data/benchmark/evops.json")
    parser.add_argument("--output", dest="output", action="store_true", default=True)
    parser.add_argument("--no-output", dest="output", action="store_false")
    parser.add_argument("--selection-budget", type=int, default=32)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--crossover-rate", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap resamples used to estimate metric standard deviations",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed for bootstrap standard-deviation estimation",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    hafed, ckpt = load_hafed(args.hafed_checkpoint, device=device)

    label_map = ckpt.get("label_map", {})
    int_to_label = {int(v): str(k) for k, v in label_map.items()}

    predictions = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    skipped_missing_label = 0
    for pt_path in list_pt_files(args.input_dir):
        case_id = pt_path.stem
        embeddings = _load_embeddings(str(pt_path))
        out = predict_one(
            hafed,
            embeddings,
            device=device,
            selection_budget=args.selection_budget,
            population_size=args.population_size,
            generations=args.generations,
            elite_fraction=args.elite_fraction,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            seed=args.seed,
        )

        true_label = None
        try:
            true_label = read_label_from_path(str(pt_path))
        except Exception:
            true_label = None

        if true_label is not None and true_label in label_map:
            true_idx = int(label_map[true_label])
            y_true.append(true_idx)
            y_pred.append(int(out["pred"]))
            y_prob.append(np.asarray(out["probs"], dtype=np.float32))
        else:
            skipped_missing_label += 1

        predictions.append(
            {
                "case_id": case_id,
                "pred_index": out["pred"],
                "pred_label": int_to_label.get(out["pred"], str(out["pred"])),
                "probs": out["probs"],
                "label": true_label,
            }
        )

    labels_np = np.asarray(y_true, dtype=np.int64)
    preds_np = np.asarray(y_pred, dtype=np.int64)
    probs_np = (
        np.asarray(y_prob, dtype=np.float32)
        if len(y_prob) > 0
        else np.zeros((0, len(label_map)), dtype=np.float32)
    )
    metrics = _classification_metrics(labels_np, preds_np, probs_np, len(label_map))
    metrics.update(
        _bootstrap_metric_std(
            labels=labels_np,
            preds=preds_np,
            probs=probs_np,
            num_classes=len(label_map),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
    )
    for row in metrics.get("per_class_accuracy_summary", []):
        class_index = int(row.get("class_index", -1))
        row["class_label"] = int_to_label.get(class_index, str(class_index))

    print(
        f"[DATA] total={len(predictions)} labeled={labels_np.size} skipped={skipped_missing_label}"
    )
    print(
        "[METRICS] "
        f"acc={metrics['accuracy']:.4f}±{metrics['accuracy_std']:.4f} "
        f"bacc={metrics['balanced_accuracy']:.4f}±{metrics['balanced_accuracy_std']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}±{metrics['macro_f1_std']:.4f} "
        f"f1={metrics['f1']:.4f}±{metrics['f1_std']:.4f} "
        f"auc={metrics['auc']:.4f}±{metrics['auc_std']:.4f}"
    )
    print(
        f"[STD] bootstrap_samples={args.bootstrap_samples} "
        f"bootstrap_seed={args.bootstrap_seed}"
    )
    print("[PER-CLASS-ACC]", metrics.get("per_class_accuracy_summary"))
    print("[CONFUSION]", metrics.get("confusion_matrix"))

    if args.output and args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "label_map": label_map,
                    "metrics": metrics,
                    "predictions": predictions,
                },
                fh,
                indent=2,
            )
        print(f"[SAVED] {args.out_json}")


if __name__ == "__main__":
    main()
