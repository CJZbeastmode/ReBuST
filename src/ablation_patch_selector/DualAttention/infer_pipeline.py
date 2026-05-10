"""Inference pipeline for Raza dual attention on embedding inputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.ablation_patch_selector.DualAttention.models import (
    RazaHardAttention,
    SoftAttentionEmbedding,
)
from src.ablation_patch_selector.DualAttention.sampling import sample_attention_candidates
from src.ablation_patch_selector.DualAttention.train_pipeline import _predict_greedy
from src.ablation_patch_selector.SASHA_archive.data import PTEmbeddingDirDataset, build_items_with_label_map


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


def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    soft = SoftAttentionEmbedding(
        embed_dim=int(ckpt.get("embed_dim", 512)),
        hidden_dim=int(ckpt.get("soft_hidden_dim", 256)),
    ).to(device)
    hard = RazaHardAttention(
        embed_dim=int(ckpt.get("embed_dim", 512)),
        coord_dim=int(ckpt.get("coord_dim", 2)),
        hidden_dim=int(ckpt.get("hard_hidden_dim", 256)),
        num_classes=int(ckpt.get("num_classes", 2)),
    ).to(device)
    soft.load_state_dict(ckpt["soft_state_dict"])
    hard.load_state_dict(ckpt["hard_state_dict"])
    soft.eval()
    hard.eval()
    return soft, hard, ckpt


def _prepare_coords(coords: torch.Tensor, coord_dim: int) -> torch.Tensor:
    if coords is None or coords.numel() == 0:
        return torch.zeros(0, coord_dim, device=coords.device if coords is not None else None)
    if coords.dim() == 1:
        coords = coords.unsqueeze(0)
    if coords.shape[1] >= coord_dim:
        return coords[:, -coord_dim:]
    pad = torch.zeros(coords.shape[0], coord_dim - coords.shape[1], device=coords.device, dtype=coords.dtype)
    return torch.cat([coords, pad], dim=1)


def run_inference(args: argparse.Namespace) -> Dict[str, object]:
    if args.output and args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_files = len(list_pt_files(args.embeddings_dir))

    soft, hard, ckpt = load_checkpoint(args.checkpoint, device=device)
    label_map = ckpt.get("label_map", {})
    int_to_label = {int(v): str(k) for k, v in label_map.items()}

    items = build_items_with_label_map(
        args.embeddings_dir,
        label_map=label_map,
        strict=False,
        input_format="pt",
    )
    dataset = PTEmbeddingDirDataset(items)

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []
    predictions: List[Dict[str, object]] = []

    with torch.no_grad():
        for sample in dataset:
            embeddings = sample.embeddings.to(device)
            coords = _prepare_coords(sample.coords.to(device), int(ckpt.get("coord_dim", 2)))
            n = int(embeddings.shape[0])
            if n <= 0:
                continue

            mask = torch.ones(1, n, dtype=torch.bool, device=device)
            attn = soft(embeddings.unsqueeze(0), mask).squeeze(0)

            cand_emb, cand_coords, _ = sample_attention_candidates(
                embeddings,
                coords,
                attn,
                num_tiles=int(ckpt.get("num_tiles", args.num_tiles)),
                pool_multiplier=args.pool_multiplier,
                noise_low=0.0,
                noise_high=0.0,
                min_dist=args.min_tile_dist,
            )

            logits, probs, chosen = _predict_greedy(
                hard,
                cand_emb,
                cand_coords,
                num_glimpses=int(ckpt.get("num_glimpses", args.num_glimpses)),
            )

            pred = int(torch.argmax(logits, dim=-1).item())
            y_true.append(int(sample.label_idx))
            y_pred.append(pred)
            y_prob.append(probs.squeeze(0).detach().cpu().tolist())

            predictions.append(
                {
                    "case_id": sample.case_id,
                    "pred_index": pred,
                    "pred_label": int_to_label.get(pred, str(pred)),
                    "probs": probs.squeeze(0).detach().cpu().tolist(),
                    "label": sample.label_str,
                }
            )

    labels_np = np.array(y_true, dtype=np.int64)
    preds_np = np.array(y_pred, dtype=np.int64)
    probs_np = (
        np.array(y_prob, dtype=np.float32)
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

    out_payload = {
        "label_map": label_map,
        "metrics": metrics,
        "predictions": predictions,
    }

    out_path = args.out_json
    if args.output and out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2)

    skipped = total_files - labels_np.size
    print(f"[DATA] total={total_files} labeled={labels_np.size} skipped={skipped}")
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
    if args.output and out_path:
        print(f"[SAVED] {out_path}")

    return {"metrics": metrics, "output_path": out_path, "output_enabled": bool(args.output)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer with Raza dual attention model")
    parser.add_argument(
        "--embeddings-dir",
        default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/full/test",
    )
    parser.add_argument("--checkpoint", default="data/models/ablation_patch_selector/dual_attention/best_dual_attention.pt")
    parser.add_argument("--out-json", default="data/benchmark/dual_attention.json")
    parser.add_argument("--output", dest="output", action="store_true", default=True)
    parser.add_argument("--no-output", dest="output", action="store_false")
    parser.add_argument("--num-tiles", type=int, default=12)
    parser.add_argument("--num-glimpses", type=int, default=6)
    parser.add_argument("--pool-multiplier", type=int, default=2)
    parser.add_argument("--min-tile-dist", type=float, default=0.0)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_inference(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
