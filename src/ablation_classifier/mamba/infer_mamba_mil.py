"""Inference for Mamba-style MIL model."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import torch

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.ablation_classifier.mamba.model import MambaMIL


# Coerce value to float32 tensor of the requested rank
def _to_tensor(value: object, dim: int) -> torch.Tensor | None:
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=torch.float32)
    else:
        value = value.detach().cpu().float()
    if value.dim() == dim - 1:
        value = value.unsqueeze(0)
    if value.dim() != dim:
        return None
    if value.numel() == 0:
        return None
    return value


# Uniform subsample indices when patch count exceeds cap
def _sample_indices(n: int, max_patches_per_wsi: int) -> np.ndarray:
    if max_patches_per_wsi <= 0 or n <= max_patches_per_wsi:
        return np.arange(n, dtype=np.int64)
    idx = np.linspace(0, n - 1, num=max_patches_per_wsi)
    return np.unique(idx.round().astype(np.int64))


# Load embeddings and coords from a .pt file, applying patch cap
def _load_embeddings_and_coords(
    pt_path: str, max_patches_per_wsi: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    loaded = torch.load(pt_path, map_location="cpu")

    label = None

    if isinstance(loaded, dict):
        embeddings = _to_tensor(loaded.get("embeddings"), dim=2)
        coords = _to_tensor(loaded.get("coords"), dim=2)
        if loaded.get("label") is not None:
            label = str(loaded.get("label"))
    elif torch.is_tensor(loaded):
        embeddings = _to_tensor(loaded, dim=2)
        coords = None
    else:
        embeddings = None
        coords = None

    if embeddings is None:
        embeddings = torch.zeros(1, 512, dtype=torch.float32)
    if coords is None:
        coords = torch.zeros(embeddings.shape[0], 3, dtype=torch.float32)

    if coords.shape[0] != embeddings.shape[0]:
        coords = torch.zeros(embeddings.shape[0], 3, dtype=torch.float32)

    idx = _sample_indices(int(embeddings.shape[0]), max_patches_per_wsi)
    embeddings = embeddings[idx]
    coords = coords[idx]

    return embeddings, coords, label


# Compute per-class confusion matrix
def _confusion_matrix(
    labels: np.ndarray, preds: np.ndarray, num_classes: int
) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels, preds):
        if 0 <= label < num_classes and 0 <= pred < num_classes:
            mat[label, pred] += 1
    return mat


# Compute classification metrics from labels, predictions, and probabilities
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
            "confusion_matrix": np.zeros(
                (num_classes, num_classes), dtype=np.int64
            ).tolist(),
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
                        average="macro",
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


# Bootstrap resampling to estimate metric standard deviations
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


# Load checkpoint and build MambaMIL model
def load_model(
    checkpoint_path: str, device: torch.device
) -> Tuple[MambaMIL, Dict[str, int], Dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    label_map = ckpt["label_map"]

    model = MambaMIL(
        embed_dim=cfg["embed_dim"],
        model_dim=cfg["model_dim"],
        num_classes=len(label_map),
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        use_coords=bool(cfg.get("use_coords", True)),
        attn_dim=cfg.get("attn_dim", 128),
        expand_factor=cfg.get("expand_factor", 2),
        conv_kernel_size=cfg.get("conv_kernel_size", 3),
        d_state=cfg.get("d_state", 16),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, label_map, cfg


# Prediction for a single WSI given embeddings and coords
@torch.no_grad()
def predict_one(
    model: MambaMIL,
    embeddings: torch.Tensor,
    coords: torch.Tensor,
    device: torch.device,
) -> Dict:
    logits, extras = model([embeddings.to(device)], [coords.to(device)])
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))

    attention = extras.get("attention")
    if attention is not None:
        attention_values = attention[0].detach().cpu().numpy().tolist()
    else:
        attention_values = []

    return {"pred": pred, "probs": probs.tolist(), "attention": attention_values}


def main(args):
    # Set up
    device = torch.device(args.device)
    model, label_map, cfg = load_model(args.checkpoint, device)
    int_to_label = {v: k for k, v in label_map.items()}

    max_patches = int(args.max_patches_per_wsi)
    if max_patches <= 0:
        max_patches = int(cfg.get("max_patches_per_wsi", 0))

    predictions = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    skipped_missing_label = 0

    # Inference loop
    if args.input_pt:
        case_id = Path(args.input_pt).stem
        embeddings, coords, label_name = _load_embeddings_and_coords(
            args.input_pt, max_patches
        )
        out = predict_one(model, embeddings, coords, device)
        predictions.append(
            {
                "case_id": case_id,
                "pred_index": out["pred"],
                "pred_label": int_to_label[out["pred"]],
                "probs": out["probs"],
                "attention": out["attention"],
                "label": label_name,
            }
        )
        if label_name is not None and label_name in label_map:
            y_true.append(int(label_map[label_name]))
            y_pred.append(int(out["pred"]))
            y_prob.append(np.asarray(out["probs"], dtype=np.float32))
        else:
            skipped_missing_label += 1
    else:
        if not args.embeddings_dir:
            raise ValueError("Provide --input-pt or --embeddings-dir")
        pt_files = sorted(
            [f for f in os.listdir(args.embeddings_dir) if f.endswith(".pt")]
        )
        for fname in pt_files:
            if fname.startswith(".") or fname.startswith("._"):
                continue
            case_id = os.path.splitext(fname)[0]
            pt_path = os.path.join(args.embeddings_dir, fname)
            embeddings, coords, label_name = _load_embeddings_and_coords(
                pt_path, max_patches
            )
            out = predict_one(model, embeddings, coords, device)
            predictions.append(
                {
                    "case_id": case_id,
                    "pred_index": out["pred"],
                    "pred_label": int_to_label[out["pred"]],
                    "probs": out["probs"],
                    "attention": out["attention"],
                    "label": label_name,
                }
            )
            if label_name is not None and label_name in label_map:
                y_true.append(int(label_map[label_name]))
                y_pred.append(int(out["pred"]))
                y_prob.append(np.asarray(out["probs"], dtype=np.float32))
            else:
                skipped_missing_label += 1

    # Compute metrics
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
        with open(args.out_json, "w") as fh:
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
    else:
        for row in predictions:
            print(f"{row['case_id']} -> {row['pred_label']} ({row['pred_index']})")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Mamba MIL model")
    parser.add_argument(
        "--checkpoint",
        default="data/models/ablation_classifier/mamba/best_mamba_mil.pt",
    )
    parser.add_argument("--input-pt", default=None)
    parser.add_argument(
        "--embeddings-dir",
        default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/full/test",
    )
    parser.add_argument("--out-json", default="data/benchmark/mamba_mil.json")
    parser.add_argument("--output", dest="output", action="store_true", default=True)
    parser.add_argument("--no-output", dest="output", action="store_false")
    parser.add_argument("--max-patches-per-wsi", type=int, default=0)
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


if __name__ == "__main__":
    main(parse_args())
