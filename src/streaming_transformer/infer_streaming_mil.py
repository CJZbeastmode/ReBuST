"""Inference for streaming MIL transformer v1."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.streaming_transformer.model import StreamingMILTransformer


def _default_checkpoint_path() -> str:
    return os.path.join(
        repo_root,
        "data",
        "models",
        "downstream_tasks",
        "streaming_mil_v1_3_2",
        "best_streaming_mil.pt",
    )


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


def _load_embeddings_and_coords(
    pt_path: str,
) -> Tuple[torch.Tensor, torch.Tensor, str | None]:
    loaded = torch.load(pt_path, map_location="cpu")

    label = None
    if isinstance(loaded, dict):
        embeddings = _to_tensor(loaded.get("embeddings"), dim=2)
        coords = _to_tensor(loaded.get("coords"), dim=2)
        if loaded.get("label") is not None:
            label = str(loaded.get("label"))
    elif isinstance(loaded, torch.Tensor):
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
    return embeddings, coords, label


def load_model(
    checkpoint_path: str, device: torch.device
) -> Tuple[StreamingMILTransformer, Dict[str, int]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    label_map = ckpt["label_map"]

    state_dict = ckpt["model_state_dict"]
    remapped_state = {}
    for key, value in state_dict.items():
        if key.startswith("local_layers."):
            new_key = "local_encoder.layers." + key[len("local_layers."):]
            remapped_state[new_key] = value
        else:
            remapped_state[key] = value

    model = StreamingMILTransformer(
        embed_dim=cfg["embed_dim"],
        model_dim=cfg["model_dim"],
        num_classes=len(label_map),
        patch_chunk_size=cfg["patch_batch_size"],
        local_num_heads=cfg["local_heads"],
        local_num_layers=cfg["local_layers"],
        local_dropout=cfg["local_dropout"],
    ).to(device)
    model.load_state_dict(remapped_state, strict=False)
    model.eval()
    return model, label_map


def predict_one(
    model: StreamingMILTransformer,
    embeddings: torch.Tensor,
    coords: torch.Tensor,
    device: torch.device,
) -> Dict:
    with torch.no_grad():
        logits, _ = model([embeddings.to(device)], [coords.to(device)])
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        pred = int(np.argmax(probs))
    return {"pred": pred, "probs": probs.tolist()}


def _confusion_matrix(
    labels: np.ndarray, preds: np.ndarray, num_classes: int
) -> np.ndarray:
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


def main(args):
    device = torch.device(args.device)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            "Checkpoint not found. Provide --checkpoint or train a model first. "
            f"Missing: {args.checkpoint}"
        )
    model, label_map = load_model(args.checkpoint, device)
    int_to_label = {v: k for k, v in label_map.items()}

    if not args.embeddings_dir:
        raise ValueError("Provide --embeddings-dir")

    predictions = []
    true_labels = []
    pred_labels = []
    pred_probs = []
    skipped_missing_label = 0

    pt_files = sorted([f for f in os.listdir(args.embeddings_dir) if f.endswith(".pt")])
    for fname in pt_files:
        if fname.startswith(".") or fname.startswith("._"):
            continue
        case_id = os.path.splitext(fname)[0]
        pt_path = os.path.join(args.embeddings_dir, fname)
        embs, coords, label_name = _load_embeddings_and_coords(pt_path)
        out = predict_one(model, embs, coords, device)
        predictions.append(
            {
                "case_id": case_id,
                "pred_index": out["pred"],
                "pred_label": int_to_label[out["pred"]],
                "probs": out["probs"],
                "label": label_name,
            }
        )
        if label_name is None or label_name not in label_map:
            skipped_missing_label += 1
            continue
        true_labels.append(int(label_map[label_name]))
        pred_labels.append(int(out["pred"]))
        pred_probs.append(np.asarray(out["probs"], dtype=np.float32))

    labels_np = np.array(true_labels, dtype=np.int64)
    preds_np = np.array(pred_labels, dtype=np.int64)
    probs_np = (
        np.array(pred_probs, dtype=np.float32)
        if len(pred_probs) > 0
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

    print(f"[DATA] total={len(predictions)} labeled={labels_np.size} skipped={skipped_missing_label}")
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

    out_path = args.out_json
    if args.output and out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(
                {
                    "label_map": label_map,
                    "metrics": metrics,
                    "predictions": predictions,
                },
                fh,
                indent=2,
            )
        print(f"[SAVED] {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for streaming MIL transformer"
    )
    parser.add_argument(
        "--checkpoint",
        default=_default_checkpoint_path(),
        help="Path to model checkpoint (defaults to trained v1.3.2 best model)",
    )
    parser.add_argument(
        "--embeddings-dir",
        default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/full/test",
        help="Directory of WSI embedding .pt files",
    )
    parser.add_argument(
        "--out-json", default="data/benchmark/streaming_transformer.json", help="Optional output predictions json"
    )
    parser.add_argument("--output", dest="output", action="store_true", default=True)
    parser.add_argument("--no-output", dest="output", action="store_false")
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
    args = parse_args()
    main(args)
