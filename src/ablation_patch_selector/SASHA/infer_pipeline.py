"""SASHA inference pipeline.

Loads trained HAFED + selector policy and exports selected patch embeddings in the
same PT schema used by extracted_with_embeddings/a2c/*.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.ablation_patch_selector.SASHA.data import read_label_from_path
from src.ablation_patch_selector.SASHA.models import HAFEDClassifier, SashaPolicyValue


# Safely coerce to float, returning nan on failure
def _safe_float(value: float) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


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


# List .pt files in a directory
def list_pt_files(input_dir: str) -> List[Path]:
    base = Path(input_dir)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = [
        p
        for p in sorted(base.iterdir())
        if p.is_file()
        and p.suffix.lower() == ".pt"
        and not p.name.startswith(".")
        and not p.name.startswith("._")
    ]
    if not files:
        raise FileNotFoundError(f"No .pt files found in {input_dir}")
    return files


# Load HAFED checkpoint and build model
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


# Load selector policy checkpoint and build model
def load_policy(
    policy_path: str, device: torch.device
) -> Tuple[SashaPolicyValue, Dict]:
    ckpt = torch.load(policy_path, map_location=device)
    model = SashaPolicyValue(
        embed_dim=int(ckpt.get("embed_dim", 512)),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# Pick top-k candidate patches by HAFED attention score
def pick_candidate_pool(
    embeddings: torch.Tensor,
    hafed: HAFEDClassifier,
    max_candidates: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(embeddings.shape[0])
    if n <= 0:
        return embeddings, torch.zeros(0, dtype=torch.long)

    with torch.no_grad():
        toks = embeddings.unsqueeze(0).to(device)
        mask = torch.ones(1, n, dtype=torch.bool, device=device)
        _, attn, _ = hafed(toks, mask)
        attn = attn.squeeze(0).detach().cpu()

    k = min(max(1, int(max_candidates)), n)
    _, topi = torch.topk(attn, k=k, largest=True)
    topi, _ = torch.sort(topi)
    return embeddings[topi], topi


# Run greedy selector on candidate patches, returning chosen indices
def run_greedy_selector(
    candidates: torch.Tensor,
    policy: SashaPolicyValue,
    max_steps: int,
    device: torch.device,
) -> List[int]:
    n = int(candidates.shape[0])
    if n <= 0:
        return []

    selected = torch.zeros(n, dtype=torch.bool, device=device)
    chosen: List[int] = []
    global_context = candidates.mean(dim=0, keepdim=True).to(device)

    for step in range(max_steps):
        step_frac = torch.tensor(
            [[float(step) / max(1, max_steps)]], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            logits, _ = policy(
                candidates.unsqueeze(0).to(device),
                selected.unsqueeze(0),
                global_context,
                step_frac,
            )
        logits = logits.squeeze(0)
        logits[:-1] = logits[:-1].masked_fill(selected, torch.finfo(logits.dtype).min)
        action = int(torch.argmax(logits).item())
        stop = n

        if action == stop:
            break

        if selected[action]:
            continue

        selected[action] = True
        chosen.append(action)

    if not chosen:
        chosen = [int(torch.argmax(candidates.norm(dim=-1)).item())]
    return chosen


# Convert coord tensor to active_patches dict
def coords_to_active_patches(coords: torch.Tensor) -> Dict[Tuple[int, int, int], Dict]:
    active: Dict[Tuple[int, int, int], Dict] = {}
    for row in coords:
        if row.numel() < 3:
            continue
        lvl = int(round(float(row[0].item())))
        x = int(round(float(row[1].item())))
        y = int(round(float(row[2].item())))
        active[(lvl, x, y)] = {}
    return active


# Select patches for one WSI and compute prediction
def process_one(
    src_path: Path,
    dst_path: Path | None,
    hafed: HAFEDClassifier,
    policy: SashaPolicyValue,
    device: torch.device,
    max_candidates: int,
    max_steps: int,
    overwrite: bool,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "status": "invalid",
        "pred_idx": None,
        "true_idx": None,
        "probs": None,
    }
    should_save = dst_path is not None and (overwrite or not dst_path.exists())

    loaded = torch.load(src_path, map_location="cpu")
    if not isinstance(loaded, dict):
        result["status"] = "invalid"
        return result

    embeddings = loaded.get("embeddings")
    if embeddings is None:
        result["status"] = "invalid"
        return result

    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().float()
    else:
        embeddings = torch.tensor(embeddings, dtype=torch.float32)

    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)

    coords = loaded.get("coords")
    if coords is None:
        coords = torch.zeros(embeddings.shape[0], 3, dtype=torch.float32)
    elif torch.is_tensor(coords):
        coords = coords.detach().cpu().float()
    else:
        coords = torch.tensor(coords, dtype=torch.float32)

    if coords.dim() == 1:
        coords = coords.unsqueeze(0)

    if coords.shape[0] != embeddings.shape[0]:
        coords = torch.zeros(embeddings.shape[0], 3, dtype=torch.float32)

    candidate_emb, candidate_idx = pick_candidate_pool(
        embeddings=embeddings,
        hafed=hafed,
        max_candidates=max_candidates,
        device=device,
    )
    candidate_coords = (
        coords[candidate_idx] if candidate_idx.numel() > 0 else torch.zeros(1, 3)
    )

    chosen_local = run_greedy_selector(
        candidates=candidate_emb,
        policy=policy,
        max_steps=max_steps,
        device=device,
    )
    chosen_local_t = torch.tensor(chosen_local, dtype=torch.long)

    selected_embeddings = candidate_emb[chosen_local_t].detach().cpu()
    selected_coords = candidate_coords[chosen_local_t].detach().cpu()

    out_payload = {
        "case_id": str(loaded.get("case_id") or src_path.stem),
        "label": loaded.get("label"),
        "img_path": loaded.get("img_path"),
        "multistage": bool(loaded.get("multistage", False)),
        "active_patches": coords_to_active_patches(selected_coords),
        "zoomed_patches": loaded.get("zoomed_patches", {}),
        "embeddings": selected_embeddings,
        "coords": selected_coords,
        "patch_count": int(selected_embeddings.shape[0]),
        "source_pt_path": str(src_path),
        "selector": "SASHA",
        "selector_meta": {
            "max_candidates": int(max_candidates),
            "max_steps": int(max_steps),
            "selected_local_indices": [int(x) for x in chosen_local],
            "candidate_count": int(candidate_emb.shape[0]),
            "source_patch_count": int(embeddings.shape[0]),
        },
    }

    if should_save and dst_path is not None:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out_payload, dst_path)

    with torch.no_grad():
        mask = torch.ones(
            1, selected_embeddings.shape[0], dtype=torch.bool, device=device
        )
        logits, _, _ = hafed(selected_embeddings.unsqueeze(0).to(device), mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))

    true_label = None
    try:
        true_label = read_label_from_path(str(src_path))
    except Exception:
        true_label = None

    result.update(
        {
            "status": "ok",
            "pred_idx": pred_idx,
            "true_label": true_label,
            "probs": probs,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SASHA selector inference/export")
    parser.add_argument(
        "--output-dir",
        default="data/ablation/sasha/selected_test",
        help="Directory to save selected patch .pt files",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Disable saving selected patch .pt files",
    )
    parser.add_argument(
        "--input-dir",
        default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/full/test",
        help="Directory with per-WSI .pt files containing test embeddings",
    )
    parser.add_argument(
        "--hafed-checkpoint",
        default="data/models/ablation_patch_selector/sasha/best_hafed.pt",
    )
    parser.add_argument(
        "--selector-checkpoint",
        default="data/models/ablation_patch_selector/sasha/best_selector.pt",
    )
    parser.add_argument("--out-json", default="data/benchmark/sasha.json")
    parser.add_argument(
        "--benchmark-output", dest="benchmark_output", action="store_true", default=True
    )
    parser.add_argument(
        "--no-benchmark-output", dest="benchmark_output", action="store_false"
    )
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=24)
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    # Set up
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = list_pt_files(args.input_dir)
    if not args.no_output:
        os.makedirs(args.output_dir, exist_ok=True)

    hafed, hafed_ckpt = load_hafed(args.hafed_checkpoint, device=device)
    label_map = hafed_ckpt.get("label_map", {}) if isinstance(hafed_ckpt, dict) else {}
    int_to_label = {int(v): str(k) for k, v in label_map.items()}
    policy, _policy_ckpt = load_policy(args.selector_checkpoint, device=device)

    # Inference loop
    counts = {"ok": 0, "skipped": 0, "invalid": 0, "failed": 0}
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    predictions: List[Dict[str, object]] = []
    t0 = time.perf_counter()

    output_label = "none" if args.no_output else args.output_dir
    print(f"[START] files={len(files)} input={args.input_dir} output={output_label}")

    for idx, src in enumerate(files, start=1):
        dst = None if args.no_output else Path(args.output_dir) / src.name
        f0 = time.perf_counter()
        try:
            result = process_one(
                src_path=src,
                dst_path=dst,
                hafed=hafed,
                policy=policy,
                device=device,
                max_candidates=args.max_candidates,
                max_steps=args.max_steps,
                overwrite=args.overwrite,
            )
            status = str(result.get("status", "invalid"))
            counts[status] += 1
            true_label = result.get("true_label")
            if (
                true_label is not None
                and true_label in label_map
                and result.get("probs") is not None
            ):
                y_true.append(int(label_map[true_label]))
                y_pred.append(int(result.get("pred_idx")))
                y_prob.append(np.asarray(result.get("probs"), dtype=np.float32))

            predictions.append(
                {
                    "case_id": src.stem,
                    "pred_index": result.get("pred_idx"),
                    "pred_label": (
                        int_to_label.get(int(result.get("pred_idx")))
                        if result.get("pred_idx") is not None
                        else None
                    ),
                    "probs": (
                        np.asarray(result.get("probs"), dtype=np.float32).tolist()
                        if result.get("probs") is not None
                        else None
                    ),
                    "label": true_label,
                }
            )
        except Exception as exc:
            status = "failed"
            counts[status] += 1
            print(f"[FAIL] {src.name}: {exc}")

        if idx == 1 or idx % max(1, args.log_interval) == 0 or idx == len(files):
            dt = time.perf_counter() - f0
            elapsed = time.perf_counter() - t0
            print(
                f"[PROGRESS] {idx}/{len(files)} status={status} "
                f"last_file={dt:.2f}s elapsed={elapsed:.2f}s"
            )

    total = time.perf_counter() - t0
    print(
        f"[DONE] ok={counts['ok']} skipped={counts['skipped']} invalid={counts['invalid']} "
        f"failed={counts['failed']} total_time={total:.2f}s"
    )

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
        f"[DATA] total={len(files)} labeled={labels_np.size} skipped={len(files) - labels_np.size}"
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

    if args.benchmark_output and args.out_json:
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
