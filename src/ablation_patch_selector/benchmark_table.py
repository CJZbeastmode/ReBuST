"""Benchmark table generator for SASHA, EvoPS, and Streaming MIL v2."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.ablation_patch_selector.EvoPS_archive.evo import evo_select_subset
from src.ablation_patch_selector.SASHA_archive.data import read_label_from_path
from src.ablation_patch_selector.SASHA_archive.infer_pipeline import (
    load_policy,
    pick_candidate_pool,
    run_greedy_selector,
)
from archive.streaming_transformer_archive.streaming_transformer_v2.infer_streaming_mil import (
    _load_embeddings_and_coords,
    load_model as load_streaming_mil_v2,
    predict_one as predict_streaming_one,
)


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


def _safe_float(v: float) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": _safe_float((y_true == y_pred).mean() if len(y_true) > 0 else 0.0),
        "f1": float("nan"),
        "auc": float("nan"),
        "num_cases": int(len(y_true)),
    }

    if len(y_true) == 0:
        return metrics

    try:
        from sklearn.metrics import f1_score, roc_auc_score

        metrics["f1"] = _safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if y_prob.ndim == 2 and y_prob.size > 0:
            num_classes = int(y_prob.shape[1])
            if num_classes == 2:
                metrics["auc"] = _safe_float(roc_auc_score(y_true, y_prob[:, 1]))
            elif num_classes > 2:
                metrics["auc"] = _safe_float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        pass

    return metrics


def eval_sasha(
    test_dir: str,
    selector_path: str,
    device: torch.device,
    max_candidates: int,
    max_steps: int,
    selection_budget: int,
    streaming_model: object,
    streaming_label_map: Dict[str, int],
) -> Tuple[Dict[str, float], List[Dict]]:
    label_map = streaming_label_map
    int_to_label = {int(v): str(k) for k, v in label_map.items()}

    policy = None
    if selector_path and os.path.exists(selector_path):
        policy, _ = load_policy(selector_path, device=device)

    if policy is None:
        print(f"[WARN] SASHA selector checkpoint missing: {selector_path}. Using top-k by L2 norm.")

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    rows: List[Dict] = []

    for pt_path in list_pt_files(test_dir):
        try:
            true_label = read_label_from_path(str(pt_path))
        except Exception:
            continue
        if true_label not in label_map:
            continue

        embeddings, coords = _load_embeddings_and_coords(str(pt_path))
        candidates, candidate_idx = pick_candidate_pool(
            embeddings=embeddings,
            max_candidates=max_candidates,
        )
        if candidates.shape[0] <= 0:
            continue

        candidate_coords = coords[candidate_idx] if candidate_idx.numel() > 0 else coords[: candidates.shape[0]]

        if policy is not None:
            chosen = run_greedy_selector(
                candidates=candidates,
                policy=policy,
                max_steps=max_steps,
                device=device,
            )
        else:
            k = min(max(1, selection_budget), int(candidates.shape[0]))
            chosen = list(range(k))

        if not chosen:
            chosen = [int(torch.argmax(candidates.norm(dim=-1)).item())]

        chosen_t = torch.tensor(chosen, dtype=torch.long)
        subset = candidates[chosen_t]
        subset_coords = candidate_coords[chosen_t]
        out = predict_streaming_one(streaming_model, subset, subset_coords, device)
        probs = np.asarray(out["probs"], dtype=np.float32)

        pred_idx = int(out["pred"])
        true_idx = int(label_map[true_label])

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        y_prob.append(probs)
        rows.append(
            {
                "case_id": pt_path.stem,
                "true_label": true_label,
                "pred_label": int_to_label.get(pred_idx, str(pred_idx)),
                "probs": [float(x) for x in probs.tolist()],
            }
        )

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    y_prob_np = np.vstack(y_prob) if y_prob else np.zeros((0, len(label_map)), dtype=np.float32)
    return _compute_metrics(y_true_np, y_pred_np, y_prob_np), rows


def eval_evops(
    test_dir: str,
    device: torch.device,
    selection_budget: int,
    population_size: int,
    generations: int,
    elite_fraction: float,
    mutation_rate: float,
    crossover_rate: float,
    seed: int,
    streaming_model: object,
    streaming_label_map: Dict[str, int],
) -> Tuple[Dict[str, float], List[Dict]]:
    label_map = streaming_label_map
    int_to_label = {int(v): str(k) for k, v in label_map.items()}

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    rows: List[Dict] = []

    for pt_path in list_pt_files(test_dir):
        try:
            true_label = read_label_from_path(str(pt_path))
        except Exception:
            continue
        if true_label not in label_map:
            continue

        embeddings, coords = _load_embeddings_and_coords(str(pt_path))

        # Determine target class and fitness using the streaming transformer so
        # selection is optimised for the downstream classifier directly.
        with torch.no_grad():
            full_out = predict_streaming_one(streaming_model, embeddings, coords, device)
        target_idx = int(full_out["pred"])

        def _streaming_fitness(indices: List[int]) -> float:
            if not indices:
                return 0.0
            idx_t = torch.tensor(indices, dtype=torch.long)
            with torch.no_grad():
                sub_out = predict_streaming_one(
                    streaming_model, embeddings[idx_t], coords[idx_t], device
                )
            return float(np.asarray(sub_out["probs"], dtype=np.float32)[target_idx])

        scores = embeddings.norm(dim=-1)
        selected, _, _ = evo_select_subset(
            candidate_embeddings=embeddings,
            fitness_fn=_streaming_fitness,
            device=device,
            budget=selection_budget,
            population_size=population_size,
            generations=generations,
            elite_fraction=elite_fraction,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            seed=seed,
            scores=scores,
            target_idx=target_idx,
        )
        if not selected:
            selected = [int(torch.argmax(scores).item())]

        selected_t = torch.tensor(selected, dtype=torch.long)
        subset = embeddings[selected_t]
        subset_coords = coords[selected_t]
        out = predict_streaming_one(streaming_model, subset, subset_coords, device)
        probs = np.asarray(out["probs"], dtype=np.float32)

        pred_idx = int(out["pred"])
        true_idx = int(label_map[true_label])

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        y_prob.append(probs)
        rows.append(
            {
                "case_id": pt_path.stem,
                "true_label": true_label,
                "pred_label": int_to_label.get(pred_idx, str(pred_idx)),
                "probs": [float(x) for x in probs.tolist()],
            }
        )

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    y_prob_np = np.vstack(y_prob) if y_prob else np.zeros((0, len(label_map)), dtype=np.float32)
    return _compute_metrics(y_true_np, y_pred_np, y_prob_np), rows


def eval_streaming_mil_v2(
    test_dir: str,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict]]:
    model, label_map = load_streaming_mil_v2(checkpoint_path, device)
    int_to_label = {int(v): str(k) for k, v in label_map.items()}

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    rows: List[Dict] = []

    for pt_path in list_pt_files(test_dir):
        try:
            true_label = read_label_from_path(str(pt_path))
        except Exception:
            continue
        if true_label not in label_map:
            continue

        embeddings, coords = _load_embeddings_and_coords(str(pt_path))
        out = predict_streaming_one(model, embeddings, coords, device)
        pred_idx = int(out["pred"])
        probs = np.asarray(out["probs"], dtype=np.float32)
        true_idx = int(label_map[true_label])

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        y_prob.append(probs)
        rows.append(
            {
                "case_id": pt_path.stem,
                "true_label": true_label,
                "pred_label": int_to_label.get(pred_idx, str(pred_idx)),
                "probs": [float(x) for x in probs.tolist()],
            }
        )

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    y_prob_np = np.vstack(y_prob) if y_prob else np.zeros((0, len(label_map)), dtype=np.float32)
    return _compute_metrics(y_true_np, y_pred_np, y_prob_np), rows


def print_table(rows: List[Tuple[str, Dict[str, float]]]) -> None:
    print("| Method | Accuracy | AUC | F1 |")
    print("| --- | --- | --- | --- |")
    for name, metrics in rows:
        acc = metrics.get("accuracy", float("nan"))
        auc = metrics.get("auc", float("nan"))
        f1 = metrics.get("f1", float("nan"))
        print(f"| {name} | {acc:.4f} | {auc:.4f} | {f1:.4f} |")


def save_csv(rows: List[Tuple[str, Dict[str, float]]], out_csv: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "accuracy", "auc", "f1", "num_cases"])
        for name, metrics in rows:
            writer.writerow(
                [
                    name,
                    metrics.get("accuracy"),
                    metrics.get("auc"),
                    metrics.get("f1"),
                    metrics.get("num_cases"),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark table for TCGA test set")
    parser.add_argument("--test-dir", default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/a2c/test", help="Directory with per-WSI .pt files containing test embeddings")
    parser.add_argument("--out-csv", default="data/benchmark/evops_sasha_streaming_tcga.csv")

    parser.add_argument("--sasha-selector", default="data/models/ablation/sasha/best_selector.pt")
    parser.add_argument("--sasha-max-candidates", type=int, default=256)
    parser.add_argument("--sasha-max-steps", type=int, default=24)
    parser.add_argument("--sasha-selection-budget", type=int, default=32)

    parser.add_argument("--evops-budget", type=int, default=32)
    parser.add_argument("--evops-population", type=int, default=24)
    parser.add_argument("--evops-generations", type=int, default=12)
    parser.add_argument("--evops-elite", type=float, default=0.25)
    parser.add_argument("--evops-mutation", type=float, default=0.2)
    parser.add_argument("--evops-crossover", type=float, default=0.7)
    parser.add_argument("--evops-seed", type=int, default=42)

    parser.add_argument("--streaming-checkpoint", default="data/models/downstream_tasks/streaming_mil_v2/best_streaming_mil_v2.pt")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    rows: List[Tuple[str, Dict[str, float]]] = []

    streaming_model, streaming_label_map = load_streaming_mil_v2(
        args.streaming_checkpoint,
        device,
    )

    sasha_metrics, _ = eval_sasha(
        test_dir=args.test_dir,
        selector_path=args.sasha_selector,
        device=device,
        max_candidates=args.sasha_max_candidates,
        max_steps=args.sasha_max_steps,
        selection_budget=args.sasha_selection_budget,
        streaming_model=streaming_model,
        streaming_label_map=streaming_label_map,
    )
    rows.append(("SASHA+StreamingMILv2", sasha_metrics))

    evops_metrics, _ = eval_evops(
        test_dir=args.test_dir,
        device=device,
        selection_budget=args.evops_budget,
        population_size=args.evops_population,
        generations=args.evops_generations,
        elite_fraction=args.evops_elite,
        mutation_rate=args.evops_mutation,
        crossover_rate=args.evops_crossover,
        seed=args.evops_seed,
        streaming_model=streaming_model,
        streaming_label_map=streaming_label_map,
    )
    rows.append(("EvoPS+StreamingMILv2", evops_metrics))

    streaming_metrics, _ = eval_streaming_mil_v2(
        test_dir=args.test_dir,
        checkpoint_path=args.streaming_checkpoint,
        device=device,
    )
    rows.append(("StreamingMILv2", streaming_metrics))

    print_table(rows)
    save_csv(rows, args.out_csv)
    print(f"[SAVED] {args.out_csv}")


if __name__ == "__main__":
    main()
