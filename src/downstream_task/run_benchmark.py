"""
Orchestrate the full benchmark across all (or a subset of) patch-selection methods.

Pipeline
--------
1. **Extraction phase** — for each method calls
   ``extract_method_embeddings.main()`` to produce
   ``data/extracted_embeddings/{method}/{case_id}.pt``.
   Pass ``--skip-extraction`` to re-use files already on disk.
2. **Benchmark phase** — calls
   ``benchmark_classification.run_kfold_benchmark()`` for each method.
3. **Results phase** — writes ``data/benchmark/results.csv`` (one row per
   method) and the full JSON with per-fold breakdowns to
   ``data/benchmark/results.json``.

Usage
-----
# Full run (all methods):
python src/downstream_task/run_benchmark.py \\
    --a2c-model        data/models/rl/a2c_lvl4/a2c_lvl4_final.pt \\
    --supervised-model data/models/supervised/score_regressor_final.pt \\
    --k 5 --epochs 20 --device cpu

# Only non-ML methods (no model checkpoints needed):
python src/downstream_task/run_benchmark.py \\
    --methods full greedy humbe \\
    --budget 0.8 --k 5 --epochs 20

# Skip extraction (embeddings already extracted), re-run benchmark only:
python src/downstream_task/run_benchmark.py \\
    --methods humbe humbe_a2c \\
    --a2c-model data/models/rl/a2c_lvl4/a2c_lvl4_final.pt \\
    --skip-extraction \\
    --k 5 --epochs 20
"""

import sys
import os
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.downstream_task.extract_method_embeddings import main as _extract_main
from src.downstream_task.benchmark_classification import run_kfold_benchmark

ALL_METHODS = ["full", "greedy", "a2c", "humbe", "humbe_a2c", "supervised"]

# Methods that require an A2C model checkpoint
_A2C_METHODS = {"a2c", "humbe_a2c"}
# Methods that require a supervised model checkpoint
_SUP_METHODS = {"supervised"}

# Columns written to results.csv
_CSV_FIELDS = [
    "method",
    "accuracy_mean",
    "accuracy_std",
    "f1_mean",
    "f1_std",
    "auc_mean",
    "auc_std",
    "avg_patches_per_wsi",
    "num_cases",
    "k",
    "epochs",
]


# ============================================================
# Helpers
# ============================================================

class _Namespace:
    """Minimal attribute bag used when calling extract_main programmatically."""
    pass


def _build_extract_args(
    method: str,
    images_dir: str,
    labels_json: str,
    out_dir: str,
    model_path,
    budget: float,
    max_depth: int,
    score: str,
    stochastic: bool,
    force_extract: bool,
) -> _Namespace:
    ns = _Namespace()
    ns.method = method
    ns.images_dir = images_dir
    ns.labels_json = labels_json
    ns.out_dir = out_dir
    ns.model_path = model_path
    ns.budget = budget
    ns.max_depth = max_depth
    ns.score = score
    ns.stochastic = stochastic
    ns.force = force_extract
    return ns


def _print_rule(char: str = "=", width: int = 70) -> None:
    print(char * width)


def _print_table(results: Dict[str, Dict]) -> None:
    """Pretty-print an aligned summary table."""
    _print_rule()
    hdr = (
        f"{'Method':<15} "
        f"{'ACC':>7} {'±':>6} "
        f"{'F1':>7} {'±':>6} "
        f"{'AUC':>7} {'±':>6} "
        f"{'Patches/WSI':>13}"
    )
    print(hdr)
    _print_rule("-")
    for method, res in results.items():
        nan = float("nan")
        print(
            f"{method:<15} "
            f"{res.get('accuracy_mean', nan):>7.4f} "
            f"{res.get('accuracy_std',  nan):>6.4f} "
            f"{res.get('f1_mean',       nan):>7.4f} "
            f"{res.get('f1_std',        nan):>6.4f} "
            f"{res.get('auc_mean',      nan):>7.4f} "
            f"{res.get('auc_std',       nan):>6.4f} "
            f"{res.get('avg_patches_per_wsi', nan):>13.1f}"
        )
    _print_rule()


# ============================================================
# Main orchestrator
# ============================================================

def run_benchmark(args) -> None:
    methods: List[str] = args.methods

    # Resolve output paths
    out_csv = getattr(args, "out_csv", "data/benchmark/results.csv")
    out_dir_root = os.path.dirname(os.path.abspath(out_csv))
    os.makedirs(out_dir_root, exist_ok=True)

    results_all: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Phase 1 — Embedding extraction
    # ------------------------------------------------------------------
    if not args.skip_extraction:
        _print_rule()
        print("PHASE 1 / 2 — EMBEDDING EXTRACTION")
        _print_rule()

        for method in methods:
            # Validate that required model paths are present
            if method in _A2C_METHODS and not args.a2c_model:
                print(
                    f"  [SKIP EXTRACT] {method}: --a2c-model not provided"
                )
                continue
            if method in _SUP_METHODS and not args.supervised_model:
                print(
                    f"  [SKIP EXTRACT] {method}: --supervised-model not provided"
                )
                continue

            model_path = (
                args.supervised_model
                if method in _SUP_METHODS
                else args.a2c_model
            )
            emb_out_dir = os.path.join("data", "extracted_embeddings", method)

            print(f"\n  ── Extracting embeddings: method={method}")
            try:
                ea = _build_extract_args(
                    method=method,
                    images_dir=args.images_dir,
                    labels_json=args.labels_json,
                    out_dir=emb_out_dir,
                    model_path=model_path,
                    budget=args.budget,
                    max_depth=args.max_depth,
                    score="text_align_score",
                    stochastic=False,
                    force_extract=args.force_extract,
                )
                _extract_main(ea)
            except Exception as exc:
                import traceback
                print(f"  [EXTRACT FAIL] {method}: {exc}")
                traceback.print_exc()
    else:
        print("[SKIP] Extraction phase skipped (--skip-extraction).")

    # ------------------------------------------------------------------
    # Phase 2 — K-fold benchmark
    # ------------------------------------------------------------------
    _print_rule()
    print("PHASE 2 / 2 — K-FOLD BENCHMARK")
    _print_rule()

    for method in methods:
        emb_dir = os.path.join("data", "extracted_embeddings", method)

        # Check that the directory has at least one .pt file
        if not os.path.isdir(emb_dir) or not any(
            fname.endswith(".pt") for fname in os.listdir(emb_dir)
        ):
            print(
                f"\n  [SKIP BENCHMARK] {method}: "
                f"no .pt files in {emb_dir}"
            )
            continue

        print(f"\n  ── Benchmarking method={method}  dir={emb_dir}")
        try:
            result = run_kfold_benchmark(
                embeddings_dir=emb_dir,
                labels_json=args.labels_json,
                k=args.k,
                epochs=args.epochs,
                lr=args.lr,
                device=args.device,
                seed=args.seed,
            )
            result["method"] = method
            results_all[method] = result
        except Exception as exc:
            import traceback
            print(f"  [BENCHMARK FAIL] {method}: {exc}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Phase 3 — Save results
    # ------------------------------------------------------------------
    if not results_all:
        print("\n[DONE] No benchmark results to save.")
        return

    # CSV — one row per method
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for method, res in results_all.items():
            row = {
                field: res.get(field, "")
                for field in _CSV_FIELDS
            }
            row["method"] = method
            writer.writerow(row)
    print(f"\n[RESULTS] CSV    → {out_csv}")

    # JSON — includes per-fold breakdowns and label map
    json_path = out_csv.replace(".csv", ".json")
    with open(json_path, "w") as fh:
        json.dump(results_all, fh, indent=2)
    print(f"[RESULTS] JSON   → {json_path}")

    # Pretty-print summary table
    print()
    _print_table(results_all)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run the full patch-selection benchmark across all methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Method selection ---
    p.add_argument(
        "--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS,
        help="Methods to benchmark (default: all)",
    )

    # --- Data paths ---
    p.add_argument("--images-dir", default="data/images")
    p.add_argument("--labels-json", default="data/labels_main.json")

    # --- Model checkpoints ---
    p.add_argument(
        "--a2c-model", default=None,
        help="A2C checkpoint path (required for 'a2c' and 'humbe_a2c')",
    )
    p.add_argument(
        "--supervised-model", default=None,
        help="Score-regressor checkpoint path (required for 'supervised')",
    )

    # --- Selection hyper-parameters ---
    p.add_argument(
        "--budget", type=float, default=0.8,
        help="HUMBE budget ratio",
    )
    p.add_argument(
        "--max-depth", type=int, default=6,
        help="Maximum zoom depth for greedy / a2c",
    )

    # --- Benchmark hyper-parameters ---
    p.add_argument("--k", type=int, default=5, help="Number of CV folds")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs per fold")
    p.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate")
    p.add_argument(
        "--device", default="cpu",
        help='Torch device ("cpu", "cuda", "mps")',
    )
    p.add_argument("--seed", type=int, default=42)

    # --- Run control ---
    p.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip embedding extraction; use .pt files already on disk",
    )
    p.add_argument(
        "--force-extract", action="store_true",
        help="Re-extract embeddings even if .pt files already exist",
    )
    p.add_argument(
        "--out-csv", default="data/benchmark/results.csv",
        help="Path for the output CSV",
    )

    args = p.parse_args()
    run_benchmark(args)
