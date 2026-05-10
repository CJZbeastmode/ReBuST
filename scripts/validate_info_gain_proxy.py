#!/usr/bin/env python3
"""Validate information-gain proxy scores against downstream label gain.

This script compares candidate proxy scores (e.g. entropy, contrastive, centroid)
against actual downstream gain and reports:

- Mutual information proxies (kNN MI and discretized MI/NMI)
- Rank correlation (Spearman rho and Kendall tau)
- A ranked summary table (highest signal first)

Input expectations
------------------
CSV rows represent one comparable unit (episode/selection/case).
Provide either:
1) a direct gain column via --gain-col, or
2) before/after metric columns via --before-col and --after-col,
   where gain = after - before.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from scipy.stats import kendalltau, spearmanr
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for rank correlations. Install with: pip install scipy"
    ) from exc

try:
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.metrics import mutual_info_score
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for MI estimation. Install with: pip install scikit-learn"
    ) from exc


DEFAULT_SCORE_ALIASES: Dict[str, List[str]] = {
    "entropy": [
        "entropy_score",
        "score_entropy",
        "entropy",
        "entropy_gain",
    ],
    "contrastive": [
        "contrastive_score",
        "score_contrastive",
        "contrastive",
        "contrastive_text_score",
    ],
    "centroid": [
        "centroid_score",
        "score_centroid",
        "centroid",
        "cancer_centroid_score",
    ],
}


@dataclass
class EvalRow:
    score_name: str
    mi_knn: float
    mi_disc: float
    nmi_disc: float
    nmi_knn_proxy: float
    spearman_rho: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float
    n_valid: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate info-gain proxies with MI and rank correlation."
    )
    parser.add_argument("--csv", required=True, help="Input CSV path")

    parser.add_argument(
        "--score-cols",
        nargs="+",
        default=None,
        help=(
            "Explicit score columns to evaluate. "
            "If omitted, script auto-detects entropy/contrastive/centroid aliases."
        ),
    )

    parser.add_argument(
        "--gain-col",
        default=None,
        help="Column containing actual downstream gain directly",
    )
    parser.add_argument(
        "--before-col",
        default=None,
        help="Before metric column (used only if --gain-col is not provided)",
    )
    parser.add_argument(
        "--after-col",
        default=None,
        help="After metric column (used only if --gain-col is not provided)",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Quantile bin count for discretized MI/NMI (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for kNN MI estimator (default: 0)",
    )

    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional output CSV path for ranked summary",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional output JSON path for ranked summary",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Optional directory to save scatter/rank plots",
    )

    return parser.parse_args()


def _safe_float(value: str) -> float:
    if value is None:
        return math.nan
    txt = str(value).strip()
    if txt == "":
        return math.nan
    try:
        return float(txt)
    except Exception:
        return math.nan


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def detect_score_columns(header: Iterable[str]) -> List[str]:
    header_list = list(header)
    lowered = {h.lower(): h for h in header_list}

    detected: List[str] = []
    for canonical in ("entropy", "contrastive", "centroid"):
        found = None
        for alias in DEFAULT_SCORE_ALIASES[canonical]:
            if alias.lower() in lowered:
                found = lowered[alias.lower()]
                break
        if found is not None:
            detected.append(found)

    if detected:
        return detected

    candidate = [
        h
        for h in header_list
        if "score" in h.lower() or "entropy" in h.lower() or "centroid" in h.lower()
    ]
    return candidate


def quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.array([], dtype=int)

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=int)

    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(finite, q)
    edges = np.unique(edges)

    if len(edges) <= 2:
        return np.zeros_like(values, dtype=int)

    # digitize using internal edges only
    internal = edges[1:-1]
    binned = np.digitize(values, internal, right=False)
    binned[~np.isfinite(values)] = -1
    return binned.astype(int)


def entropy_from_bins(bins: np.ndarray) -> float:
    valid = bins[bins >= 0]
    if valid.size == 0:
        return 0.0
    counts = np.bincount(valid)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def evaluate_single(
    score: np.ndarray, gain: np.ndarray, bins: int, seed: int
) -> Tuple[float, float, float, float, float, float, float, float, int]:
    mask = np.isfinite(score) & np.isfinite(gain)
    x = score[mask]
    y = gain[mask]
    n_valid = int(mask.sum())

    if n_valid < 5:
        return (
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            n_valid,
        )

    # kNN MI (continuous target)
    mi_knn = float(mutual_info_regression(x.reshape(-1, 1), y, random_state=seed)[0])

    # discretized MI/NMI
    x_bin = quantile_bin(x, bins)
    y_bin = quantile_bin(y, bins)
    valid_disc = (x_bin >= 0) & (y_bin >= 0)

    if valid_disc.sum() < 5:
        mi_disc = math.nan
        nmi_disc = math.nan
    else:
        mi_disc = float(mutual_info_score(x_bin[valid_disc], y_bin[valid_disc]))
        h_y = entropy_from_bins(y_bin[valid_disc])
        nmi_disc = float(mi_disc / h_y) if h_y > 0 else math.nan

    h_y_cont_proxy = entropy_from_bins(y_bin)
    nmi_knn_proxy = float(mi_knn / h_y_cont_proxy) if h_y_cont_proxy > 0 else math.nan

    rho, p_rho = spearmanr(x, y, nan_policy="omit")
    tau, p_tau = kendalltau(x, y, nan_policy="omit")

    return (
        mi_knn,
        mi_disc,
        nmi_disc,
        nmi_knn_proxy,
        float(rho),
        float(p_rho),
        float(tau),
        float(p_tau),
        n_valid,
    )


def maybe_make_plots(
    plot_dir: str,
    rows: List[EvalRow],
    score_vectors: Dict[str, np.ndarray],
    gain: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not installed; skipping plots.")
        return

    os.makedirs(plot_dir, exist_ok=True)

    for row in rows:
        x = score_vectors[row.score_name]
        mask = np.isfinite(x) & np.isfinite(gain)
        x_ = x[mask]
        y_ = gain[mask]

        if len(x_) == 0:
            continue

        fig = plt.figure(figsize=(6, 4))
        plt.scatter(x_, y_, s=14, alpha=0.7)
        plt.title(f"{row.score_name} vs gain")
        plt.xlabel(row.score_name)
        plt.ylabel("gain")
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"scatter_{row.score_name}.png"), dpi=150)
        plt.close(fig)

        fig = plt.figure(figsize=(6, 4))
        rx = np.argsort(np.argsort(x_))
        ry = np.argsort(np.argsort(y_))
        plt.scatter(rx, ry, s=14, alpha=0.7)
        plt.title(f"Rank plot: {row.score_name} vs gain")
        plt.xlabel(f"rank({row.score_name})")
        plt.ylabel("rank(gain)")
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"rank_{row.score_name}.png"), dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_csv_rows(args.csv)

    header = rows[0].keys()
    score_cols = args.score_cols or detect_score_columns(header)
    if not score_cols:
        raise ValueError(
            "Could not detect score columns. Pass them explicitly via --score-cols."
        )

    if args.gain_col:
        if args.gain_col not in header:
            raise ValueError(f"--gain-col not found in CSV: {args.gain_col}")
        gain = np.array([_safe_float(r.get(args.gain_col)) for r in rows], dtype=float)
    else:
        if not args.before_col or not args.after_col:
            raise ValueError("Provide --gain-col OR both --before-col and --after-col.")
        if args.before_col not in header:
            raise ValueError(f"--before-col not found in CSV: {args.before_col}")
        if args.after_col not in header:
            raise ValueError(f"--after-col not found in CSV: {args.after_col}")
        before = np.array(
            [_safe_float(r.get(args.before_col)) for r in rows], dtype=float
        )
        after = np.array(
            [_safe_float(r.get(args.after_col)) for r in rows], dtype=float
        )
        gain = after - before

    score_vectors: Dict[str, np.ndarray] = {}
    for c in score_cols:
        if c not in header:
            raise ValueError(f"Score column not found in CSV: {c}")
        score_vectors[c] = np.array([_safe_float(r.get(c)) for r in rows], dtype=float)

    results: List[EvalRow] = []
    for c in score_cols:
        (
            mi_knn,
            mi_disc,
            nmi_disc,
            nmi_knn_proxy,
            rho,
            p_rho,
            tau,
            p_tau,
            n_valid,
        ) = evaluate_single(score_vectors[c], gain, bins=args.bins, seed=args.seed)

        results.append(
            EvalRow(
                score_name=c,
                mi_knn=mi_knn,
                mi_disc=mi_disc,
                nmi_disc=nmi_disc,
                nmi_knn_proxy=nmi_knn_proxy,
                spearman_rho=rho,
                spearman_p=p_rho,
                kendall_tau=tau,
                kendall_p=p_tau,
                n_valid=n_valid,
            )
        )

    results.sort(
        key=lambda r: (
            -np.nan_to_num(r.nmi_disc, nan=-1e9),
            -np.nan_to_num(r.spearman_rho, nan=-1e9),
        )
    )

    print("\n=== Info-gain Proxy Validation (ranked) ===")
    print(
        "score, n_valid, nmi_disc, mi_disc, mi_knn, spearman_rho, spearman_p, kendall_tau, kendall_p"
    )
    for r in results:
        print(
            f"{r.score_name}, {r.n_valid}, {r.nmi_disc:.6f}, {r.mi_disc:.6f}, {r.mi_knn:.6f}, "
            f"{r.spearman_rho:.6f}, {r.spearman_p:.3g}, {r.kendall_tau:.6f}, {r.kendall_p:.3g}"
        )

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "score",
                    "n_valid",
                    "nmi_disc",
                    "mi_disc",
                    "mi_knn",
                    "nmi_knn_proxy",
                    "spearman_rho",
                    "spearman_p",
                    "kendall_tau",
                    "kendall_p",
                ]
            )
            for r in results:
                writer.writerow(
                    [
                        r.score_name,
                        r.n_valid,
                        r.nmi_disc,
                        r.mi_disc,
                        r.mi_knn,
                        r.nmi_knn_proxy,
                        r.spearman_rho,
                        r.spearman_p,
                        r.kendall_tau,
                        r.kendall_p,
                    ]
                )
        print(f"[OK] Wrote CSV summary: {args.out_csv}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        payload = [r.__dict__ for r in results]
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[OK] Wrote JSON summary: {args.out_json}")

    if args.plot_dir:
        maybe_make_plots(args.plot_dir, results, score_vectors, gain)
        print(f"[OK] Wrote plots under: {args.plot_dir}")


if __name__ == "__main__":
    main()
