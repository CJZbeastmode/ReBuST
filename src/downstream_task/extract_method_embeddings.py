"""
Extract PLIP embeddings for all WSIs using a given patch-selection method.

For each WSI that exists in ``images_dir`` and appears in ``labels_json``,
this script:
  1. Runs the selection method to obtain a list of (level, x, y) coordinates.
  2. Reads each selected patch via the WSI object.
  3. Embeds every patch with PLIP.
  4. Saves ``data/extracted_embeddings/{method}/{case_id}.pt`` as::

        {
            "embeddings":  Tensor[N, 512],
            "patch_count": int,
            "method":      str,
            "case_id":     str,
        }

Usage examples
--------------
# Full-slide baseline (all patches at max_level):
python src/downstream_task/extract_method_embeddings.py --method full

# Greedy selection:
python src/downstream_task/extract_method_embeddings.py --method greedy --max-depth 6

# A2C only (no HUMBE):
python src/downstream_task/extract_method_embeddings.py \\
    --method a2c

# HUMBE only:
python src/downstream_task/extract_method_embeddings.py --method humbe --budget 0.8

# HUMBE + A2C (production variant):
python src/downstream_task/extract_method_embeddings.py \\
    --method humbe_a2c \\
    --budget 0.8

# Supervised score regressor:
python src/downstream_task/extract_method_embeddings.py \\
    --method supervised_regressor
"""

import sys
import argparse
import os
from pathlib import Path

# Make sure repo root is importable when run directly.
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import json
import torch
from tqdm import tqdm

from src.utils.wsi import WSI
from src.utils.embedder import Embedder

METHODS = ["humbe_a2c", "humbe", "a2c", "greedy", "supervised_regressor", "supervised_zoom", "full"]
DEFAULT_MODELS = {
    "humbe_a2c": "data/models/rl/a2c_lvl4/a2c_lvl4_final.pt",
    "a2c": "data/models/rl/a2c_lvl4/a2c_lvl4_final.pt",
    "supervised_regressor": "data/models/supervised/score_regressor_final.pt",
    "supervised_zoom": "data/models/supervised/zoom_classifier_final.pt",
}

METHODS_REQUIRING_MODEL = {"a2c", "humbe_a2c", "supervised_regressor", "supervised_zoom"}


DEFAULT_IMAGES_DIR = "/Volumes/Xbox_HD/Data/med_img"
DEFAULT_OUT_ROOT   = "/Volumes/Xbox_HD/Data/downstream_data"


def parse_label_from_stem(stem: str) -> str:
    """Extract the cancer-type label from a TCGA filename stem.

    E.g. ``TCGA-D3-A1QB-SKCM`` → ``SKCM``.
    Falls back to the whole stem if there is no hyphen.
    """
    return stem.rsplit("-", 1)[-1]


def discover_cases(images_dir: str) -> dict:
    """Scan *images_dir* for .svs files and return ``{case_id: label}``."""
    cases = {}
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(".svs"):
            continue
        stem = os.path.splitext(fname)[0]
        cases[stem] = parse_label_from_stem(stem)
    return cases


# ============================================================
# Device helper
# ============================================================

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


# ============================================================
# Shared embedding extractor
# ============================================================

def extract_embeddings_for_patches(
    wsi: WSI,
    patch_coords: list,
    embedder: Embedder,
) -> torch.Tensor:
    """Embed each (level, x, y) patch with PLIP; return stacked Tensor [N, 512]."""
    embs = []
    for lvl, x, y in patch_coords:
        try:
            patch = wsi.get_patch(lvl, x, y)
            emb = embedder.img_emb(patch)
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().float().view(-1)
            else:
                emb = torch.tensor(emb, dtype=torch.float32).view(-1)
            embs.append(emb)
        except Exception as e:
            print(f"    [SKIP] lvl={lvl} x={x} y={y}: {e}")
    if not embs:
        return torch.zeros(1, 512)
    return torch.stack(embs)  # [N, 512]


# ============================================================
# Per-method selection functions
# Each returns a list of (level, x, y) tuples.
# ============================================================

def select_full(wsi: WSI, **kwargs) -> list:
    """All patches at the coarsest level (max_level)."""
    lvl = wsi.max_level
    w, h = wsi.levels_info[lvl]["size"]
    ps = wsi.patch_size
    return [
        (lvl, x, y)
        for y in range(0, h, ps)
        for x in range(0, w, ps)
    ]


def select_humbe(
    wsi: WSI,
    budget_ratio: float = 0.8,
    score_key: str = "text_align_score",
    **kwargs,
) -> list:
    """HUMBE global budget enforcement; uses wsi.active_patches after selection."""
    from src.global_budget_enforcer.HUMBE import humbe
    from src.utils.patch_scores import PATCH_SCORE_MODULES

    score_module = PATCH_SCORE_MODULES[score_key]()
    wsi = humbe(wsi, score_module=score_module, budget_ratio=budget_ratio, verbose=False)
    return list(wsi.active_patches.keys())  # list of (lvl, x, y)


def select_humbe_a2c(
    wsi: WSI,
    model_path: str,
    budget_ratio: float = 0.8,
    score_key: str = "text_align_score",
    deterministic: bool = True,
    **kwargs,
) -> list:
    """HUMBE followed by A2C refinement (production variant)."""
    from src.global_budget_enforcer.HUMBE import humbe
    from src.utils.patch_scores import PATCH_SCORE_MODULES
    from src.inference.a2c.infer_rl_a2c import infer_wsi_a2c

    score_module = PATCH_SCORE_MODULES[score_key]()
    wsi = humbe(wsi, score_module=score_module, budget_ratio=budget_ratio, verbose=False)
    wsi = infer_wsi_a2c(
        wsi, model_path=model_path, deterministic=deterministic, verbose=False
    )
    return list(wsi.active_patches.keys())


def select_a2c(
    wsi: WSI,
    model_path: str,
    max_depth: int = 6,
    deterministic: bool = True,
    **kwargs,
) -> list:
    """Standalone A2C patch selection (no HUMBE pre-filtering)."""
    from src.inference.a2c.infer_rl_a2c import infer_wsi_a2c

    wsi = infer_wsi_a2c(
        wsi,
        model_path=model_path,
        deterministic=deterministic,
        verbose=False,
    )
    return list(wsi.active_patches.keys())


def select_greedy(
    wsi: WSI,
    max_depth: int = 6,
    **kwargs,
) -> list:
    """Greedy information-gain patch selection."""
    from src.inference.greedy_infer import greedy_infer_zoom
    from src.utils.dynamic_patch_env import DynamicPatchEnv

    env = DynamicPatchEnv(wsi, patch_score="text_align_score")
    lvl = wsi.max_level
    w, h = wsi.levels_info[lvl]["size"]
    ps = wsi.patch_size

    kept_all = []
    for y in range(0, h, ps):
        for x in range(0, w, ps):
            k, _ = greedy_infer_zoom(env, lvl, x, y, max_depth=max_depth)
            kept_all.extend(k)

    return [(meta["level"], meta["x"], meta["y"]) for _, meta in kept_all]


def _load_supervised_model(model_path: str, device: torch.device):
    """Load a supervised model checkpoint (plain state-dict or wrapped)."""
    from src.training.supervised.score_regressor import ScoreRegressor
    model = ScoreRegressor(state_dim=515, hidden=256)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def select_supervised_regressor(
    wsi: WSI,
    model_path: str,
    max_depth: int = 6,
    **kwargs,
) -> list:
    """Supervised score-regressor patch selection."""
    from src.inference.supervised_score_regressor_infer import (
        greedy_infer_zoom_regressor,
    )
    from src.utils.dynamic_patch_env import DynamicPatchEnv

    device = _get_device()
    model = _load_supervised_model(model_path, device)
    env = DynamicPatchEnv(wsi, patch_score="text_align_score")
    lvl = wsi.max_level
    w, h = wsi.levels_info[lvl]["size"]
    ps = wsi.patch_size

    kept_all = []
    for y in range(0, h, ps):
        for x in range(0, w, ps):
            k, _ = greedy_infer_zoom_regressor(
                env, model, lvl, x, y, max_depth=max_depth, device=device
            )
            kept_all.extend(k)

    return [(meta["level"], meta["x"], meta["y"]) for _, meta in kept_all]


def select_supervised_zoom(
    wsi: WSI,
    model_path: str,
    max_depth: int = 6,
    **kwargs,
) -> list:
    """Supervised zoom-classifier patch selection."""
    from src.inference.supervised_zoom_classifier_infer import (
        greedy_infer_zoom_regressor,
    )
    from src.utils.dynamic_patch_env import DynamicPatchEnv

    device = _get_device()
    model = _load_supervised_model(model_path, device)
    env = DynamicPatchEnv(wsi, patch_score="text_align_score")
    lvl = wsi.max_level
    w, h = wsi.levels_info[lvl]["size"]
    ps = wsi.patch_size

    kept_all = []
    for y in range(0, h, ps):
        for x in range(0, w, ps):
            k, _ = greedy_infer_zoom_regressor(
                env, model, lvl, x, y, max_depth=max_depth, device=device
            )
            kept_all.extend(k)

    return [(meta["level"], meta["x"], meta["y"]) for _, meta in kept_all]


SELECTORS = {
    "full": select_full,
    "greedy": select_greedy,
    "a2c": select_a2c,
    "humbe": select_humbe,
    "humbe_a2c": select_humbe_a2c,
    "supervised_regressor": select_supervised_regressor,
    "supervised_zoom": select_supervised_zoom,
}


# ============================================================
# Main
# ============================================================

def _run_one_method(args, method: str) -> None:
    """Extract embeddings for a single *method* and write .pt files."""
    if method not in SELECTORS:
        raise ValueError(f"Unknown method '{method}'. Choose from {METHODS}")

    model_path = DEFAULT_MODELS.get(method)
    if method in METHODS_REQUIRING_MODEL and not model_path:
        raise ValueError(
            f"No default model path configured for method '{method}'."
        )
    if model_path and not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Default checkpoint for method '{method}' not found: {model_path}"
        )

    out_root = args.out_dir or DEFAULT_OUT_ROOT
    out_dir = os.path.join(out_root, method)
    os.makedirs(out_dir, exist_ok=True)

    images_dir = os.path.abspath(args.images_dir)

    # ── Build {case_id: label} ──────────────────────────────────────────────
    # Prefer labels_json if explicitly supplied and the file exists;
    # otherwise fall back to parsing labels directly from filenames.
    if args.labels_json and os.path.exists(args.labels_json):
        with open(args.labels_json, "r") as f:
            labels = json.load(f)
        print(f"[LABELS] Loaded {len(labels)} entries from {args.labels_json}")
    else:
        labels = discover_cases(images_dir)
        print(f"[LABELS] Discovered {len(labels)} cases from filenames in {images_dir}")

    if not labels:
        print(f"[WARN] No cases found — check images_dir: {images_dir}")
        return

    # Shared kwargs forwarded to every selector;
    # selectors that don't need a kwarg simply ignore it.
    selector_kwargs = dict(
        model_path=model_path,
        budget_ratio=args.budget,
        max_depth=args.max_depth,
        score_key=args.score,
        deterministic=not args.stochastic,
    )

    embedder = Embedder(img_backend="plip")
    selector_fn = SELECTORS[method]

    case_ids = sorted(labels.keys())
    print(
        f"[EXTRACT] method={method}  cases={len(case_ids)}  out_dir={out_dir}"
    )

    success, skipped, failed = 0, 0, 0

    for case_id in tqdm(case_ids, desc=f"extract/{method}"):
        out_path = os.path.join(out_dir, f"{case_id}.pt")

        if os.path.exists(out_path) and not args.force:
            skipped += 1
            continue

        img_path = os.path.join(images_dir, f"{case_id}.svs")
        if not os.path.exists(img_path):
            print(f"  [MISSING] {img_path}")
            failed += 1
            continue

        try:
            wsi = WSI(img_path, multistage=(method == "humbe_a2c"), embedder=embedder)
            coords = selector_fn(wsi, **selector_kwargs)
            embeddings = extract_embeddings_for_patches(wsi, coords, embedder)

            label = labels[case_id]
            torch.save(
                {
                    "embeddings": embeddings,   # [N, 512]
                    "patch_count": len(coords),
                    "method": method,
                    "case_id": case_id,
                    "label": label,
                },
                out_path,
            )
            print(
                f"  [SAVED] {case_id}: {len(coords)} patches → {out_path}"
            )
            success += 1

        except Exception as e:
            import traceback
            print(f"  [FAIL] {case_id}: {e}")
            traceback.print_exc()
            failed += 1

    print(
        f"\n[DONE] method={method}  success={success}  skipped={skipped}  failed={failed}"
    )

    # ── Consolidate individual .pt files into one dataset file ──────────────
    _consolidate(out_dir, method, out_root=out_root)


def _consolidate(pt_dir: str, method: str, out_root: str) -> None:
    """Read every per-WSI .pt in *pt_dir* and write a single dataset file.

    Output: ``{out_root}/{method}_dataset.pt``::

        {
            "embeddings": [Tensor[N0,512], Tensor[N1,512], ...],  # one per WSI
            "labels":     ["SKCM", "LUAD", ...],
            "case_ids":   ["TCGA-...", ...],
            "method":     str,
            "n_wsi":      int,
        }
    """
    pt_files = sorted(f for f in os.listdir(pt_dir) if f.endswith(".pt"))
    if not pt_files:
        print(f"[CONSOLIDATE] No .pt files found in {pt_dir} — skipping.")
        return

    all_embeddings, all_labels, all_case_ids = [], [], []
    for fname in tqdm(pt_files, desc=f"consolidate/{method}"):
        data = torch.load(os.path.join(pt_dir, fname), map_location="cpu")
        all_embeddings.append(data["embeddings"])   # Tensor [N, 512]
        all_labels.append(data["label"])
        all_case_ids.append(data["case_id"])

    out_path = os.path.join(out_root, f"{method}_dataset.pt")
    torch.save(
        {
            "embeddings": all_embeddings,   # list of variable-length tensors
            "labels":     all_labels,
            "case_ids":   all_case_ids,
            "method":     method,
            "n_wsi":      len(all_embeddings),
        },
        out_path,
    )
    print(
        f"[CONSOLIDATE] {method}: {len(all_embeddings)} WSIs → {out_path}"
    )


def main(args):
    # Default behaviour (no flags supplied) is to run all methods.
    if args.method is None:
        for m in METHODS:
            print(f"\n{'='*60}\n[ALL-METHODS] Starting: {m}\n{'='*60}")
            _run_one_method(args, m)
        print(f"\n[ALL-METHODS] All {len(METHODS)} methods complete.")
    else:
        _run_one_method(args, args.method)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extract PLIP embeddings for a given patch-selection method"
    )
    method_group = p.add_mutually_exclusive_group()
    method_group.add_argument(
        "--method", choices=METHODS, default=None,
        help="Single patch selection method to run",
    )
    method_group.add_argument(
        "--all-methods", action="store_true",
        help="Run all 6 methods sequentially (ignores --method)",
    )
    p.add_argument(
        "--images-dir", default=DEFAULT_IMAGES_DIR,
        help=f"Directory containing .svs files (default: {DEFAULT_IMAGES_DIR})",
    )
    p.add_argument(
        "--labels-json", default=None,
        help="Optional JSON mapping case_id → label (auto-parsed from filenames if omitted)",
    )
    p.add_argument(
        "--out-dir", default=None,
        help=f"Output root directory (default: {DEFAULT_OUT_ROOT}/{{method}})",
    )
    p.add_argument(
        "--budget", type=float, default=0.8,
        help="HUMBE budget ratio (default: 0.8)",
    )
    p.add_argument(
        "--max-depth", type=int, default=6,
        help="Maximum zoom depth for greedy / a2c (default: 6)",
    )
    p.add_argument(
        "--score", default="text_align_score",
        help="Patch score module key (default: text_align_score)",
    )
    p.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic policy sampling (default: deterministic)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing .pt files",
    )
    args = p.parse_args()
    main(args)
