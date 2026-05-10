"""
A2C Preprocessing Pipeline (HUMBE baseline -> A2C refinement)

This pipeline iterates all HUMBE baseline files in ``humbe_dir`` and, for each case:
    1) loads HUMBE-preprocessed baseline from ``humbe_dir/{case_id}.pt``
    2) reconstructs WSI from the original image
    3) restores HUMBE ``active_patches`` as baseline state
    4) runs A2C inference refinement
    5) saves output in the same serialized format as HUMBE preprocessing

Default paths:
    - HUMBE baseline dir: /Volumes/Xbox_HD/Data/humbe_extracted
    - Images dir:         /Volumes/Xbox_HD/Data/med_img
    - Output dir:         /Volumes/Xbox_HD/Data/full_pipeline_extracted

Usage:
------
python src/downstream_task/preprocess/a2c_preprocess_pipeline.py \
        --model /path/to/a2c_checkpoint.pt
"""

import sys
import argparse
import os
from pathlib import Path

# Make sure repo root is importable when run directly.
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from tqdm import tqdm

from src.utils.wsi import WSI
from src.utils.embedder import Embedder
from src.inference.rl.a2c.infer_rl_a2c import infer_wsi_a2c


DEFAULT_IMAGES_DIR = "/Volumes/Xbox_HD/Data/med_img/val"
DEFAULT_HUMBE_DIR = "/Volumes/Xbox_HD/Data/extracted/humbe/val"
DEFAULT_OUTPUT_DIR = "/Volumes/Xbox_HD/Data/extracted/full/val"
DEFAULT_A2C_MODEL = "data/models/rl/a2c/a2c.pt"


def parse_label_from_stem(stem: str) -> str:
    """Extract the cancer-type label from a TCGA filename stem."""
    return stem.rsplit("-", 1)[-1]


def discover_humbe_cases(humbe_dir: str) -> list[str]:
    """Scan *humbe_dir* for .pt files and return sorted case IDs."""
    case_ids = []
    for fname in sorted(os.listdir(humbe_dir)):
        if fname.startswith(".") or fname.startswith("._"):
            continue
        if not fname.lower().endswith(".pt"):
            continue
        case_ids.append(os.path.splitext(fname)[0])
    return case_ids


def load_humbe_baseline_to_wsi(humbe_pt_path: str, img_path: str) -> WSI:
    """
    Load a HUMBE-preprocessed result and reconstruct a WSI baseline.

    Args:
        humbe_pt_path: Path to HUMBE baseline .pt
        img_path:      Absolute path to original .svs image

    Returns:
        WSI object with active_patches restored to HUMBE-filtered state.
    """
    data = torch.load(humbe_pt_path, map_location="cpu")

    embedder = Embedder(img_backend="plip")
    wsi = WSI(
        img_path, multistage=bool(data.get("multistage", True)), embedder=embedder
    )

    wsi.active_patches = data["active_patches"]
    wsi.zoomed_patches = data.get("zoomed_patches", {})

    return wsi


def main(args):
    """Run A2C refinement on top of HUMBE baseline for all WSIs."""
    images_dir = os.path.abspath(args.images_dir)
    humbe_dir = os.path.abspath(args.humbe_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"A2C checkpoint not found: {args.model}")

    # Discover all HUMBE baselines (source of truth)
    case_ids = discover_humbe_cases(humbe_dir)
    if not case_ids:
        print(f"[WARN] No baseline .pt files found in {humbe_dir}")
        return

    print(f"[DISCOVER] Found {len(case_ids)} baseline files in {humbe_dir}")

    success, skipped, failed = 0, 0, 0

    for case_id in tqdm(case_ids, desc="a2c_preprocess"):
        out_path = os.path.join(output_dir, f"{case_id}.pt")
        humbe_path = os.path.join(humbe_dir, f"{case_id}.pt")

        # Skip if already processed
        if os.path.exists(out_path) and not args.force:
            skipped += 1
            print(f"Skipping (exists): {case_id}")
            continue

        print(f"Processing {case_id}...")

        img_path = os.path.join(images_dir, f"{case_id}.svs")
        if not os.path.exists(img_path):
            print(f"  [MISSING] {img_path}")
            failed += 1
            continue

        if not os.path.exists(humbe_path):
            print(f"  [MISSING BASELINE] {humbe_path}")
            failed += 1
            continue

        try:
            humbe_data = torch.load(humbe_path, map_location="cpu")

            wsi = load_humbe_baseline_to_wsi(
                humbe_pt_path=humbe_path,
                img_path=img_path,
            )

            wsi = infer_wsi_a2c(
                wsi,
                model_path=args.model,
                deterministic=not args.stochastic,
                verbose=args.verbose,
            )

            budget_ratio = humbe_data.get("budget_ratio", None)
            label_value = humbe_data.get("label", parse_label_from_stem(case_id))

            torch.save(
                {
                    "case_id": case_id,
                    "label": label_value,
                    "img_path": img_path,
                    "active_patches": wsi.active_patches,
                    "zoomed_patches": wsi.zoomed_patches,
                    "levels_info": wsi.levels_info,
                    "patch_size": wsi.patch_size,
                    "multistage": wsi.multistage,
                    "patch_count": len(wsi.active_patches),
                    "zoomed_count": len(wsi.zoomed_patches),
                    "budget_ratio": budget_ratio,
                },
                out_path,
            )

            n_patches = len(wsi.active_patches)
            print(f"  [SAVED] {case_id}: {n_patches} patches → {out_path}")
            success += 1

        except Exception as e:
            import traceback

            print(f"  [FAIL] {case_id}: {e}")
            if args.verbose:
                traceback.print_exc()
            failed += 1

    print(f"\n[DONE] success={success}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Preprocess all WSIs with HUMBE baseline + A2C refinement"
    )
    p.add_argument(
        "--images-dir",
        default=DEFAULT_IMAGES_DIR,
        help=f"Directory containing .svs files (default: {DEFAULT_IMAGES_DIR})",
    )
    p.add_argument(
        "--humbe-dir",
        default=DEFAULT_HUMBE_DIR,
        help=f"Directory containing HUMBE baseline .pt files (default: {DEFAULT_HUMBE_DIR})",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for A2C-refined .pt files (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_A2C_MODEL,
        help=f"A2C checkpoint path (default: {DEFAULT_A2C_MODEL})",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling (default: deterministic argmax)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .pt files",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print full traceback on errors",
    )
    args = p.parse_args()
    main(args)
