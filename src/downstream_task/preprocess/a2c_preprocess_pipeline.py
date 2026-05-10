"""
A2C Preprocessing Pipeline (single-stage)

This pipeline iterates all WSIs in ``images_dir`` and, for each case:
    1) constructs a fresh WSI in single-stage mode
    2) runs standalone A2C inference
    3) saves output in HUMBE-compatible serialized format

Default paths:
    - Images dir: /Volumes/Xbox_HD/Data/med_img
    - Output dir: /Volumes/Xbox_HD/Data/a2c_extracted

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
DEFAULT_OUTPUT_DIR = "/Volumes/Xbox_HD/Data/extracted/a2c/val"
DEFAULT_A2C_MODEL = "data/models/rl/a2c/a2c.pt"


def parse_label_from_stem(stem: str) -> str:
    """Extract the cancer-type label from a TCGA filename stem."""
    return stem.rsplit("-", 1)[-1]


def discover_cases(images_dir: str) -> dict:
    """Scan *images_dir* for .svs files and return ``{case_id: label}``."""
    cases = {}
    for fname in sorted(os.listdir(images_dir)):
        if fname.startswith(".") or fname.startswith("._"):
            continue
        if not fname.lower().endswith(".svs"):
            continue
        stem = os.path.splitext(fname)[0]
        cases[stem] = parse_label_from_stem(stem)
    return cases


def main(args):
    """Run single-stage A2C preprocessing for all WSIs."""
    images_dir = os.path.abspath(args.images_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"A2C checkpoint not found: {args.model}")

    # Discover all image cases
    cases = discover_cases(images_dir)
    if not cases:
        print(f"[WARN] No cases found in {images_dir}")
        return

    print(f"[DISCOVER] Found {len(cases)} cases in {images_dir}")

    success, skipped, failed = 0, 0, 0
    case_ids = sorted(cases.keys())

    for case_id in tqdm(case_ids, desc="a2c_preprocess"):
        out_path = os.path.join(output_dir, f"{case_id}.pt")

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

        try:
            embedder = Embedder(img_backend="plip")
            wsi = WSI(img_path, multistage=False, embedder=embedder)

            wsi = infer_wsi_a2c(
                wsi,
                model_path=args.model,
                deterministic=not args.stochastic,
                verbose=args.verbose,
            )

            label_value = cases[case_id]

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
                    "budget_ratio": None,
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
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for A2C-preprocessed .pt files (default: {DEFAULT_OUTPUT_DIR})",
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
