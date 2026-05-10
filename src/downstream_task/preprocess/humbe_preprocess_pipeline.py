"""
HUMBE Preprocessing Pipeline

Preprocess all WSIs with HUMBE patch selection and save complete multistage
state for later A2C-on-top refinement.

Each processed WSI is saved to:
        /Volumes/Xbox_HD/Data/humbe_extracted/{case_id}.pt

Saved schema:
    - active_patches: dict[(level, x, y)] -> metadata
    - zoomed_patches: dict[(level, x, y)] -> metadata
    - levels_info:    dict with pyramid hierarchy
    - case_id:        str
    - label:          str
    - multistage:     bool (True)
    - patch_size:     int
    - patch_count:    int (len(active_patches))
    - zoomed_count:   int (len(zoomed_patches))
    - budget_ratio:   float
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
from src.global_budget_enforcer.HUMBE import humbe
from src.utils.patch_scores import PATCH_SCORE_MODULES


DEFAULT_IMAGES_DIR = "/Volumes/Xbox_HD/Data/med_img/val"
DEFAULT_OUTPUT_DIR = "/Volumes/Xbox_HD/Data/extracted/humbe/val"


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


def load_humbe_preprocessed(humbe_pt_path: str) -> WSI:
    """
    Load a HUMBE-preprocessed result and reconstruct as WSI object.

    Args:
        humbe_pt_path:  Path to the .pt file saved by this pipeline

    Returns:
        WSI object with active_patches restored to HUMBE-filtered state.
    """
    data = torch.load(humbe_pt_path, map_location="cpu")
    img_path = data["img_path"]

    # Reload the original WSI from the saved image path
    embedder = Embedder(img_backend="plip")
    wsi = WSI(
        img_path, multistage=bool(data.get("multistage", True)), embedder=embedder
    )

    # Restore the HUMBE-filtered active/zoomed hierarchy
    wsi.active_patches = data["active_patches"]
    wsi.zoomed_patches = data.get("zoomed_patches", {})

    return wsi


def main(args):
    """Run HUMBE on all WSIs and save processed objects."""
    images_dir = os.path.abspath(args.images_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Discover all cases
    cases = discover_cases(images_dir)
    if not cases:
        print(f"[WARN] No cases found in {images_dir}")
        return

    print(f"[DISCOVER] Found {len(cases)} cases in {images_dir}")

    # Load score module once
    score_module = PATCH_SCORE_MODULES[args.score]()

    # Note: We create embedder only for WSI initialization (not used for HUMBE itself)
    embedder = Embedder(img_backend="plip")

    success, skipped, failed = 0, 0, 0
    case_ids = sorted(cases.keys())

    for case_id in tqdm(case_ids, desc="humbe_preprocess"):
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
            # Load WSI in multistage mode for HUMBE->A2C compatibility
            wsi = WSI(img_path, multistage=True, embedder=embedder)

            # Apply HUMBE
            wsi = humbe(
                wsi,
                score_module=score_module,
                budget_ratio=args.budget,
                verbose=False,
            )

            # Save complete WSI state as .pt file
            torch.save(
                {
                    "case_id": case_id,
                    "label": cases[case_id],
                    "img_path": img_path,  # full path so loading is self-contained
                    "active_patches": wsi.active_patches,  # dict of (level, x, y)
                    "zoomed_patches": wsi.zoomed_patches,
                    "levels_info": wsi.levels_info,
                    "patch_size": wsi.patch_size,
                    "multistage": True,
                    "patch_count": len(wsi.active_patches),
                    "zoomed_count": len(wsi.zoomed_patches),
                    "budget_ratio": args.budget,
                },
                out_path,
            )

            n_patches = len(wsi.active_patches)
            n_zoomed = len(wsi.zoomed_patches)
            print(
                f"  [SAVED] {case_id}: active={n_patches} zoomed={n_zoomed} → {out_path}"
            )
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
        description="Preprocess all WSIs with HUMBE and save as pickled objects"
    )
    p.add_argument(
        "--images-dir",
        default=DEFAULT_IMAGES_DIR,
        help=f"Directory containing .svs files (default: {DEFAULT_IMAGES_DIR})",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for pickled WSI objects (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--budget",
        type=float,
        default=0.8,
        help="HUMBE budget ratio (default: 0.8)",
    )
    p.add_argument(
        "--score",
        default="text_align_score",
        help="Patch score module key (default: text_align_score)",
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
