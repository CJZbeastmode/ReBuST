"""
HUMBE Preprocessing Pipeline

Preprocess all WSIs with HUMBE patch selection and save the complete WSI state
for later reuse (loaded as a WSI object with HUMBE pre-applied).

Each processed WSI will be saved to:
    /Volumes/Xbox_HD/Data/humbe/{case_id}.pt

The .pt file contains all serializable WSI state:
  - active_patches:  dict of (level, x, y) surviving HUMBE budget constraint
  - levels_info:     dict with image hierarchy
  - case_id:         str
  - label:           str
  - multistage:      bool (always False for preprocessing)
  - patch_size:      int
  - budget_ratio:    float (for reference)

Load with: wsi = load_humbe_preprocessed(pt_path, images_dir)

Usage:
------
python src/downstream_task/humbe_preprocess_pipeline.py \\
    --images-dir /Volumes/Xbox_HD/Data/med_img \\
    --output-dir /Volumes/Xbox_HD/Data/humbe \\
    --budget 0.8 \\
    --score text_align_score
"""

import sys
import argparse
import os
from pathlib import Path

# Make sure repo root is importable when run directly.
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from tqdm import tqdm

from src.utils.wsi import WSI
from src.utils.embedder import Embedder
from src.global_budget_enforcer.HUMBE import humbe
from src.utils.patch_scores import PATCH_SCORE_MODULES


DEFAULT_IMAGES_DIR = "/Volumes/Xbox_HD/Data/med_img"
DEFAULT_OUTPUT_DIR = "/Volumes/Xbox_HD/Data/humbe_extracted"


def parse_label_from_stem(stem: str) -> str:
    """Extract the cancer-type label from a TCGA filename stem."""
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
    wsi = WSI(img_path, multistage=False, embedder=embedder)
    
    # Restore the HUMBE-filtered active_patches
    wsi.active_patches = data["active_patches"]
    
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

    print(
        f"[DISCOVER] Found {len(cases)} cases in {images_dir}"
    )

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
            # Load WSI
            wsi = WSI(img_path, multistage=False, embedder=embedder)

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
                    "levels_info": wsi.levels_info,
                    "patch_size": wsi.patch_size,
                    "multistage": False,  # always False for this pipeline
                    "patch_count": len(wsi.active_patches),
                    "budget_ratio": args.budget,
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

    print(
        f"\n[DONE] success={success}  skipped={skipped}  failed={failed}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Preprocess all WSIs with HUMBE and save as pickled objects"
    )
    p.add_argument(
        "--images-dir", default=DEFAULT_IMAGES_DIR,
        help=f"Directory containing .svs files (default: {DEFAULT_IMAGES_DIR})",
    )
    p.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for pickled WSI objects (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--budget", type=float, default=0.8,
        help="HUMBE budget ratio (default: 0.8)",
    )
    p.add_argument(
        "--score", default="text_align_score",
        help="Patch score module key (default: text_align_score)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing .pt files",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print full traceback on errors",
    )
    args = p.parse_args()
    main(args)
