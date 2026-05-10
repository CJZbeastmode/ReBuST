"""
Test script to load HUMBE-preprocessed results and visualize them.

Directly loads the WSI from the hardcoded image path, restores active_patches
from the HUMBE .pt file, and calls wsi.visualize().
"""

import sys
import os
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from src.utils.wsi import WSI
from src.utils.embedder import Embedder


# Hard-coded paths
CASE = "TCGA-05-4415-LUAD"
IMAGE_PATH = f"/Volumes/Xbox_HD/Data/med_img/{CASE}.svs"

HUMBE_PATH = f"/Volumes/Xbox_HD/Data/humbe_extracted/{CASE}.pt"
OUTPUT_PATH = "data/test_humbe_viz_humbe.html"

# HUMBE_PATH   = f"/Volumes/Xbox_HD/Data/full_pipeline_extracted/{CASE}.pt"
# OUTPUT_PATH = "data/test_humbe_viz_full.html"


def main():
    for path in (IMAGE_PATH, HUMBE_PATH):
        if not os.path.exists(path):
            print(f"[ERROR] Not found: {path}")
            return

    # Load WSI directly from the image file so self.slide is always valid
    print(f"[LOAD] WSI from {IMAGE_PATH}...")
    embedder = Embedder(img_backend="plip")
    wsi = WSI(IMAGE_PATH, multistage=False, embedder=embedder)
    print(f"  ✓ WSI loaded")

    # Load HUMBE result and restore patch selection
    print(f"[LOAD] HUMBE result from {HUMBE_PATH}...")
    data = torch.load(HUMBE_PATH, map_location="cpu")
    wsi.active_patches = data["active_patches"]
    print(f"  ✓ Restored {data['patch_count']} active patches")

    print("[VIZ] Generating visualization...")
    wsi.visualize(
        output_html=OUTPUT_PATH,
        metadata={
            "Case": CASE,
            "Budget": str(data.get("budget_ratio", "?")),
            "Patches": str(data["patch_count"]),
        },
    )
    print(f"[DONE] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
