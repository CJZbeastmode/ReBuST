"""
Test script to load HUMBE-preprocessed results and visualize them.

Directly loads the WSI from the hardcoded image path, restores active_patches
from the HUMBE .pt file, and calls wsi.visualize().
"""

import sys
import os
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from src.utils.wsi import WSI
from src.utils.embedder import Embedder


# Hard-coded paths
CASE        = "TCGA-05-4245-LUAD"
IMAGE_PATH  = f"/Volumes/Xbox_HD/Data/med_img/{CASE}.svs"
HUMBE_PATH  = f"/Volumes/Xbox_HD/Data/humbe/{CASE}.pt"
OUTPUT_PATH = "data/test_humbe_viz.html"


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

    # Visualize — thumbnail is read live from self.slide so it's always present.
    # visualize() uses self.max_level as the thumbnail level and passes it directly
    # to OpenSlide, which only knows about native level indices. Synthetic levels
    # (which make up max_level) are not in OpenSlide, producing a blank background.
    # Workaround: temporarily point max_level at the highest unfrozen native level.
    # Coordinate mapping still works because visualize() rescales via levels_info ds[].
    native_levels = sorted(
        [lvl for lvl, info in wsi.levels_info.items()
         if info.get("type") == "native" and not info.get("frozen", False)],
        reverse=True,
    )
    if not native_levels:
        print("[WARN] No native levels found; background may be blank.")
        thumbnail_level = wsi.max_level
    else:
        thumbnail_level = native_levels[0]

    original_max_level = wsi.max_level
    wsi.max_level = thumbnail_level
    print(f"[VIZ] Generating visualization (thumbnail from native level {thumbnail_level})...")
    wsi.visualize(
        output_html=OUTPUT_PATH,
        metadata={"Case": CASE, "Budget": str(data.get("budget_ratio", "?")), "Patches": str(data["patch_count"])},
    )
    wsi.max_level = original_max_level
    print(f"[DONE] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
