"""Pre-extract PLIP embeddings for a folder of WSIs.

Saves one file per input WSI under `out_dir` (default: `data/embeddings/plip`).

Usage:
    python src/downstream_tasks/wsi_classification/preextract_plip_embeddings.py \
        --images-dir data/images --out-dir data/embeddings/plip --force

The saved file is a PyTorch checkpoint containing:
  {
    'embeddings': Tensor [N, 512],
    'levels': list_of_levels_processed,
    'n_patches_per_level': {level: count}
  }

This script uses the project's `WSI` and `Embedder` implementations.
"""

import os
import sys
from pathlib import Path
import argparse
import torch
from tqdm import tqdm

# Ensure repo root on sys.path when running directly
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.wsi import WSI
from src.utils.embedder import Embedder


def extract_embeddings_for_wsi(image_path: str, skip_level_thresh: int = 40000):
    """Extract PLIP embeddings for one WSI and return a dict for saving.

    Skips any pyramid level whose width or height exceeds `skip_level_thresh`.
    """
    embedder = Embedder(img_backend="plip")
    wsi = WSI(image_path=image_path, embedder=embedder, multistage=True)

    embs = []
    levels_processed = []
    n_patches_per_level = {}

    for lvl_id in sorted(wsi.levels_info.keys()):
        # per-level guard
        try:
            lvl_w, lvl_h = wsi.levels_info[lvl_id]["size"]
            if lvl_w > skip_level_thresh or lvl_h > skip_level_thresh:
                print(
                    f"Skipping level {lvl_id} for {image_path}: {lvl_w}x{lvl_h} > {skip_level_thresh}"
                )
                continue
        except Exception:
            pass

        count = 0
        for x, y in wsi.iterate_patches(lvl_id):
            try:
                patch = wsi.get_patch(lvl_id, x, y)
                emb = wsi.get_emb(patch)
                # Ensure CPU tensor
                if hasattr(emb, "cpu"):
                    emb = emb.cpu()
                embs.append(emb)
                count += 1
            except Exception:
                continue

        if count > 0:
            levels_processed.append(lvl_id)
            n_patches_per_level[lvl_id] = count

    if len(embs) == 0:
        embeddings = torch.zeros(1, 512)
    else:
        embeddings = torch.stack(embs)

    return {
        "embeddings": embeddings,
        "levels": levels_processed,
        "n_patches_per_level": n_patches_per_level,
    }


def main(
    images_dir: str, out_dir: str, force: bool = False, skip_level_thresh: int = 40000
):
    images_dir = os.path.abspath(images_dir)
    out_dir = os.path.abspath(out_dir)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    os.makedirs(out_dir, exist_ok=True)

    svs_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".svs")]

    for fname in tqdm(sorted(svs_files), desc="Images"):
        case_id = os.path.splitext(fname)[0]
        img_path = os.path.join(images_dir, fname)
        out_path = os.path.join(out_dir, f"{case_id}.pt")

        if os.path.exists(out_path) and not force:
            print(f"Skipping (exists): {case_id}")
            continue
        
        print(f"Processing {case_id}...")
        try:
            data = extract_embeddings_for_wsi(
                img_path, skip_level_thresh=skip_level_thresh
            )
            # Save using torch so tensors are preserved
            torch.save(data, out_path)
            print(
                f"Saved embeddings → {out_path} | patches_total={data['embeddings'].shape[0]}"
            )
        except Exception as e:
            print(f"Failed to extract {img_path}: {e}")
            continue


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pre-extract PLIP embeddings for WSIs")
    p.add_argument(
        "--images-dir", default="/Volumes/Xbox_HD/data/med_img", help="Directory with .svs images"
    )
    p.add_argument(
        "--out-dir",
        default="/Volumes/Xbox_HD/data/humbe_extracted",
        help="Directory to write per-image embeddings",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing embeddings")
    p.add_argument(
        "--skip-level-thresh",
        type=int,
        default=40000,
        help="Per-level side threshold to skip levels",
    )
    args = p.parse_args()

    main(
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        force=args.force,
        skip_level_thresh=args.skip_level_thresh,
    )
