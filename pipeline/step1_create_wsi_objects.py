"""Step 1: create initial WSI state .pt files for train/val/test splits.

This script reads `.svs` files split by directory and writes one `.pt` file per
case containing a serializable snapshot of a freshly constructed `WSI` object.

Expected input structure:
    <images-root>/train/*.svs
    <images-root>/val/*.svs
    <images-root>/test/*.svs

Output structure:
    <output-root>/train/<case_id>.pt
    <output-root>/val/<case_id>.pt
    <output-root>/test/<case_id>.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.embedder import Embedder
from src.utils.wsi import WSI


def parse_label_from_stem(stem: str) -> str:
    return stem.rsplit("-", 1)[-1]


def discover_svs(images_dir: str) -> list[str]:
    if not os.path.isdir(images_dir):
        return []
    files = []
    for fname in sorted(os.listdir(images_dir)):
        if fname.startswith(".") or fname.startswith("._"):
            continue
        if fname.lower().endswith(".svs"):
            files.append(fname)
    return files


def save_wsi_snapshot(wsi: WSI, case_id: str, img_path: str, out_path: str) -> None:
    payload = {
        "case_id": case_id,
        "label": parse_label_from_stem(case_id),
        "img_path": img_path,
        "active_patches": wsi.active_patches,
        "zoomed_patches": wsi.zoomed_patches,
        "levels_info": wsi.levels_info,
        "patch_size": wsi.patch_size,
        "multistage": wsi.multistage,
        "min_level": wsi.min_level,
        "max_level": wsi.max_level,
        "patch_count": len(wsi.active_patches),
        "zoomed_count": len(wsi.zoomed_patches),
        "source": "step1_create_wsi_objects",
    }
    torch.save(payload, out_path)


def run_split(images_dir: str, out_dir: str, multistage: bool, force: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)
    svs_files = discover_svs(images_dir)
    if not svs_files:
        print(f"[WARN] No .svs files found in {images_dir}")
        return

    embedder = Embedder(img_backend="plip")
    success, skipped, failed = 0, 0, 0

    for fname in tqdm(svs_files, desc=f"step1/{Path(images_dir).name}"):
        case_id = os.path.splitext(fname)[0]
        img_path = os.path.join(images_dir, fname)
        out_path = os.path.join(out_dir, f"{case_id}.pt")

        if os.path.exists(out_path) and not force:
            skipped += 1
            continue

        try:
            wsi = WSI(img_path, multistage=multistage, embedder=embedder)
            save_wsi_snapshot(
                wsi, case_id=case_id, img_path=img_path, out_path=out_path
            )
            success += 1
        except Exception as exc:
            print(f"[FAIL] {case_id}: {exc}")
            failed += 1

    print(
        f"[DONE] split={Path(images_dir).name} success={success} skipped={skipped} failed={failed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create initial WSI-state .pt files for train/val/test"
    )
    parser.add_argument(
        "--images-root",
        default="/Volumes/Xbox_HD/Data/med_img",
        help="Root with train/val/test subfolders containing .svs files",
    )
    parser.add_argument(
        "--output-root",
        default="/Volumes/Xbox_HD/Data/wsi_objects",
        help="Root output folder where split subfolders will be created",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated split names",
    )
    parser.add_argument(
        "--single-stage",
        action="store_true",
        help="Use single-stage WSI mode (default: multistage)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    args = parser.parse_args()

    images_root = os.path.abspath(args.images_root)
    output_root = os.path.abspath(args.output_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    multistage = not args.single_stage

    for split in splits:
        split_images = os.path.join(images_root, split)
        split_out = os.path.join(output_root, split)
        print(f"\n[STEP1] split={split} images={split_images} out={split_out}")
        run_split(
            images_dir=split_images,
            out_dir=split_out,
            multistage=multistage,
            force=args.force,
        )


if __name__ == "__main__":
    main()
