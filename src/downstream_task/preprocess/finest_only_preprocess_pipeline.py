"""Extract only finest-level patches for all slides in a directory."""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.wsi import WSI
from src.utils.embedder import Embedder

DEFAULT_IMAGES_DIR = "/Volumes/Xbox_HD/Data/med_img/train"
DEFAULT_OUTPUT_DIR = "/Volumes/Xbox_HD/Data/extracted/finest_only/train"


def parse_label_from_stem(stem: str) -> str:
    return stem.rsplit("-", 1)[-1]


def discover_cases(images_dir: str) -> dict:
    cases = {}
    for fname in sorted(os.listdir(images_dir)):
        if fname.startswith(".") or fname.startswith("._"):
            continue
        if not fname.lower().endswith(".svs"):
            continue
        stem = os.path.splitext(fname)[0]
        cases[stem] = parse_label_from_stem(stem)
    return cases


def main(args) -> None:
    images_dir = os.path.abspath(args.images_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cases = discover_cases(images_dir)
    if not cases:
        print(f"[WARN] No .svs files found in {images_dir}")
        return

    embedder = Embedder(img_backend="plip")
    success = skipped = failed = 0

    for case_id in tqdm(sorted(cases.keys()), desc="finest_only_preprocess"):
        out_path = os.path.join(output_dir, f"{case_id}.pt")
        if os.path.exists(out_path) and not args.force:
            skipped += 1
            continue

        img_path = os.path.join(images_dir, f"{case_id}.svs")
        if not os.path.exists(img_path):
            failed += 1
            continue

        try:
            wsi = WSI(img_path, embedder=embedder)
            finest_level = wsi.min_level
            active_patches = {
                (finest_level, int(x), int(y)): {}
                for x, y in wsi.iterate_patches(finest_level)
            }

            torch.save(
                {
                    "case_id": case_id,
                    "label": cases[case_id],
                    "img_path": img_path,
                    "active_patches": active_patches,
                    "zoomed_patches": {},
                    "levels_info": wsi.levels_info,
                    "patch_size": wsi.patch_size,
                    "multistage": wsi.multistage,
                    "patch_count": len(active_patches),
                    "zoomed_count": 0,
                    "method": "finest_only",
                },
                out_path,
            )
            success += 1
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {case_id}: {exc}")

    print(f"[DONE] success={success} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract finest-level patches only")
    p.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    main(args)
