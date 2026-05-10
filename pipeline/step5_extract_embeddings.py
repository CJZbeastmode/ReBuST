"""Step 5: extract direct embeddings from extracted patch-selection PT files.

Reads per-case `.pt` files (with `active_patches`) and writes per-case `.pt`
files that include direct patch embeddings and coordinates.

This script reuses extraction logic from:
    src.streaming_transformer_v1.data.WSIEmbeddingDataset
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

from archive.streaming_transformer_archive.streaming_transformer_v1.data import WSIEmbeddingDataset


def discover_pt_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    files = []
    for fname in sorted(os.listdir(folder)):
        if fname.startswith(".") or fname.startswith("._"):
            continue
        if fname.lower().endswith(".pt"):
            files.append(fname)
    return files


def run_split(
    input_dir: str,
    output_dir: str,
    images_dir: str | None,
    force: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    pt_files = discover_pt_files(input_dir)
    if not pt_files:
        print(f"[WARN] No .pt files found in {input_dir}")
        return

    helper = WSIEmbeddingDataset(items=[], embeddings_dir=input_dir, images_dir=images_dir)
    success, skipped, failed = 0, 0, 0

    for fname in tqdm(pt_files, desc=f"step5/{Path(input_dir).name}"):
        case_id = os.path.splitext(fname)[0]
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        if os.path.exists(out_path) and not force:
            skipped += 1
            continue

        try:
            loaded = torch.load(in_path, map_location="cpu")
            if not isinstance(loaded, dict):
                raise ValueError("Unsupported PT schema: expected dict payload")

            embeddings, coords = helper._extract_from_active_patches(case_id, loaded)

            out_payload = dict(loaded)
            out_payload["embeddings"] = embeddings
            out_payload["coords"] = coords
            out_payload["embedding_dim"] = int(embeddings.shape[-1])
            out_payload["patch_count"] = int(embeddings.shape[0])
            out_payload["source"] = "step5_extract_embeddings"

            torch.save(out_payload, out_path)
            success += 1
        except Exception as exc:
            print(f"[FAIL] {case_id}: {exc}")
            failed += 1

    print(
        f"[DONE] split={Path(input_dir).name} success={success} skipped={skipped} failed={failed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract direct embeddings from extracted PTs into extracted_with_embeddings PTs"
    )
    parser.add_argument(
        "--input-root",
        default="/Volumes/Xbox_HD/Data/extracted/a2c",
        help="Root with train/val/test extracted .pt files",
    )
    parser.add_argument(
        "--output-root",
        default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/a2c",
        help="Root for output .pt files with embeddings",
    )
    parser.add_argument(
        "--images-root",
        default="/Volumes/Xbox_HD/Data/med_img",
        help="Optional images root (train/val/test) used when img_path is missing",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated split names",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite outputs")
    args = parser.parse_args()

    input_root = os.path.abspath(args.input_root)
    output_root = os.path.abspath(args.output_root)
    images_root = os.path.abspath(args.images_root) if args.images_root else None
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        split_input = os.path.join(input_root, split)
        split_output = os.path.join(output_root, split)
        split_images = os.path.join(images_root, split) if images_root else None

        if not os.path.isdir(split_input):
            print(f"[WARN] Missing input split dir: {split_input} (skipping)")
            continue

        print(
            f"\n[STEP5] split={split} input={split_input} output={split_output} images={split_images}"
        )
        run_split(
            input_dir=split_input,
            output_dir=split_output,
            images_dir=split_images,
            force=args.force,
        )


if __name__ == "__main__":
    main()
