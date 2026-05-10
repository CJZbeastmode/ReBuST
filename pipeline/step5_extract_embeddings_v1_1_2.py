"""Step 5 (v1.1.2): extract embeddings using optional fine-tuned embedder."""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.streaming_transformer.data import WSIEmbeddingDataset


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
    embedder_backend: str,
    embedder_ckpt: str | None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    pt_files = discover_pt_files(input_dir)
    if not pt_files:
        print(f"[WARN] No .pt files found in {input_dir}")
        return

    helper = WSIEmbeddingDataset(
        items=[],
        embeddings_dir=input_dir,
        images_dir=images_dir,
        embedder_backend=embedder_backend,
        embedder_ckpt=embedder_ckpt,
    )
    success, skipped, failed = 0, 0, 0

    for fname in tqdm(pt_files, desc=f"step5_v1_1_2/{Path(input_dir).name}"):
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
            out_payload["source"] = "step5_extract_embeddings_v1_1_2"

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
        description="Extract embeddings using fine-tuned embedder (v1.1.2)"
    )
    parser.add_argument(
        "--input-root",
        default="/Volumes/Xbox_HD/Data/extracted/a2c",
        help="Root with train/val/test extracted .pt files",
    )
    parser.add_argument(
        "--output-root",
        default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/a2c_finetuned",
        help="Root for output .pt files with embeddings",
    )
    parser.add_argument(
        "--images-root",
        default=None,
        help="Optional root for .svs images (train/val/test)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing embedding files",
    )
    parser.add_argument("--embedder-backend", default="plip")
    parser.add_argument(
        "--embedder-ckpt",
        default=None,
        help="Fine-tuned embedder checkpoint path",
    )
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        in_dir = os.path.join(args.input_root, split)
        out_dir = os.path.join(args.output_root, split)
        images_dir = None
        if args.images_root:
            images_dir = os.path.join(args.images_root, split)
        run_split(
            in_dir,
            out_dir,
            images_dir,
            args.force,
            args.embedder_backend,
            args.embedder_ckpt,
        )


if __name__ == "__main__":
    main()
