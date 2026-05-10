"""Module for precompute a2c embeddings."""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import torch

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.embedder import Embedder
from src.utils.wsi import WSI


DEFAULT_INPUT_DIR = "/Volumes/Xbox_HD/Data/extracted/full/train"
DEFAULT_OUTPUT_DIR = "/Volumes/Xbox_HD/Data/extracted_with_embeddings/full/train"


def extract_active_keys(active_patches: object) -> List[Tuple[int, int, int]]:
    keys: List[Tuple[int, int, int]] = []

    if isinstance(active_patches, dict):
        iterable = active_patches.keys()
    elif isinstance(active_patches, list):
        iterable = active_patches
    else:
        iterable = []

    for key in iterable:
        if isinstance(key, tuple) and len(key) == 3:
            lvl, x, y = key
        elif isinstance(key, list) and len(key) == 3:
            lvl, x, y = key
        else:
            continue

        try:
            keys.append((int(lvl), int(x), int(y)))
        except Exception:
            continue

    keys.sort(key=lambda item: (item[0], item[2], item[1]))
    return keys


def extract_embeddings_from_pt(
    loaded: Dict,
    case_id: str,
    embedder: Embedder,
    images_dir: str | None,
    expected_embed_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    img_path = loaded.get("img_path")
    if (not img_path or not os.path.exists(img_path)) and images_dir:
        candidate = os.path.join(images_dir, f"{case_id}.svs")
        if os.path.exists(candidate):
            img_path = candidate

    if not img_path or not os.path.exists(img_path):
        return (
            torch.zeros(1, expected_embed_dim, dtype=torch.float32),
            torch.zeros(1, 3, dtype=torch.float32),
        )

    wsi = WSI(
        img_path,
        multistage=bool(loaded.get("multistage", False)),
        embedder=embedder,
    )
    active_patches = loaded.get("active_patches", {})
    wsi.active_patches = active_patches
    wsi.zoomed_patches = loaded.get("zoomed_patches", {})

    keys = extract_active_keys(active_patches)
    embs: List[torch.Tensor] = []
    coords: List[torch.Tensor] = []

    for lvl, x, y in keys:
        try:
            patch = wsi.get_patch(lvl, x, y)
            emb = wsi.get_emb(patch)
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().float().view(-1)
            else:
                emb = torch.tensor(emb, dtype=torch.float32).view(-1)

            if emb.numel() != expected_embed_dim:
                continue

            embs.append(emb)
            coords.append(
                torch.tensor([float(lvl), float(x), float(y)], dtype=torch.float32)
            )
        except Exception:
            continue

    if not embs:
        return (
            torch.zeros(1, expected_embed_dim, dtype=torch.float32),
            torch.zeros(1, 3, dtype=torch.float32),
        )

    return torch.stack(embs), torch.stack(coords)


def process_file(
    src_path: Path,
    dst_path: Path,
    embedder: Embedder,
    images_dir: str | None,
    expected_embed_dim: int,
    overwrite: bool,
) -> str:
    if dst_path.exists() and not overwrite:
        return "skipped"

    loaded = torch.load(src_path, map_location="cpu")
    if not isinstance(loaded, dict):
        return "invalid"

    case_id = src_path.stem
    embeddings, coords = extract_embeddings_from_pt(
        loaded=loaded,
        case_id=case_id,
        embedder=embedder,
        images_dir=images_dir,
        expected_embed_dim=expected_embed_dim,
    )

    out_payload = {
        "case_id": case_id,
        "label": loaded.get("label"),
        "img_path": loaded.get("img_path"),
        "multistage": loaded.get("multistage", False),
        "active_patches": loaded.get("active_patches", {}),
        "zoomed_patches": loaded.get("zoomed_patches", {}),
        "embeddings": embeddings.detach().cpu(),
        "coords": coords.detach().cpu(),
        "patch_count": int(embeddings.shape[0]),
        "source_pt_path": str(src_path),
    }

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, dst_path)
    return "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute patch embeddings for extracted A2C PT files and store training-ready PT outputs."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Optional fallback directory containing <case_id>.svs",
    )
    parser.add_argument("--expected-embed-dim", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".pt"
        and not p.name.startswith(".")
        and not p.name.startswith("._")
    )

    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {input_dir}")

    embedder = Embedder(img_backend="plip")

    started_at = time.perf_counter()
    counts = {"ok": 0, "skipped": 0, "invalid": 0, "failed": 0}

    print(f"[START] files={len(pt_files)} input={input_dir} output={output_dir}")

    for idx, src_path in enumerate(pt_files, start=1):
        dst_path = output_dir / src_path.name
        file_start = time.perf_counter()
        try:
            status = process_file(
                src_path=src_path,
                dst_path=dst_path,
                embedder=embedder,
                images_dir=args.images_dir,
                expected_embed_dim=args.expected_embed_dim,
                overwrite=args.overwrite,
            )
            counts[status] += 1
        except Exception as exc:
            counts["failed"] += 1
            status = "failed"
            print(f"[FAIL] {src_path.name}: {exc}")

        file_elapsed = time.perf_counter() - file_start
        if idx == 1 or idx % max(1, args.log_interval) == 0 or idx == len(pt_files):
            elapsed = time.perf_counter() - started_at
            print(
                f"[PROGRESS] {idx}/{len(pt_files)} status={status} "
                f"last_file={file_elapsed:.2f}s elapsed={elapsed:.2f}s"
            )

    total_elapsed = time.perf_counter() - started_at
    print(
        f"[DONE] files={len(pt_files)} ok={counts['ok']} skipped={counts['skipped']} "
        f"invalid={counts['invalid']} failed={counts['failed']} total_time={total_elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
