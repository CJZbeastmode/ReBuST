"""Export patch counts from .pt files to a sorted JSON report."""

import argparse
import json
from pathlib import Path

import torch


def _count_from_embeddings(embeddings) -> int:
    if embeddings is None:
        return -1
    if not torch.is_tensor(embeddings):
        embeddings = torch.tensor(embeddings)
    if embeddings.dim() == 0:
        return 1
    if embeddings.dim() == 1:
        return 1
    return int(embeddings.shape[0])


def _count_from_active_patches(active_patches) -> int:
    if active_patches is None:
        return -1
    if isinstance(active_patches, dict):
        return len(active_patches)
    if isinstance(active_patches, list):
        return len(active_patches)
    return -1


def get_patch_count(pt_path: Path) -> tuple[int, str]:
    try:
        loaded = torch.load(pt_path, map_location="cpu")
    except Exception as exc:
        return -1, f"load_error: {exc}"

    if torch.is_tensor(loaded):
        return _count_from_embeddings(loaded), "ok_tensor"

    if not isinstance(loaded, dict):
        return -1, "unsupported_schema"

    emb_count = _count_from_embeddings(loaded.get("embeddings"))
    if emb_count >= 0:
        return emb_count, "ok_embeddings"

    active_count = _count_from_active_patches(loaded.get("active_patches"))
    if active_count >= 0:
        return active_count, "ok_active_patches"

    return -1, "missing_embeddings_and_active_patches"


def collect_patch_counts(root: Path, recursive: bool) -> list[dict]:
    pattern = "**/*.pt" if recursive else "*.pt"
    rows = []

    for pt_path in sorted(root.glob(pattern)):
        if pt_path.name.startswith(".") or pt_path.name.startswith("._"):
            continue

        patch_count, status = get_patch_count(pt_path)
        rows.append(
            {
                "file": str(pt_path),
                "relative_file": str(pt_path.relative_to(root)),
                "case_id": pt_path.stem,
                "patch_count": patch_count,
                "status": status,
            }
        )

    rows.sort(
        key=lambda item: (
            item["patch_count"] < 0,
            item["patch_count"],
            item["relative_file"],
        )
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan .pt files and export patch counts to JSON (ascending)."
    )
    parser.add_argument(
        "--root",
        default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/a2c/test",
        help="Root folder containing .pt files",
    )
    parser.add_argument(
        "--out-json",
        default="data/patch_counts_report.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subfolders for .pt files",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()

    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Invalid root directory: {root}")

    rows = collect_patch_counts(root, args.recursive)

    payload = {
        "root": str(root),
        "recursive": bool(args.recursive),
        "num_files": len(rows),
        "num_errors": sum(1 for row in rows if row["patch_count"] < 0),
        "items": rows,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Scanned files: {payload['num_files']}")
    print(f"Rows with errors: {payload['num_errors']}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
