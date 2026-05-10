"""Module for overwrite a2c train img paths."""

import os
from pathlib import Path

import torch


PT_DIR = "/Volumes/Xbox_HD/Data/extracted/a2c/train"
IMG_DIR = "/Volumes/Xbox_HD/Data/med_img/train"


def main() -> None:
    pt_dir = Path(PT_DIR)
    img_dir = Path(IMG_DIR)

    if not pt_dir.exists() or not pt_dir.is_dir():
        raise FileNotFoundError(f"PT directory not found: {pt_dir}")

    updated = 0
    skipped = 0
    failed = 0

    pt_files = sorted(
        p
        for p in pt_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".pt"
        and not p.name.startswith(".")
        and not p.name.startswith("._")
    )

    for pt_path in pt_files:
        case_id = pt_path.stem
        new_img_path = str(img_dir / f"{case_id}.svs")

        try:
            obj = torch.load(pt_path, map_location="cpu")

            if not isinstance(obj, dict):
                skipped += 1
                continue

            obj["img_path"] = new_img_path
            torch.save(obj, pt_path)
            updated += 1

        except Exception as exc:
            failed += 1
            print(f"[FAIL] {pt_path}: {exc}")

    print(
        f"[DONE] files={len(pt_files)} updated={updated} skipped={skipped} failed={failed}"
    )


if __name__ == "__main__":
    main()
