"""Module for count img categories."""

import os
import json
from collections import defaultdict

# ==========================
# CONFIG (EDIT HERE)
# ==========================
DATA_DIR = "/Volumes/Xbox_HD/data/med_img/val"
FILE_EXT = ".svs"
OUTPUT_JSON = "image_counts_per_category.json"
# ==========================


def main():
    counts = defaultdict(int)
    skipped = 0

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(FILE_EXT):
            continue

        # Remove extension
        name_no_ext = fname[: -len(FILE_EXT)]

        # Split by "-"
        parts = name_no_ext.split("-")

        # We expect: TCGA-XX-YYYY-CATEGORY
        if len(parts) < 4:
            skipped += 1
            continue

        category = parts[-1]  # last suffix
        counts[category] += 1

    # Write JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(dict(counts), f, indent=2, sort_keys=True)

    print("Image counts per category:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")

    print(f"\nWritten to: {OUTPUT_JSON}")
    if skipped > 0:
        print(f"Skipped files (unexpected format): {skipped}")


if __name__ == "__main__":
    main()
