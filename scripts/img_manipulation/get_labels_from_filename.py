"""Module for get labels from filename."""

import os
import json

# ==========================
# CONFIG (EDIT HERE)
# ==========================
DATA_DIR = "/Volumes/Xbox_HD/data/med_img"
FILE_EXT = ".svs"
OUTPUT_JSON = "labels_from_filenames.json"
# ==========================


def main():
    labels = {}
    skipped = 0

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(FILE_EXT):
            continue

        name = fname[: -len(FILE_EXT)]  # remove .svs
        parts = name.split("-")

        # Expect at least: TCGA-XX-YYYY-CATEGORY
        if len(parts) < 4:
            skipped += 1
            continue

        category = parts[-1]  # last token
        case_id = "-".join(parts[:-1])  # everything before

        labels[case_id] = f"TCGA-{category}"

    with open(OUTPUT_JSON, "w") as f:
        json.dump(labels, f, indent=2, sort_keys=True)

    print(f"Wrote {len(labels)} labels to {OUTPUT_JSON}")
    if skipped:
        print(f"Skipped {skipped} files with unexpected format")


if __name__ == "__main__":
    main()
