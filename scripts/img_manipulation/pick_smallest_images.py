"""Module for pick smallest images."""

import json
import os
from collections import defaultdict

"""
TCGA-CHOL: 1
TCGA-COAD: 10
TCGA-ESCA: 4
TCGA-LIHC: 9
TCGA-LUAD: 12
TCGA-LUSC: 11
TCGA-MESO: 2
TCGA-PAAD: 4
TCGA-READ: 4
TCGA-SKCM: 11
TCGA-STAD: 10
TCGA-UVM: 2
"""

# ==========================
# CONFIG (EDIT HERE)
# ==========================
CATEGORY = "TCGA-UVM"
INPUT_JSON = "data/labels_main.json"  # case_id -> TCGA project
DATA_DIR = "/Volumes/Xbox_HD/data/med_img"  # directory containing .svs files
OUTPUT_JSON = f"{CATEGORY}.json"  # output file
FILE_EXT = ".svs"
N_PER_CLASS = 2  # keep N smallest per category
# ==========================


def main():
    # Load mapping
    with open(INPUT_JSON, "r") as f:
        case_to_project = json.load(f)

    # Group files by project
    by_project = defaultdict(list)
    missing_files = []

    for case_id, project in case_to_project.items():
        if project != CATEGORY:
            continue

        project_suffix = project.replace("TCGA-", "")
        path = os.path.join(DATA_DIR, f"{case_id}-{project_suffix}{FILE_EXT}")

        print(path)

        if not os.path.exists(path):
            missing_files.append(case_id)
            continue

        size = os.path.getsize(path)
        by_project[project].append((case_id, path, size))

    # Decide KEEP / REMOVE
    keep_remove = {}

    for project, entries in by_project.items():
        # Sort by file size (ascending)
        entries.sort(key=lambda x: x[2])

        keep = entries[:N_PER_CLASS]
        remove = entries[N_PER_CLASS:]

        for case_id, _, _ in keep:
            keep_remove[case_id] = "KEEP"

        for case_id, _, _ in remove:
            keep_remove[case_id] = "REMOVE"

        print(f"{project}: " f"KEEP {len(keep)} / REMOVE {len(remove)}")

    # Mark missing files explicitly
    for case_id in missing_files:
        keep_remove[case_id] = "MISSING"

    # Write output JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(keep_remove, f, indent=2, sort_keys=True)

    print(f"\nOutput written to: {OUTPUT_JSON}")
    print(f"Missing files: {len(missing_files)}")


if __name__ == "__main__":
    main()
