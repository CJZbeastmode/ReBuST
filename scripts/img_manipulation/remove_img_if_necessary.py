"""Module for remove img if necessary."""

import json
import os

# ==========================
# CONFIG (EDIT HERE)
# ==========================
INPUT_JSON = "scripts/metadata/TCGA-READ.json"  # JSON you pasted
DATA_DIR = "/Volumes/Xbox_HD/data/med_img"
FILE_EXT = ".svs"
CASE_SUFFIX = "-READ"  # appended for KEEP
# ==========================


def main():
    with open(INPUT_JSON, "r") as f:
        decisions = json.load(f)

    removed = 0
    renamed = 0
    missing = 0

    for case_id, action in decisions.items():
        src_path = os.path.join(DATA_DIR, f"{case_id}{FILE_EXT}")

        if not os.path.exists(src_path):
            print(f"[MISSING] {case_id}")
            missing += 1
            continue

        if action == "REMOVE":
            os.remove(src_path)
            print(f"[REMOVE] {case_id}")
            removed += 1

        elif action == "KEEP":
            new_name = f"{case_id}{CASE_SUFFIX}{FILE_EXT}"
            dst_path = os.path.join(DATA_DIR, new_name)

            if os.path.exists(dst_path):
                print(f"[SKIP] already renamed: {new_name}")
                continue

            os.rename(src_path, dst_path)
            print(f"[KEEP] {case_id} → {new_name}")
            renamed += 1

        else:
            print(f"[UNKNOWN ACTION] {case_id}: {action}")

    print("\nSUMMARY")
    print(f"  Renamed (KEEP): {renamed}")
    print(f"  Removed       : {removed}")
    print(f"  Missing       : {missing}")


if __name__ == "__main__":
    main()
