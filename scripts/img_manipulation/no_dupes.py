"""Module for no dupes."""

import json

# ==========================
# CONFIG (EDIT HERE)
# ==========================
JSON_A = "data/labels_from_filenames.json"
JSON_B = "data/to_fetch.json"
# ==========================


def main():
    with open(JSON_A, "r") as f:
        labels_a = json.load(f)

    with open(JSON_B, "r") as f:
        labels_b = json.load(f)

    keys_a = set(labels_a.keys())
    keys_b = set(labels_b.keys())

    # --------------------------
    # Duplicate case IDs
    # --------------------------
    dupes = keys_a & keys_b

    # --------------------------
    # Conflicting labels
    # --------------------------
    conflicts = {
        k: (labels_a[k], labels_b[k]) for k in dupes if labels_a[k] != labels_b[k]
    }

    print("SUMMARY")
    print(f"Cases in {JSON_A}: {len(keys_a)}")
    print(f"Cases in {JSON_B}: {len(keys_b)}")
    print(f"Duplicate case IDs: {len(dupes)}")
    print(f"Conflicting labels: {len(conflicts)}")

    # --------------------------
    # Detailed reporting
    # --------------------------
    if dupes:
        print("\nDUPLICATE CASE IDs:")
        for k in sorted(dupes):
            print(f"  {k}: {labels_a[k]} | {labels_b[k]}")

    if conflicts:
        print("\nCONFLICTING LABELS:")
        for k, (a, b) in conflicts.items():
            print(f"  {k}: {a} ≠ {b}")

    # --------------------------
    # Hard failure if any dupes
    # --------------------------
    if dupes:
        raise SystemExit(
            "\nERROR: Duplicate case IDs detected. " "Resolve before proceeding."
        )

    print("\nOK: No duplicates, no conflicts.")


if __name__ == "__main__":
    main()
