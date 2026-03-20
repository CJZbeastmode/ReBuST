import os
import json
from typing import Dict, Optional
import requests
import time


GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"


def tcga_label_from_filename(svs_path):
    """
    Given a filename like:
        tcga_wsis/TCGA-3H-AB3L.svs

    Queries the GDC API and returns:
        project_id (e.g. 'TCGA-COAD')

    This is an ONLINE lookup. Requires internet.
    """

    # 1. Extract case submitter ID from filename
    case_id = os.path.splitext(os.path.basename(svs_path))[0]

    # 2. Query GDC for this case
    payload = {
        "filters": {
            "op": "=",
            "content": {
                "field": "submitter_id",
                "value": case_id,
            },
        },
        "fields": "project.project_id",
        "format": "JSON",
        "size": 1,
    }

    r = requests.post(GDC_CASES_ENDPOINT, json=payload, timeout=20)
    r.raise_for_status()
    hits = r.json()["data"]["hits"]

    if len(hits) == 0:
        return None

    project_id = hits[0]["project"]["project_id"]

    return project_id


def process_images(
    images_dir: str = "data/images",
    case_to_project: Optional[Dict[str, str]] = None,
    out_file: Optional[str] = None,
):
    """
    Iterate all .svs images in `images_dir`, derive `case_id` from filename
    (basename without extension) and call `tcga_project_label_from_case` for
    each image. If `case_to_project` is None, the function will only print
    discovered case IDs.
    """
    images_dir = os.path.abspath(images_dir)
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    print(f"Processing images in: {images_dir}")
    print(f"Images count: {len(os.listdir(images_dir))}")
    svs_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".svs")]

    results = {}
    i = 0
    t0 = time.time()
    for fname in sorted(svs_files):
        case_id = os.path.splitext(fname)[0]
        if case_to_project is not None:
            # Use provided mapping
            label = case_to_project.get(case_id)
        else:
            # Fallback to online lookup (may fail if offline)
            try:
                full_path = os.path.join(images_dir, fname)
                label = tcga_label_from_filename(full_path)
            except Exception as e:
                label = None

        results[case_id] = label
        # print(f"{case_id}: {label}")
        i += 1

        t1 = time.time()
        dt = t1 - t0
        print(
            f"Processed {i}/{len(svs_files)} images in {dt:.1f} sec ({dt/i:.2f} sec/image)"
        )

        if i % 10 == 0:
            print("Backup write to output file...")
            if out_file:
                out_file = os.path.abspath(out_file)
                out_dir = os.path.dirname(out_file)
                try:
                    os.makedirs(out_dir, exist_ok=True)
                    # Best-effort delete existing file before writing to avoid partial overwrites
                    if os.path.exists(out_file):
                        try:
                            os.remove(out_file)
                        except Exception as del_err:
                            print(
                                f"Could not delete existing file {out_file}: {del_err}"
                            )

                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"Wrote mapping to: {out_file}")
                except Exception as e:
                    print(f"Failed to write mapping to {out_file}: {e}")

    if out_file:
        out_file = os.path.abspath(out_file)
        out_dir = os.path.dirname(out_file)
        try:
            os.makedirs(out_dir, exist_ok=True)
            # Best-effort delete existing file before writing to avoid partial overwrites
            if os.path.exists(out_file):
                try:
                    os.remove(out_file)
                except Exception as del_err:
                    print(f"Could not delete existing file {out_file}: {del_err}")

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Wrote mapping to: {out_file}")
        except Exception as e:
            print(f"Failed to write mapping to {out_file}: {e}")

    return results


def _load_mapping(json_path: str) -> Dict[str, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Map TCGA case IDs for images in data/images/"
    )
    p.add_argument(
        "--images-dir",
        default="/Volumes/Xbox_HD/data/temp",
        help="Directory with .svs images",
    )
    p.add_argument(
        "--mapping-json",
        default=None,
        help="Optional JSON file mapping case_id -> project",
    )
    p.add_argument(
        "--out-file",
        default="data/labels_main.json",
        help="Output file to store mapping (default: data/images/metadata/labels.json)",
    )
    args = p.parse_args()

    mapping = None
    if args.mapping_json:
        mapping = _load_mapping(args.mapping_json)

    process_images(
        images_dir=args.images_dir, case_to_project=mapping, out_file=args.out_file
    )
