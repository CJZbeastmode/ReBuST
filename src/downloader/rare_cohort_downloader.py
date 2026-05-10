"""
Rare cohort sampler for TCGA WSIs (SVS).

This module:
- Queries TCGA (GDC API) for available cases per rare project
- Samples 20 cases for each requested cancer type
- Optionally writes sampled cases to JSON

Typical usage:
    sampler = RareCohortDownloader()
    sampler.run()
"""

import json
import os
import requests


# =========================================================
# Constants
# =========================================================

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
DATA_DIR = "/Volumes/Xbox_HD/data/rare_cohort/img"
FILE_EXT = ".svs"
RARE_PROJECTS = [
    "TCGA-CHOL",
    "TCGA-ESCA",
    "TCGA-MESO",
    "TCGA-PAAD",
    "TCGA-READ",
    "TCGA-UVM",
]
SAMPLES_PER_PROJECT = 20


# =========================================================
# Core Class
# =========================================================


class RareCohortDownloader:
    """
    Minimal sampler for rare TCGA cohorts.

    Attributes:
        PROJECTS (list[str]): Rare TCGA projects to include.
        samples_per_project (int): Number of cases to sample per project.
        sampled_cases (dict[str, str]): Mapping case_id -> project.
    """

    def __init__(
        self,
        PROJECTS=RARE_PROJECTS,
        samples_per_project=SAMPLES_PER_PROJECT,
        output_json=None,
        GDC_FILES_ENDPOINT=GDC_FILES_ENDPOINT,
        GDC_DATA_ENDPOINT=GDC_DATA_ENDPOINT,
        DATA_DIR=DATA_DIR,
        FILE_EXT=FILE_EXT,
    ):
        self.PROJECTS = PROJECTS
        self.samples_per_project = samples_per_project
        self.output_json = output_json
        self.GDC_FILES_ENDPOINT = GDC_FILES_ENDPOINT
        self.GDC_DATA_ENDPOINT = GDC_DATA_ENDPOINT
        self.DATA_DIR = DATA_DIR
        self.FILE_EXT = FILE_EXT
        self.sampled_cases = {}

    # ---------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------

    def sample_images(self, output_json=None):
        """
        Sample fixed number of cases per rare TCGA project.

        Args:
            output_json (str, optional): Path to save selected cases.

        Returns:
            dict[str, str]: Mapping case_id -> project.
        """

        selected = {}

        # Iterate projects and sample cases
        for project in self.PROJECTS:
            payload = {
                "filters": {
                    "op": "and",
                    "content": [
                        {
                            "op": "=",
                            "content": {
                                "field": "cases.project.project_id",
                                "value": project,
                            },
                        },
                        {
                            "op": "=",
                            "content": {
                                "field": "data_type",
                                "value": "Slide Image",
                            },
                        },
                        {
                            "op": "in",
                            "content": {
                                "field": "data_format",
                                "value": ["SVS"],
                            },
                        },
                    ],
                },
                "fields": "cases.submitter_id,cases.project.project_id",
                "format": "JSON",
                "size": 20000,
            }

            print(f"\nQuerying {project} …")
            r = requests.post(self.GDC_FILES_ENDPOINT, json=payload)
            r.raise_for_status()
            hits = r.json()["data"]["hits"]

            cases = sorted(
                {
                    case["submitter_id"]
                    for item in hits
                    for case in item.get("cases", [])
                    if "submitter_id" in case
                }
            )

            take = cases[: self.samples_per_project]
            if len(take) < self.samples_per_project:
                print(
                    f"  WARNING: requested {self.samples_per_project}, "
                    f"only {len(take)} available"
                )

            for case_id in take:
                selected[case_id] = project

            print(
                f"  Selected {len(take)} cases "
                f"(from {len(cases)} total with SVS)"
            )

        out_path = output_json if output_json is not None else self.output_json
        if out_path is not None:
            with open(out_path, "w") as f:
                json.dump(selected, f, indent=2, sort_keys=True)
            print(f"\nDONE. Wrote {len(selected)} cases to {out_path}")

        self.sampled_cases = selected
        print(f"\nDONE. Sampled {len(selected)} cases in total.")
        return selected

    def download_images(self):
        """
        Download one SVS slide per sampled case.
        """

        if not self.sampled_cases:
            print("No sampled cases found. Running sampling first...")
            self.sample_images(output_json=self.output_json)

        os.makedirs(self.DATA_DIR, exist_ok=True)

        total = len(self.sampled_cases)
        downloaded = 0
        skipped = 0
        missing = 0

        # Iterate sampled cases and download
        for i, (case_id, project) in enumerate(self.sampled_cases.items(), start=1):
            category = project.replace("TCGA-", "")
            case_parts = case_id.split("-")
            case_stub = "-".join(case_parts[:3]) if len(case_parts) >= 3 else case_id
            out_name = f"{case_stub}-{category}{self.FILE_EXT}"
            out_path = os.path.join(self.DATA_DIR, out_name)

            # Skip if already exists
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                skipped += 1
                print(f"[SKIP] {out_name} already exists ({i}/{total})")
                continue

            # Query SVS slides for case
            payload = {
                "filters": {
                    "op": "and",
                    "content": [
                        {
                            "op": "=",
                            "content": {
                                "field": "cases.submitter_id",
                                "value": case_id,
                            },
                        },
                        {
                            "op": "=",
                            "content": {
                                "field": "data_type",
                                "value": "Slide Image",
                            },
                        },
                        {
                            "op": "in",
                            "content": {
                                "field": "data_format",
                                "value": ["SVS"],
                            },
                        },
                    ],
                },
                "fields": "file_id,file_name",
                "format": "JSON",
                "size": 100,
            }

            r = requests.post(self.GDC_FILES_ENDPOINT, json=payload)
            r.raise_for_status()
            hits = r.json()["data"]["hits"]

            if not hits:
                missing += 1
                print(f"[MISSING] No SVS found for {case_id} ({i}/{total})")
                continue

            file_id = hits[0]["file_id"]
            url = f"{self.GDC_DATA_ENDPOINT}/{file_id}"

            print(f"[DOWNLOADING] {case_id} → {out_name} ({i}/{total})")
            try:
                with requests.get(url, stream=True) as resp:
                    resp.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                downloaded += 1
            except Exception as e:
                print(f"  ERROR downloading {case_id}: {e}")
                if os.path.exists(out_path):
                    os.remove(out_path)

        print("\nSUMMARY")
        print(f"  Downloaded : {downloaded}")
        print(f"  Skipped    : {skipped}")
        print(f"  Missing    : {missing}")
        print(f"  Total      : {total}")

    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------

    def run(self):
        """Run rare cohort sampling."""

        self.sample_images(output_json=self.output_json)
        self.download_images()


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    sampler = RareCohortDownloader()
    sampler.run()
