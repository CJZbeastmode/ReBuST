"""
Image downloader for TCGA WSIs (SVS).

This module:
- Queries TCGA (GDC API) for available cases per project
- Computes proportional sampling across all cases
- Selects cases not yet downloaded
- Downloads one SVS slide per case

Typical usage:
    downloader = ImageDownloader(target_total=1000)
    downloader.run()
"""

import requests
import os
from decimal import Decimal, ROUND_HALF_UP
import json
import time


# =========================================================
# Constants
# =========================================================

FIELDS = "cases.submitter_id,cases.project.project_id"
GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
DATA_DIR = "/Volumes/Xbox_HD/data/med_img"
FILE_EXT = ".svs"

NORMAL_PROJECTS = [
    "TCGA-COAD",
    "TCGA-READ",
    "TCGA-ESCA",
    "TCGA-STAD",
    "TCGA-LUAD",
    "TCGA-LUSC",
    "TCGA-MESO",
    "TCGA-CHOL",
    "TCGA-LIHC",
    "TCGA-PAAD",
    "TCGA-UVM",
    "TCGA-SKCM",
]

RARE_PROJECTS = []


# =========================================================
# Core Class
# =========================================================


class ImageDownloader:
    """
    Pipeline for querying, selecting, and downloading TCGA SVS images.

    Workflow:
        1. Query available cases per project
        2. Compute sampling proportions
        3. Select cases (excluding already downloaded)
        4. Download one slide per case

    Attributes:
        target_total (int): Total number of images to download.
        proportions (dict[str, float]): Project sampling proportions.
        image_counts (dict[str, int]): Number of images per project.
        fetched_images (dict[str, str]): Mapping case_id -> project.
        DATA_DIR (str): Output directory for images.
    """

    def __init__(
        self,
        target_total=1000,
        proportions=None,
        image_count=None,
        fetched_images=None,
        fetched_images_output_json=None,
        PROJECTS=NORMAL_PROJECTS,
        GDC_FILES_ENDPOINT=GDC_FILES_ENDPOINT,
        GDC_DATA_ENDPOINT=GDC_DATA_ENDPOINT,
        DATA_DIR=DATA_DIR,
        FILE_EXT=FILE_EXT,
    ):
        """
        Initialize downloader configuration.

        Args:
            target_total (int): Targeted total number of images.
            proportions (dict, optional): Precomputed proportions.
            image_count (dict, optional): Precomputed image counts.
            fetched_images (dict, optional): Prefetched cases.
            fetched_images_output_json (str, optional): Path to save selected cases.
            PROJECTS (list[str]): TCGA projects to include.
        """

        self.PROJECTS = PROJECTS
        self.target_total = target_total
        self.proportions = proportions
        self.image_counts = image_count
        self.fetched_images = fetched_images
        self.fetched_images_output_json = fetched_images_output_json
        self.GDC_FILES_ENDPOINT = GDC_FILES_ENDPOINT
        self.GDC_DATA_ENDPOINT = GDC_DATA_ENDPOINT
        self.DATA_DIR = DATA_DIR
        self.FILE_EXT = FILE_EXT

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _existing_cases_on_disk(self):
        """
        Scan DATA_DIR and extract existing TCGA case IDs.

        Assumes filenames follow:
            <case_id>-<project>.svs

        Returns:
            set[str]: Case IDs already exist locally.
        """

        cases = set()

        # Iterate files in DATA_DIR
        for fname in os.listdir(self.DATA_DIR):
            # Skip non-SVS and hidden files
            if not fname.endswith(self.FILE_EXT):
                continue
            # Consruct case_id from filename
            name = fname[: -len(self.FILE_EXT)]
            parts = name.split("-")
            if len(parts) >= 3:
                case_id = "-".join(parts[:3])
                cases.add(case_id)

        return cases

    def _query_tcga_cases(self, project_id):
        """
        Query GDC API for all cases in a project that have SVS slide images.

        Args:
            project_id (str): TCGA project (e.g., "TCGA-COAD")

        Returns:
            list[str]: Unique case submitter IDs.
        """

        # Metadata
        fields = "cases.submitter_id,cases.project.project_id"
        payload = {
            "filters": {
                "op": "and",
                "content": [
                    {
                        "op": "=",
                        "content": {
                            "field": "cases.project.project_id",
                            "value": project_id,
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
            "fields": fields,
            "format": "JSON",
            "size": 20000,  # large enough for all WSIs
        }

        # Request
        r = requests.post(self.GDC_FILES_ENDPOINT, json=payload)
        r.raise_for_status()

        hits = r.json()["data"]["hits"]

        # Deduplicate by case ID
        cases = set()
        for item in hits:
            for case in item["cases"]:
                cases.add(case["submitter_id"])

        return sorted(cases)

    def _query_svs_for_case(self, case_id):
        """
        Get SVS files for a given TCGA case.

        Args:
            case_id (str): TCGA case ID.

        Returns:
            list[tuple[str, str]]: List of (file_id, file_name).
        """

        # Metadata
        fields = "file_id,file_name,cases.submitter_id"
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
            "fields": fields,
            "format": "JSON",
            "size": 100,
        }

        # Request
        r = requests.post(self.GDC_FILES_ENDPOINT, json=payload)
        r.raise_for_status()

        hits = r.json()["data"]["hits"]
        return [(h["file_id"], h["file_name"]) for h in hits]

    def _download_file(self, file_id, out_path):
        """
        Download a file from GDC.

        Args:
            file_id (str): GDC file UUID.
            out_path (str): Destination file path.
        """

        url = f"{self.GDC_DATA_ENDPOINT}/{file_id}"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    # ---------------------------------------------------------
    # Image Proportions and Counts Manipulation
    # ---------------------------------------------------------

    def calc_img_proportions(self):
        """
        Compute sampling proportions based on available cases.

        Returns:
            dict[str, float]: Project proportions.
        """

        case_counts = {}
        all_cases = set()

        print("Querying TCGA projects...\n")

        # Iterate projects and count cases
        for proj in self.PROJECTS:
            cases = self._query_tcga_cases(proj)
            case_counts[proj] = len(cases)
            all_cases |= set(cases)
            print(f"{proj}: {len(cases)} cases")

        total_cases = sum(case_counts.values())

        print("\nTotal unique cases (sum of projects):", total_cases)

        # Compute proportions
        proportions = {proj: count / total_cases for proj, count in case_counts.items()}

        print("\nPROPORTIONS = {")
        for proj in sorted(proportions):
            print(f'    "{proj}": {proportions[proj]:.6f},')
        print("}")

        # Sanity check
        s = sum(proportions.values())
        print(f"\nSum of proportions: {s:.8f}")

        self.proportions = proportions
        return proportions

    def calc_img_counts(self):
        """
        Calculate absolute image counts by precomputed proportions.

        Returns:
            dict[str, int]: Number of images per project.
        """

        image_counts = {}

        # Iterate projects and compute counts
        for i in self.proportions:
            q = int(
                (
                    Decimal(str(self.proportions[i])) * Decimal(self.target_total)
                ).quantize(0, rounding=ROUND_HALF_UP)
            )
            image_counts[i] = q
            print(f"{i}: {q}")

        self.image_counts = image_counts
        return image_counts

    # ---------------------------------------------------------
    # Fetch and Download Images
    # ---------------------------------------------------------

    def fetch_images(self, output_json=None):
        """
        Select cases to download per project (exluding existing cases).

        Args:
            output_json (str, optional): Save selected cases to JSON.
        """

        existing = self._existing_cases_on_disk()
        print(f"Existing local cases: {len(existing)}")

        selected = {}

        # Iterate projects and select cases
        for project, target in self.image_counts.items():
            print(f"\nQuerying {project} …")
            tcga_cases = self._query_tcga_cases(project)

            # Remove cases already on disk
            candidates = [c for c in tcga_cases if c not in existing]

            # Warn if not enough candidates
            if len(candidates) < target:
                print(
                    f"  WARNING: requested {target}, "
                    f"only {len(candidates)} available"
                )

            take = candidates[:target]

            for case_id in take:
                selected[case_id] = project

            print(
                f"  Selected {len(take)} new cases "
                f"(from {len(tcga_cases)} total with SVS)"
            )

        # Write output
        if output_json is not None:
            with open(output_json, "w") as f:
                json.dump(selected, f, indent=2, sort_keys=True)
            print(f"\nDONE. Wrote {len(selected)} cases to {output_json}")

        self.fetched_images = selected

        print(f"\nDONE. Selected {len(selected)} cases in total.")

    def download_images(self):
        """
        Download one SVS slide per selected case.

        Skips existing files and deletes partial files on failure.
        """
        os.makedirs(self.DATA_DIR, exist_ok=True)

        total = len(self.fetched_images)
        downloaded = 0
        skipped = 0
        missing = 0

        t0 = time.time()

        # Iterate selected cases and download
        for i, (case_id, project) in enumerate(self.fetched_images.items(), start=1):
            category = project.replace("TCGA-", "")
            out_name = f"{case_id}-{category}{FILE_EXT}"
            out_path = os.path.join(self.DATA_DIR, out_name)

            # Skip if already exists
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                skipped += 1
                print(f"[SKIP] {out_name} already exists " f"({i}/{total})")
                continue

            # Query TCGA
            slides = self._query_svs_for_case(case_id)

            # Handle missing cases
            if not slides:
                missing += 1
                print(f"[MISSING] No SVS found for {case_id} " f"({i}/{total})")
                continue

            # Deterministic choice: first slide
            file_id, file_name = slides[0]

            # Download file
            print(f"[DOWNLOADING] {case_id} → {out_name} " f"({i}/{total})")
            try:
                self._download_file(file_id, out_path)
                downloaded += 1
                t1 = time.time()
                print(
                    f"[PROGRESS] {downloaded} / {total} downloaded in {t1 - t0:.1f} sec ({(t1 - t0)/downloaded:.2f} sec/file)"
                )
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
    # Run Pipeline
    # ---------------------------------------------------------

    def run(self):
        """
        Execute full pipeline:
            proportions → counts → selection → download
        """

        if self.fetched_images is None:
            if self.image_counts is None:
                if self.proportions is None:
                    # Step 1: Compute proportions
                    self.calc_img_proportions()
                # Step 2: Compute counts
                self.calc_img_counts()
            # Step 3: Fetch/select cases
            self.fetch_images(output_json=self.fetched_images_output_json)
        # Step 4: Download images
        self.download_images()


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    downloader = ImageDownloader(
        target_total=1000,
        PROJECTS=NORMAL_PROJECTS,
        fetched_images_output_json="tcga_cases_to_fetch.json",
    )
    downloader.run()
