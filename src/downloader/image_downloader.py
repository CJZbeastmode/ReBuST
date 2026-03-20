import requests
import os
from decimal import Decimal, ROUND_HALF_UP
import json
import time

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

RARE_PROJECTS = [] # TODO


class ImageDownloader:
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
        cases = set()
        for fname in os.listdir(DATA_DIR):
            if not fname.endswith(FILE_EXT):
                continue
            name = fname[: -len(FILE_EXT)]
            parts = name.split("-")
            if len(parts) >= 3:
                case_id = "-".join(parts[:3])
                cases.add(case_id)
        return cases

    def _query_tcga_cases(self, project_id):
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
        Return a list of (file_id, file_name) for SVS slides of a TCGA case.
        """
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

        r = requests.post(self.GDC_FILES_ENDPOINT, json=payload)
        r.raise_for_status()

        hits = r.json()["data"]["hits"]
        return [(h["file_id"], h["file_name"]) for h in hits]

    def _download_file(self, file_id, out_path):
        url = f"{self.GDC_DATA_ENDPOINT}/{file_id}"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    # ---------------------------------------------------------
    # Calculate Image Proportions
    # ---------------------------------------------------------
    def calc_img_proportions(self):
        case_counts = {}
        all_cases = set()

        print("Querying TCGA projects...\n")

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

    # ---------------------------------------------------------
    # Query Images
    # ---------------------------------------------------------
    def calc_img_counts(self):
        image_counts = {}
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
    # Fetch Images
    # ---------------------------------------------------------
    def fetch_images(self, output_json=None):
        existing = self._existing_cases_on_disk()
        print(f"Existing local cases: {len(existing)}")

        selected = {}

        for project, target in self.image_counts.items():
            print(f"\nQuerying {project} …")
            tcga_cases = self._query_tcga_cases(project)

            # Remove cases already on disk
            candidates = [c for c in tcga_cases if c not in existing]

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

    # ---------------------------------------------------------
    # Download Images
    # ---------------------------------------------------------
    def download_images(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)

        total = len(self.fetched_images)
        downloaded = 0
        skipped = 0
        missing = 0

        t0 = time.time()
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

            if not slides:
                missing += 1
                print(f"[MISSING] No SVS found for {case_id} " f"({i}/{total})")
                continue

            # Deterministic choice: first slide
            file_id, file_name = slides[0]

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
    # Run Whole Pipeline
    # ---------------------------------------------------------
    def run(self):
        if self.fetched_images is None:
            if self.image_counts is None:
                if self.proportions is None:
                    self.calc_img_proportions()
                self.calc_img_counts()
            self.fetch_images(output_json=self.fetched_images_output_json)
        self.download_images()


if __name__ == "__main__":
    downloader = ImageDownloader(
        target_total=1000,
        PROJECTS=NORMAL_PROJECTS,
        fetched_images_output_json="tcga_cases_to_fetch.json",
    )
    #downloader.run()
