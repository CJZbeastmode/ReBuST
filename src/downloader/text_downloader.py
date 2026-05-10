"""
Text downloader and FAISS index builder for TCGA-related publications.

This module:
- Queries Europe PMC (EBI) for open-access papers related to TCGA projects
- Extracts text snippets (figure captions, histopathology, microscopy, results)
- Saves each paper's snippets as a .txt file per project
- Optionally builds a FAISS index of text embeddings for semantic search

Typical workflow:
    downloader = TextDownloader(TCGA_KEYWORDS, per_project=50, build_faiss=True)
    downloader.run()
"""

import os
import time
import glob
import requests
import json
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup

import torch
from transformers import CLIPProcessor, CLIPModel
import faiss_util


BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


# =========================================================
# Text Downloader Class
# =========================================================
class TextDownloader:
    """
    Pipeline to fetch, process, and index TCGA-related publications.

    Attributes:
        TCGA_KEYWORDS (dict): Mapping project codes -> search queries.
        out_dir (str): Root directory to save text files.
        per_project (int): Max number of papers per project.
        build_faiss_flag (bool): Option to build FAISS index after downloading.
        faiss_index_out (str): Output file for FAISS index.
        faiss_paths_out (str): Output file for mapping text chunks to source files.
    """

    def __init__(
        self,
        TCGA_KEYWORDS,
        out_dir="./data/epmc_tcga_corpus",
        per_project=50,
        build_faiss=False,
        faiss_index_out="txt_index.faiss",
        faiss_paths_out="filenames.npy",
    ):
        self.TCGA_KEYWORDS = TCGA_KEYWORDS
        self.out_dir = out_dir
        self.per_project = per_project

        self.build_faiss_flag = build_faiss
        self.faiss_index_out = faiss_index_out
        self.faiss_paths_out = faiss_paths_out

        # Ensure output directory exists
        os.makedirs(self.out_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _query_epmc(self, keyword_query, cursor="*", page_size=25):
        """
        Query Europe PMC API for open-access publications matching query.

        Args:
            keyword_query (str): Search query for Europe PMC.
            cursor (str): Cursor for pagination.
            page_size (int): Number of results per request.

        Returns:
            dict: JSON response from Europe PMC.
        """

        params = {
            "query": f"({keyword_query}) AND OPEN_ACCESS:Y",
            "format": "json",
            "pageSize": page_size,
            "cursorMark": cursor,
        }
        r = requests.get(BASE_URL, params=params)
        r.raise_for_status()
        return r.json()

    def _fetch_fulltext_xml(self, pmcid):
        """
        Download the full-text XML of a paper from Europe PMC.

        Args:
            pmcid (str): PubMed Central ID of the paper.

        Returns:
            str or None: XML text if available, else None.
        """

        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.text

    def _extract_snippets(self, xml_text):
        """
        Extract relevant text snippets from full-text XML.

        Extracted content:
            - Figure captions (<fig><caption>)
            - Sections (<sec>) with titles related to:
              histopathology, microscopy, or results

        Args:
            xml_text (str): Full-text XML.

        Returns:
            list[str]: Extracted text snippets.
        """

        soup = BeautifulSoup(xml_text, "xml")
        snippets = []

        # Figure captions
        for fig in soup.find_all("fig"):
            cap = fig.find("caption")
            if cap:
                snippets.append(cap.get_text(" ", strip=True))

        # Histopathology / microscopy / results
        for sec in soup.find_all("sec"):
            title = sec.find("title")
            title_text = title.get_text().lower() if title else ""
            if any(k in title_text for k in ["histopath", "microscop", "result"]):
                snippets.append(sec.get_text(" ", strip=True))

        return snippets

    # ---------------------------------------------------------
    # Fetch TCGA Texts
    # ---------------------------------------------------------
    def fetch_for_project(self, code, keyword_query):
        """
        Download and save open-access publications for a single TCGA project.

        Args:
            code (str): Project code (e.g., 'TCGA-COAD').
            keyword_query (str): Europe PMC search query.

        Returns:
            int: Number of papers successfully downloaded.
        """

        save_dir = os.path.join(self.out_dir, code)
        os.makedirs(save_dir, exist_ok=True)

        cursor = "*"
        downloaded = 0

        print(f"\n=== Fetching {code}: {keyword_query} ===")

        # Iterate through paginated results, until per_project limit is reached
        while True:
            # Query Europe PMC for results
            data = self._query_epmc(keyword_query, cursor)
            results = data.get("resultList", {}).get("result", [])

            # Quit if no results
            if not results:
                break

            # Iterate through results and download full-text XML
            # Extract snippets
            # And save
            for hit in tqdm(results, desc=f"{code} batch"):
                pmcid = hit.get("pmcid")
                # Quit on missing PMCID (can't fetch full text)
                if not pmcid:
                    continue

                out_path = os.path.join(save_dir, f"{pmcid}.txt")
                # Quit on existing file (already downloaded)
                if os.path.exists(out_path):
                    continue

                xml_text = self._fetch_fulltext_xml(pmcid)
                # Quit on missing full-text XML (not available for this paper)
                if xml_text is None:
                    continue

                snippets = self._extract_snippets(xml_text)
                # Quit on no relevant snippets extracted (not useful for our purposes)
                if not snippets:
                    continue

                text = "\n".join(snippets)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)

                downloaded += 1
                time.sleep(0.2)

                if downloaded >= self.per_project:
                    return downloaded

            cursor = data.get("nextCursorMark")
            if not cursor:
                break

        return downloaded

    def fetch_texts(self):
        """
        Loop over all TCGA projects and fetch their publications.
        """
        # Iterate over projects and fetch texts
        for code, query in self.TCGA_KEYWORDS.items():
            count = self.fetch_for_project(code, query)
            print(f"{code}: downloaded {count} articles")

    # ---------------------------------------------------------
    # Text Processing
    # ---------------------------------------------------------
    @staticmethod
    def _load_text(path):
        """Load a text file into a string."""

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _chunk_text(text, max_chars=300, stride=250):
        """
        Chunk text into overlapping segments for embedding.

        Args:
            text (str): Full text string.
            max_chars (int): Max characters per chunk.
            stride (int): Step size between chunks.

        Returns:
            list[str]: List of text chunks.
        """

        chunks = []
        start = 0

        # Iterate with stride until end of text
        # Creating overlapping chunks
        while start < len(text):
            chunks.append(text[start : start + max_chars])
            start += stride

        return chunks

    # ---------------------------------------------------------
    # Build FAISS
    # ---------------------------------------------------------
    def build_faiss(self):
        """
        Encode all text chunks using CLIP/PLIP and build FAISS index.
        Saves:
            - FAISS index (.faiss)
            - Corresponding file paths (.npy)
        """

        print("\nCollecting text files...")
        txt_files = glob.glob(f"{self.out_dir}/**/*.txt", recursive=True)

        docs = []
        file_refs = []

        # Iterate through all text files
        # Load and chunk them
        # Keep track of source file for each chunk
        for path in txt_files:
            text = self._load_text(path)
            # Iterate through chunks of the text (overlapping segments)
            for chunk in self._chunk_text(text):
                docs.append(chunk[:256])
                file_refs.append(path)

        print(f"Loaded {len(txt_files)} files → {len(docs)} chunks")

        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Encode all chunks using CLIP/PLIP text encoder
        model = CLIPModel.from_pretrained("vinid/plip").to(device)
        processor = CLIPProcessor.from_pretrained("vinid/plip")
        model.eval()

        embeddings = []
        batch_size = 8

        # Iterate through chunks in batches
        # Encode with CLIP/PLIP
        # Collect embeddings
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            inputs = processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(device)

            with torch.no_grad():
                feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())

        embeddings = np.vstack(embeddings).astype("float32")

        dim = embeddings.shape[1]
        index = faiss_util.IndexFlatIP(dim)
        index.add(embeddings)

        faiss_util.write_index(index, self.faiss_index_out)
        np.save(self.faiss_paths_out, np.array(file_refs))

        print(f"Saved FAISS index → {self.faiss_index_out}")
        print(f"Saved paths      → {self.faiss_paths_out}")

    # ---------------------------------------------------------
    # Run Pipeline
    # ---------------------------------------------------------
    def run(self):
        self.fetch_texts()
        if self.build_faiss_flag:
            self.build_faiss()


if __name__ == "__main__":
    TCGA_KEYWORDS = {
        "TCGA-COAD": "colon adenocarcinoma OR colorectal adenocarcinoma",
        "TCGA-READ": "rectal adenocarcinoma OR colorectal adenocarcinoma",
    }

    downloader = TextDownloader(
        TCGA_KEYWORDS=TCGA_KEYWORDS,
        per_project=10,
        build_faiss=True,
    )

    downloader.run()
