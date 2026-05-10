"""
Phrase scraper for biomedical literature using PubMed.

This module:
- Fetches abstracts from PubMed using Entrez
- Extracts noun phrases using spaCy
- Extracts keywords using KeyBERT
- Combines and ranks phrases by frequency

Typical use case:
    Identify common biomedical phrases for a TCGA project.

Example:
    SEARCH_QUERY = "lung squamous cell carcinoma"

P.S.: For China: set "export HF_ENDPOINT=https://hf-mirror.com"
"""

import spacy
from collections import Counter
from Bio import Entrez
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


# =========================================================
# Configuration
# =========================================================

SEARCH_QUERY = "lung squamous cell carcinoma"
NUM_PAPERS = 50

# Required by NCBI Entrez API (identification)
Entrez.email = "jay0816@outlook.com"  # TODO: replace with env

# Load NLP models
nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(sentence_model)


# =========================================================
# Query Functions
# =========================================================


def fetch_pubmed_abstracts(query, max_results=50):
    """
    Fetch abstracts from PubMed using the Entrez API.

    Process:
        1. Search PubMed for matching paper IDs
        2. Fetch abstracts for those IDs

    Args:
        query (str): PubMed search query.
        max_results (int): Maximum number of papers to retrieve.

    Returns:
        str: Concatenated abstracts as raw text.

    Notes:
        - Returns plain text, not structured JSON.
        - No retry or rate limiting implemented.
    """

    # Search for paper IDs matching the query
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)

    # Extract IDs
    ids = record["IdList"]

    # Fetch abstracts for the IDs
    handle = Entrez.efetch(
        db="pubmed",
        id=ids,
        rettype="abstract",
        retmode="text",
    )

    return handle.read()


def extract_noun_phrases(text):
    """
    Extract noun phrases from text using spaCy.

    Filters:
        - Phrase length between 2 and 5 words
        - Lowercased and stripped

    Args:
        text (str): Input text corpus.

    Returns:
        list[str]: List of noun phrases.

    Rationale:
        Noun phrases capture domain-specific terminologies
        (e.g., "tumor microenvironment", "gene expression").
    """

    doc = nlp(text)

    phrases = []

    # Iterate noun chunks and filter by length
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()

        # Keep phrases with 2-5 words
        if 2 <= len(phrase.split()) <= 5:
            phrases.append(phrase)

    return phrases


def extract_keybert(text):
    """
    Extract keyphrases using KeyBERT (semantic similarity-based).

    Args:
        text (str): Input text corpus.

    Returns:
        list[str]: List of extracted keyphrases.
    """

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=50,
    )

    return [k[0] for k in keywords]


# =========================================================
# Run Pipeline
# =========================================================
def run():
    """
    Run full phrase extraction pipeline.

    Steps:
        1. Fetch PubMed abstracts
        2. Extract noun phrases (syntactic)
        3. Extract KeyBERT phrases (semantic)
        4. Combine and rank by frequency

    Output:
        Prints top phrases and their counts.
    """

    # Step 1: Fetch abstracts
    text = fetch_pubmed_abstracts(SEARCH_QUERY, NUM_PAPERS)

    # Step 2 & 3: Extract phrases using both methods
    noun_phrases = extract_noun_phrases(text)
    keybert_phrases = extract_keybert(text)

    # Step 4: Combine and count frequencies
    all_phrases = noun_phrases + keybert_phrases

    # Count frequencies and print top phrases
    freq = Counter(all_phrases)
    top_phrases = freq.most_common(80)

    print("\nTop extracted phrases:\n")

    for phrase, count in top_phrases:
        print(f"{phrase} -> {count}")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    run()
