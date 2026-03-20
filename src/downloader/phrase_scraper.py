import spacy
from collections import Counter
from Bio import Entrez
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# For China: export HF_ENDPOINT=https://hf-mirror.com  

# -----------------------------
# CONFIG
# -----------------------------
SEARCH_QUERY = "lung squamous cell carcinoma"
NUM_PAPERS = 50

Entrez.email = "jay0816@outlook.com"

nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(sentence_model)

# -----------------------------
# 1. FETCH PUBMED ABSTRACTS
# -----------------------------
def fetch_pubmed_abstracts(query, max_results=50):

    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results
    )
    record = Entrez.read(handle)

    ids = record["IdList"]

    handle = Entrez.efetch(
        db="pubmed",
        id=ids,
        rettype="abstract",
        retmode="text"
    )

    return handle.read()


# -----------------------------
# 2. EXTRACT NOUN PHRASES
# -----------------------------
def extract_noun_phrases(text):

    doc = nlp(text)

    phrases = []

    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()

        if 2 <= len(phrase.split()) <= 5:
            phrases.append(phrase)

    return phrases


# -----------------------------
# 3. KEYBERT KEYWORDS
# -----------------------------
def extract_keybert(text):

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1,3),
        stop_words="english",
        top_n=50
    )

    return [k[0] for k in keywords]


# -----------------------------
# MAIN PIPELINE
# -----------------------------
text = fetch_pubmed_abstracts(SEARCH_QUERY, NUM_PAPERS)

noun_phrases = extract_noun_phrases(text)
keybert_phrases = extract_keybert(text)

all_phrases = noun_phrases + keybert_phrases

freq = Counter(all_phrases)

top_phrases = freq.most_common(80)

print("\nTop extracted phrases:\n")

for phrase, count in top_phrases:
    print(phrase, "->", count)