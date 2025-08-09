# bm25_test.py
# Purpose: Build a BM25 index over preprocessed chunks from JSON and run a sanity query.
# Folder layout assumption:
#   project_root/
#     Data_Processing_and_Indexing/
#       finance_book_chunks_enriched.json
#     Retriever_Development/
#       bm25_test.py  <-- you are here

import os
import json
import re
from rank_bm25 import BM25Okapi

# --- Resolve path to the JSON with enriched chunks ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT, "Data_Processing_and_Indexing", "finance_book_chunks_enriched.json")

# --- Load chunks ---
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- Prepare corpus for BM25 ---
# We keep parallel arrays so we can map back from tokens -> original text/metadata.
documents = [d["text"] for d in data]
metadatas = [{"id": d["id"], "chapter": d.get("chapter"), "order": d.get("order"), "tokens": d.get("tokens")} for d in data]

# Simple whitespace tokenizer; BM25 works with token lists
def tokenize(text: str):
    # Very light normalization: lowercase + split on non-letters/digits
    return [t for t in re.split(r"\W+", text.lower()) if t]

tokenized_corpus = [tokenize(doc) for doc in documents]

# --- Build BM25 index ---
bm25 = BM25Okapi(tokenized_corpus)

# --- Test query ---
query_text = "how to stop money leaks quickly without a full budget"
query_tokens = tokenize(query_text)
scores = bm25.get_scores(query_tokens)

# --- Get top-k results ---
k = 3
top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

print("🔎 BM25 sanity check")
print("Query:", query_text)
print("=" * 80)
for rank, idx in enumerate(top_indices, start=1):
    meta = metadatas[idx]
    snippet = documents[idx][:300].replace("\n", " ")
    print(f"[Rank {rank}] ID: {meta['id']} | Chapter: {meta.get('chapter')} | Order: {meta.get('order')} | Tokens: {meta.get('tokens')}")
    print("Text snippet:", snippet, "...\n")
