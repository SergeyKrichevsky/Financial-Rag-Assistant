# Retriever_Development/bm25_test_v3.py
# Purpose: Build a BM25 index over v3 chunks JSON and run a sanity query.
# This file is non-destructive and does not touch Chroma; it only loads the JSON.
#
# Deps:
#   pip install rank-bm25

import os
import json
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

# ---------- Config ----------
# Path to the enriched chunks JSON (v3)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(
    ROOT,
    "Data_Processing_and_Indexing",
    "finance_book_chunks_enriched.fixed.v3.json",
)

# Hardcoded test query
TEST_QUERY = "how to stop money leaks quickly without a full budget"
TOP_K = 3  # number of top results to show

# ---------- Tokenizer ----------
def tokenize(text: str) -> List[str]:
    """Very light normalization: lowercase + split on non-alphanumerics."""
    return [t for t in re.split(r"\W+", (text or "").lower()) if t]

# ---------- Load corpus ----------
def load_chunks(path: str) -> List[Dict[str, Any]]:
    """Load JSON list of chunks with fields: id, text, chapter, order, tokens, etc."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i, d in enumerate(data):
        d.setdefault("id", f"chunk_{i+1}")
        d.setdefault("text", "")
    return data

# ---------- Build BM25 ----------
def build_bm25(docs: List[str]) -> BM25Okapi:
    """Create BM25Okapi instance from raw documents (auto-tokenized)."""
    tokenized = [tokenize(d) for d in docs]
    return BM25Okapi(tokenized)

# ---------- Rank + pretty print ----------
def bm25_topk(bm25: BM25Okapi, query: str, data: List[Dict[str, Any]], k: int = 3) -> None:
    """Print top-k document metadata and snippet for given query."""
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    print("🔎 BM25 — sanity check")
    print("Query:", query)
    print("=" * 80)
    for rank, idx in enumerate(top_idx, start=1):
        d = data[idx]
        snippet = (d["text"][:300] or "").replace("\n", " ")
        print(f"[Rank {rank}] ID: {d['id']} | Chapter: {d.get('chapter')} | Order: {d.get('order')} | Tokens: {d.get('tokens')}")
        print("Text snippet:", snippet, "...\n")

# ---------- Main ----------
def main() -> None:
    print(f"[JSON] {JSON_PATH}")
    data = load_chunks(JSON_PATH)
    print(f"[Load] {len(data)} chunks")
    docs = [d["text"] for d in data]
    bm25 = build_bm25(docs)
    bm25_topk(bm25, TEST_QUERY, data, k=TOP_K)

if __name__ == "__main__":
    main()
