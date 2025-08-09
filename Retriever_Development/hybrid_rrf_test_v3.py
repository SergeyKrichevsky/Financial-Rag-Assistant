# Retriever_Development/hybrid_rrf_test_v3.py
# Purpose: Hybrid retrieval demo — BM25 (JSON) + Dense (Chroma) fused via RRF.
# Runs with a built-in test query; no CLI args are needed.
#
# Deps:
#   pip install rank-bm25 "chromadb>=0.5.0" sentence-transformers

import os
import json
import re
from typing import List, Dict, Tuple, Any

from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer

# ------------------ Config ------------------
# Project root = parent of this file's parent
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JSON_PATH = os.path.join(
    ROOT,
    "Data_Processing_and_Indexing",
    "finance_book_chunks_enriched.fixed.v3.json",
)

# Chroma persistent dir (repo root / chroma_store)
CHROMA_PATH = os.path.join(ROOT, "chroma_store")

# Chroma collection name (new, safe alongside older ones)
COLLECTION_NAME = "finance_book_v3"

# Embedding model used both for indexing and querying
EMBED_MODEL = "all-MiniLM-L6-v2"

# Built-in test query and settings
TEST_QUERY = "how to stop money leaks quickly without a full budget"
BM25_TOPK = 10
DENSE_TOPK = 10
FINAL_TOPK = 5
RRF_K = 60  # typical RRF stabilization constant


# ------------------ Utils ------------------
def tokenize(s: str) -> List[str]:
    """Very light normalization for BM25: lowercase + split on non-alphanumerics."""
    return [t for t in re.split(r"\W+", (s or "").lower()) if t]


def load_chunks(path: str) -> List[Dict[str, Any]]:
    """Load JSON list of chunks and ensure minimal fields."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i, d in enumerate(data):
        d.setdefault("id", f"chunk_{i+1}")
        d.setdefault("text", "")
    return data


def rrf_fuse(ranked_lists: List[List[str]], k: int = 60) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion.
    ranked_lists: list of id-lists ordered by rank (rank 1 is best).
    Returns fused ranking as list of (id, score) sorted descending.
    """
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ------------------ Main ------------------
def main() -> None:
    # ---- Load JSON & build BM25 ----
    print(f"[JSON] {JSON_PATH}")
    data = load_chunks(JSON_PATH)
    docs = [d["text"] for d in data]
    metas = {d["id"]: {"chapter": d.get("chapter"), "order": d.get("order"), "tokens": d.get("tokens")} for d in data}
    bm25 = BM25Okapi([tokenize(d) for d in docs])

    # ---- Dense (Chroma) client/collection ----
    print(f"[Chroma] {CHROMA_PATH} | collection='{COLLECTION_NAME}'")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    # ---- Embedder ----
    embedder = SentenceTransformer(EMBED_MODEL)

    # ---- BM25 ranking (ids) ----
    bm25_scores = bm25.get_scores(tokenize(TEST_QUERY))
    bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:BM25_TOPK]
    bm25_ids = [data[i]["id"] for i in bm25_indices]

    # ---- Dense ranking (ids) ----
    q_emb = embedder.encode([TEST_QUERY], convert_to_tensor=False)
    dense = collection.query(query_embeddings=q_emb, n_results=DENSE_TOPK)
    dense_ids: List[str] = dense["ids"][0]

    # ---- RRF fusion ----
    fused = rrf_fuse([bm25_ids, dense_ids], k=RRF_K)
    fused_top = [cid for cid, _ in fused[:FINAL_TOPK]]

    # ---- Fetch final docs from Chroma for consistent metadata ----
    got = collection.get(ids=fused_top)
    id_to_doc = {got["ids"][i]: got["documents"][i] for i in range(len(got["ids"]))}
    id_to_meta = {got["ids"][i]: got["metadatas"][i] for i in range(len(got["ids"]))}

    # ---- Pretty print ----
    print("🔀 Hybrid RRF — query:", TEST_QUERY)
    print("=" * 80)
    for rank, cid in enumerate(fused_top, start=1):
        meta = id_to_meta.get(cid, metas.get(cid, {})) or {}
        text = id_to_doc.get(cid, "")
        snippet = (text[:300] or "").replace("\n", " ")
        print(f"[Rank {rank}] ID: {cid} | Chapter: {meta.get('chapter')} | Order: {meta.get('order')} | Tokens: {meta.get('tokens')}")
        print("Text snippet:", snippet, "...\n")


if __name__ == "__main__":
    main()
