# hybrid_rrf_test.py
# Purpose: Run BM25 and Dense (Chroma) retrieval, then fuse rankings via Reciprocal Rank Fusion (RRF).

import os, json, re
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ------------------ Paths ------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT, "Data_Processing_and_Indexing", "finance_book_chunks_enriched.json")
CHROMA_PATH = os.path.join(ROOT, "Data_Processing_and_Indexing", "chroma_store")

# ------------------ Load BM25 corpus ------------------
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
docs = [d["text"] for d in data]
metas = { d["id"]: {"chapter": d.get("chapter"), "order": d.get("order"), "tokens": d.get("tokens")} for d in data }

def tokenize(s: str) -> List[str]:
    # Very light normalization for BM25
    return [t for t in re.split(r"\W+", s.lower()) if t]

bm25 = BM25Okapi([tokenize(d) for d in docs])

# ------------------ Dense (Chroma) ------------------
client = PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
collection = client.get_collection("finance_book")
embedder = SentenceTransformer("thenlper/gte-large")  # same model as indexing

# ------------------ RRF helper ------------------
def rrf_fuse(ranked_lists: List[List[str]], k: int = 60) -> List[Tuple[str, float]]:
    """
    ranked_lists: list of id-lists ordered by descending relevance (rank 1 is best).
    Returns fused ranking as list of (id, score), higher is better.
    """
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ------------------ Query ------------------
query_text = "how to stop money leaks quickly without a full budget"

# BM25 ranking (ids)
bm25_scores = bm25.get_scores(tokenize(query_text))
bm25_topk = 10
bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:bm25_topk]
bm25_ids = [data[i]["id"] for i in bm25_indices]

# Dense ranking (ids)
q_emb = embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
dense_top = collection.query(query_embeddings=q_emb, n_results=10)
dense_ids = dense_top["ids"][0]

# RRF fusion
fused = rrf_fuse([bm25_ids, dense_ids], k=60)
top_n = 5
fused_top = [cid for cid, _ in fused[:top_n]]

# Fetch final docs from Chroma by ids (for consistency) 
got = collection.get(ids=fused_top)

# Pretty print
print("🔀 Hybrid RRF — query:", query_text)
print("=" * 80)
# Keep the original order returned by fused_top
id_to_doc = {got["ids"][i]: got["documents"][i] for i in range(len(got["ids"]))}
id_to_meta = {got["ids"][i]: got["metadatas"][i] for i in range(len(got["ids"]))}

for rank, cid in enumerate(fused_top, start=1):
    meta = id_to_meta.get(cid, metas.get(cid, {}))
    text = id_to_doc.get(cid, "")
    snippet = text[:300].replace("\n", " ")
    print(f"[Rank {rank}] ID: {cid} | Chapter: {meta.get('chapter')} | Order: {meta.get('order')} | Tokens: {meta.get('tokens')}")
    print("Text snippet:", snippet, "...\n")
