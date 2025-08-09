# Retriever_Development/eval_retriever_v3.py
# Purpose: Evaluate the hybrid retriever (BM25 + Dense with RRF) on a small devset.
# Metrics: hit@k and precision@k for k in {1, 3, 5}.
#
# Usage:
#   python Retriever_Development/eval_retriever_v3.py
#
# Devset format (JSONL):
#   {"q": "question text", "gold_chunk_ids": ["chunk_8", "chunk_10"]}
#   {"q": "...", "gold_chunk_ids": ["chunk_14"]}
#
# Deps:
#   pip install rank-bm25 "chromadb>=0.5.0" sentence-transformers

import os
import json
import re
from typing import List, Dict, Any, Tuple

from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer

# ------------------ Paths & Config ------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JSON_PATH = os.path.join(
    ROOT,
    "Data_Processing_and_Indexing",
    "finance_book_chunks_enriched.fixed.v3.json",
)
CHROMA_PATH = os.path.join(ROOT, "chroma_store")
COLLECTION_NAME = "finance_book_v3"
DEVSET_PATH = os.path.join(ROOT, "Retriever_Development", "devset_q2chunk.jsonl")

EMBED_MODEL = "all-MiniLM-L6-v2"
BM25_TOPK = 50     # broad candidate pool from BM25
DENSE_TOPK = 50    # broad candidate pool from Dense
FUSE_TOPK = 10     # how many we consider after fusion for metric@k

K_LIST = [1, 3, 5]  # ks for hit@k and precision@k

# ------------------ Tokenization ------------------
def tokenize(s: str) -> List[str]:
    """Very light normalization for BM25: lowercase + split on non-alphanumerics."""
    return [t for t in re.split(r"\W+", (s or "").lower()) if t]

# ------------------ Loading ------------------
def load_chunks(path: str) -> List[Dict[str, Any]]:
    """Load v3 chunks from JSON and ensure 'id'/'text' presence."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i, d in enumerate(data):
        d.setdefault("id", f"chunk_{i+1}")
        d.setdefault("text", "")
    return data

def load_devset(path: str) -> List[Dict[str, Any]]:
    """Load JSONL devset with fields: q, gold_chunk_ids (list[str])."""
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "q" in obj and "gold_chunk_ids" in obj and isinstance(obj["gold_chunk_ids"], list):
                rows.append(obj)
    return rows

# ------------------ Hybrid Retriever (RRF) ------------------
def rrf_fuse(ranked_lists: List[List[str]], k: int = 60) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion: return [(id, score)] sorted by score desc."""
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

class HybridRetriever:
    """Hybrid retriever over local JSON (BM25) + Chroma (Dense)."""

    def __init__(self, json_path: str, chroma_path: str, collection: str, embed_model: str):
        # Build BM25 over texts from JSON
        data = load_chunks(json_path)
        self.data = data
        self.id_by_index = [d["id"] for d in data]
        self.texts = [d["text"] for d in data]
        self.metas = {d["id"]: {"chapter": d.get("chapter"), "order": d.get("order"), "tokens": d.get("tokens")}
                      for d in data}
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])

        # Dense side (Chroma)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(collection)
        self.embedder = SentenceTransformer(embed_model)

    def retrieve(self, query: str, bm25_topk: int, dense_topk: int, fuse_topk: int) -> List[str]:
        """Return fused top-k chunk IDs for a given query."""
        # BM25 ids
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:bm25_topk]
        bm25_ids = [self.id_by_index[i] for i in bm25_idx]

        # Dense ids
        q_emb = self.embedder.encode([query], convert_to_tensor=False)
        dense = self.collection.query(query_embeddings=q_emb, n_results=dense_topk)
        dense_ids: List[str] = dense["ids"][0]

        # RRF fusion
        fused = rrf_fuse([bm25_ids, dense_ids], k=60)
        top_ids = [cid for cid, _ in fused[:fuse_topk]]
        return top_ids

# ------------------ Metrics ------------------
def hit_at_k(pred_ids: List[str], gold_ids: List[str], k: int) -> float:
    """1.0 if any of the first-k predictions intersects gold, else 0.0."""
    return 1.0 if set(pred_ids[:k]) & set(gold_ids) else 0.0

def precision_at_k(pred_ids: List[str], gold_ids: List[str], k: int) -> float:
    """|top-k ∩ gold| / k."""
    return len(set(pred_ids[:k]) & set(gold_ids)) / float(k)

def evaluate(devset: List[Dict[str, Any]], retriever: HybridRetriever) -> None:
    """Compute and print hit@k and precision@k across devset."""
    if not devset:
        print(f"[WARN] Devset not found: {DEVSET_PATH}")
        print("Create it with lines like:")
        print('  {"q": "how to stop money leaks", "gold_chunk_ids": ["chunk_8", "chunk_10"]}')
        return

    print(f"[Eval] {len(devset)} queries | ks = {K_LIST}")
    agg_hits = {k: 0.0 for k in K_LIST}
    agg_prec = {k: 0.0 for k in K_LIST}

    for i, row in enumerate(devset, start=1):
        q = row["q"]
        gold = row["gold_chunk_ids"]
        pred = retriever.retrieve(q, BM25_TOPK, DENSE_TOPK, FUSE_TOPK)

        for k in K_LIST:
            agg_hits[k] += hit_at_k(pred, gold, k)
            agg_prec[k] += precision_at_k(pred, gold, k)

        # Optional: brief trace for the first few queries
        print(f"\n[Q{i}] {q}")
        print(f" gold: {gold}")
        print(f" pred(top{FUSE_TOPK}): {pred}")

    n = float(len(devset))
    print("\n===== Results =====")
    for k in K_LIST:
        print(f"hit@{k}: {agg_hits[k] / n:.3f} | precision@{k}: {agg_prec[k] / n:.3f}")

# ------------------ Main ------------------
def main() -> None:
    print(f"[JSON]   {JSON_PATH}")
    print(f"[Chroma] {CHROMA_PATH} | collection='{COLLECTION_NAME}'")
    print(f"[Devset] {DEVSET_PATH}")

    retriever = HybridRetriever(
        json_path=JSON_PATH,
        chroma_path=CHROMA_PATH,
        collection=COLLECTION_NAME,
        embed_model=EMBED_MODEL,
    )
    devset = load_devset(DEVSET_PATH)
    evaluate(devset, retriever)

if __name__ == "__main__":
    main()
