# file: Retriever_Development/v4/hybrid_retriever_v4.py
# -*- coding: utf-8 -*-
"""
Hybrid retriever v4: BM25s (sparse) + Chroma (dense) -> RRF -> (optional) rerank -> MMR
- Expects a prebuilt BM25s index saved under artifacts/v4/bm25_index/
- Works against existing Chroma collection "finance_book_v4_cos" stored in ./chroma_store

Usage (local quick check):
    python -c "from Retriever_Development.v4.hybrid_retriever_v4 import HybridRetrieverV4; \
               r=HybridRetrieverV4(); \
               res=r.retrieve('How to build emergency fund?', k=8); \
               print(res[0]['document'][:200], res[0]['metadata'])"

Notes:
- All comments are in English by project policy.
"""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np

# Dense store (Chroma)
import chromadb
from chromadb.config import Settings

# Sparse lexical search (BM25S)
import bm25s


# ------------------------------- Config ------------------------------------- #

@dataclass
class RetrieverConfig:
    # Paths (project-root relative)
    chroma_path: str = os.path.join(".", "chroma_store")
    chroma_collection: str = "finance_book_v4_cos"

    bm25_dir: str = os.path.join("artifacts", "v4", "bm25_index")
    bm25_index_name: str = "bm25_fb_v4"  # basename used by bm25s.save(...)
    bm25_meta_filename: str = "index_meta.json"  # stores the ordered list of doc IDs

    # Fusion / post-processing
    rrf_k: int = 60
    candidate_pool: int = 40  # how many dense/sparse to pull before fusion
    mmr_lambda: float = 0.7
    final_k: int = 10

    # Safety caps
    max_get_batch: int = 256  # max ids to fetch from Chroma in one call


# ---------------------------- Hybrid Retriever ------------------------------ #

class HybridRetrieverV4:
    def __init__(self, cfg: RetrieverConfig | None = None) -> None:
        self.cfg = cfg or RetrieverConfig()
        self._client = self._load_chroma_client()
        self._collection = self._get_existing_collection(self.cfg.chroma_collection)
        self._bm25, self._bm25_ids = self._load_bm25_with_ids()

    # ---- Chroma (dense) ----
    def _load_chroma_client(self):
        """Create a local persistent Chroma client anchored to ./chroma_store."""
        path = os.path.abspath(self.cfg.chroma_path)
        os.makedirs(path, exist_ok=True)
        # We do not supply an embedding function here; the collection already exists with its own settings.
        return chromadb.PersistentClient(path=path, settings=Settings())

    def _get_existing_collection(self, name: str):
        """Return an existing collection by name (no implicit creation)."""
        try:
            return self._client.get_collection(name=name)
        except Exception as e:
            raise RuntimeError(
                f'Chroma collection "{name}" not found at {self.cfg.chroma_path}. '
                f"Make sure the vector store is present and named correctly."
            ) from e

    # ---- BM25s (sparse) ----
    def _load_bm25_with_ids(self):
        """
        Load bm25s index and accompanying id mapping. The mapping is required to
        convert bm25s result indices into Chroma IDs for fusion.
        """
        index_dir = os.path.abspath(self.cfg.bm25_dir)
        index_base = os.path.join(index_dir, self.cfg.bm25_index_name)
        meta_path = os.path.join(index_dir, self.cfg.bm25_meta_filename)

        if not os.path.exists(index_dir):
            raise FileNotFoundError(
                f'BM25 index directory not found: "{index_dir}". '
                f'Please run "build_bm25_index_v4.py" to create it.'
            )

        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f'Missing BM25 meta file with IDs: "{meta_path}". '
                f'It must contain JSON {{"ids": [...]}} aligned with the bm25s index order.'
            )

        # Load id mapping
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "ids" not in meta or not isinstance(meta["ids"], list):
            raise ValueError(f'Invalid meta file structure in "{meta_path}". Expected key "ids".')
        bm25_ids = meta["ids"]

        # Load bm25s index; mmap=True keeps memory usage modest for large indexes
        bm25 = bm25s.BM25.load(index_base, mmap=True, load_corpus=False)
        return bm25, bm25_ids

    # ---- Public API ----
    def retrieve(
        self,
        query: str,
        k: int | None = None,
        use_rerank: bool = False,  # placeholder; cross-encoder rerank can be added later
        use_hyde: bool = False,    # placeholder; hypothetical doc expansion can be added later
        filters: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Run hybrid retrieval: dense + sparse -> RRF -> (opt) rerank -> MMR.
        Returns a list of dicts: {id, score, document, metadata}.
        """
        final_k = k or self.cfg.final_k

        # 1) Dense retrieval from Chroma
        dense = self._dense_search(query, top_k=self.cfg.candidate_pool, filters=filters)

        # 2) Sparse retrieval from BM25s
        sparse = self._sparse_search(query, top_k=self.cfg.candidate_pool)

        # 3) RRF fuse (works on ranks only, so it's robust to score scales)
        fused_scores = self._rrf_fuse(dense_results=dense, sparse_results=sparse, k=self.cfg.rrf_k)

        # Optional: placeholder for cross-encoder rerank (not implemented here)
        # if use_rerank: fused_scores = self._cross_encoder_rerank(query, fused_scores)

        # 4) MMR diversity on fused candidates (uses embeddings from Chroma)
        ranked_ids = [doc_id for doc_id, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
        selected_ids = self._mmr_select(ranked_ids, fused_scores, k=final_k, lambda_mult=self.cfg.mmr_lambda)

        # 5) Fetch final payload (documents + metadata) and return
        final_payload = self._get_documents_by_ids(selected_ids)
        # Attach fused+mmr score for transparency
        score_map = fused_scores
        for item in final_payload:
            item["score"] = float(score_map.get(item["id"], 0.0))
        # Keep the original order as selected by MMR
        id_to_pos = {doc_id: i for i, doc_id in enumerate(selected_ids)}
        final_payload.sort(key=lambda d: id_to_pos.get(d["id"], 10**9))
        return final_payload

    # ---- Dense and Sparse retrieval primitives ----
    # def _dense_search(self, query: str, top_k: int, filters: Dict[str, Any] | None) -> List[Tuple[str, float]]:
    #     """
    #     Query Chroma collection; return list of (id, distance) sorted by ascending distance.
    #     We request embeddings now because MMR later needs them for doc-doc similarity.
    #     """
    #     result = self._collection.query(
    #         query_texts=[query],
    #         n_results=top_k,
    #         where=filters or {},
    #         include=["ids", "documents", "metadatas", "distances", "embeddings"],
    #     )
    #     ids = result.get("ids", [[]])[0]
    #     distances = result.get("distances", [[]])[0]
    #     # already ordered by best match (smallest distance) from Chroma
    #     return list(zip(ids, distances))
    
    def _dense_search(self, query: str, top_k: int, filters: Dict[str, Any] | None) -> List[Tuple[str, float]]:
        """
        Query Chroma collection; return list of (id, distance) sorted by ascending distance.
        NOTE: Don't send empty {} to `where` — pass None/omit instead.
        """
        query_kwargs = dict(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "embeddings"],  # valid keys only
        )
        # Only add 'where' if user supplied filters
        if filters:
            query_kwargs["where"] = filters  # for metadata filtering
        # (если потребуется фильтровать по тексту, используем where_document=... отдельно)

        result = self._collection.query(**query_kwargs)
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return list(zip(ids, distances))



    def _sparse_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        BM25s retrieve; return list of (id, bm25_score) sorted by descending score.
        We map bm25's index positions back to Chroma IDs via bm25_meta['ids'].
        """
        q_tokens = bm25s.tokenize(query)
        results, scores = self._bm25.retrieve(q_tokens, k=top_k)  # shapes: (1, k)
        idxs = results[0].tolist() if hasattr(results, "shape") else list(results[0])
        scs = scores[0].tolist() if hasattr(scores, "shape") else list(scores[0])
        pairs = [(self._bm25_ids[i], float(s)) for i, s in zip(idxs, scs)]
        # ensure sorting by score desc (bm25s returns in-order but keep explicit)
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    # ---- Fusion (RRF) and Diversity (MMR) ---------------------------------- #
    @staticmethod
    def _rrf(scores_by_id: Dict[str, int], k: int) -> Dict[str, float]:
        """Convert rank positions (1-based) to RRF scores for one ranking list."""
        return {doc_id: 1.0 / (k + rank) for doc_id, rank in scores_by_id.items()}

    def _rrf_fuse(
        self,
        dense_results: List[Tuple[str, float]],  # (id, distance asc)
        sparse_results: List[Tuple[str, float]], # (id, bm25_score desc)
        k: int,
    ) -> Dict[str, float]:
        """Reciprocal Rank Fusion over dense & sparse lists."""
        # Build rank dictionaries (1-based ranks)
        dense_rank: Dict[str, int] = {}
        for i, (doc_id, _dist) in enumerate(dense_results, start=1):
            # if the same id appears multiple times (shouldn't), keep best rank
            dense_rank[doc_id] = min(i, dense_rank.get(doc_id, i))
        sparse_rank: Dict[str, int] = {}
        for i, (doc_id, _score) in enumerate(sparse_results, start=1):
            sparse_rank[doc_id] = min(i, sparse_rank.get(doc_id, i))

        # Compute per-list RRF scores
        dense_rrf = self._rrf(dense_rank, k=k)
        sparse_rrf = self._rrf(sparse_rank, k=k)

        # Sum them up
        all_ids = set(dense_rank) | set(sparse_rank)
        fused: Dict[str, float] = {}
        for doc_id in all_ids:
            fused[doc_id] = dense_rrf.get(doc_id, 0.0) + sparse_rrf.get(doc_id, 0.0)
        return fused

    @staticmethod
    def _cosine_sim_matrix(vecs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix for (n_docs, dim)."""
        # Normalize rows to unit vectors
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        V = vecs / norms
        return V @ V.T  # (n, n)

    def _mmr_select(
        self,
        ranked_ids: List[str],
        rel_scores: Dict[str, float],
        k: int,
        lambda_mult: float,
    ) -> List[str]:
        """
        MMR variant using fused relevance scores and doc-doc similarity.
        Greedy selection: argmax λ*rel(d) - (1-λ)*max_{s in S} sim(d, s).
        """
        if not ranked_ids:
            return []

        # Normalize relevance scores to [0,1]
        rel = np.array([rel_scores.get(doc_id, 0.0) for doc_id in ranked_ids], dtype=float)
        if rel.size == 0:
            return []
        r_min, r_max = float(rel.min()), float(rel.max())
        if r_max > r_min:
            rel = (rel - r_min) / (r_max - r_min)
        else:
            rel = np.zeros_like(rel)

        # Fetch embeddings for all candidates (in the ranked order)
        embeddings: List[List[float]] = []
        for start in range(0, len(ranked_ids), self.cfg.max_get_batch):
            batch_ids = ranked_ids[start : start + self.cfg.max_get_batch]
            got = self._collection.get(ids=batch_ids, include=["embeddings"])
            # Keep order aligned with batch_ids
            id_to_emb = {i: e for i, e in zip(got.get("ids", []), got.get("embeddings", []))}
            embeddings.extend([id_to_emb[i] for i in batch_ids])

        E = np.array(embeddings, dtype=float)
        if E.ndim != 2 or E.shape[0] != len(ranked_ids):
            # Fallback: if embeddings missing, just return top-k by relevance
            return ranked_ids[:k]

        # Precompute doc-doc cosine similarity matrix
        sim = self._cosine_sim_matrix(E)

        selected: List[int] = []
        candidate_idxs: List[int] = list(range(len(ranked_ids)))

        # Start with the highest relevance
        first = int(np.argmax(rel))
        selected.append(first)
        candidate_idxs.remove(first)

        while len(selected) < min(k, len(ranked_ids)) and candidate_idxs:
            best_idx = None
            best_val = -1e9
            # For each candidate, compute diversity penalty to the already selected set
            for idx in candidate_idxs:
                penalty = 0.0
                if selected:
                    penalty = float(np.max(sim[idx, selected]))
                mmr_val = lambda_mult * float(rel[idx]) - (1.0 - lambda_mult) * penalty
                if mmr_val > best_val:
                    best_val = mmr_val
                    best_idx = idx
            selected.append(best_idx)
            candidate_idxs.remove(best_idx)

        return [ranked_ids[i] for i in selected]

    # ---- Helpers ----
    def _get_documents_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch documents + metadata for the given IDs, preserving order."""
        out: List[Dict[str, Any]] = []
        for start in range(0, len(ids), self.cfg.max_get_batch):
            batch_ids = ids[start : start + self.cfg.max_get_batch]
            got = self._collection.get(ids=batch_ids, include=["documents", "metadatas"])
            meta_map = {i: m for i, m in zip(got.get("ids", []), got.get("metadatas", []))}
            doc_map = {i: d for i, d in zip(got.get("ids", []), got.get("documents", []))}
            out.extend(
                [
                    {"id": i, "document": doc_map.get(i, ""), "metadata": meta_map.get(i, {})}
                    for i in batch_ids
                ]
            )
        return out


# ------------------------------ Manual run ---------------------------------- #
if __name__ == "__main__":
    cfg = RetrieverConfig()
    retriever = HybridRetrieverV4(cfg)
    demo_query = "How to build an emergency fund?"
    results = retriever.retrieve(demo_query, k=cfg.final_k)
    for r in results:
        print(f"[{r['id']}] score={r['score']:.5f}  meta={r['metadata']}")
        print(r["document"][:200].replace("\n", " "), "\n" + "-" * 80)
