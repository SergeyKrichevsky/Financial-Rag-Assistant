# file: Retriever_Development/v4/auto_make_qrels_v4.py
# -*- coding: utf-8 -*-

# How to run: python -m Retriever_Development.v4.auto_make_qrels_v4 --k-dense 20 --k-sparse 30 --rrf-k 60 --min-rel 3

"""
Auto-generate weak (silver) qrels for v4:
- For each query, retrieve dense (Chroma) and sparse (BM25s) candidates,
  fuse with RRF, and mark as relevant:
    1) IDs in the intersection of top-dense and top-sparse,
    2) if not enough, fill from top RRF until min_rel is reached.
- Writes JSONL to configs/eval/qrels_v4.jsonl:
    {"query": "...", "relevant_ids": ["fb-v4c-0133", ...], "filters": null}

Run (from project root):
    python -m Retriever_Development.v4.auto_make_qrels_v4 --k-dense 20 --k-sparse 30 --rrf-k 60 --min-rel 3

You can also provide your own queries file:
    python -m Retriever_Development.v4.auto_make_qrels_v4 --queries-file configs/eval/queries_v4.txt

All comments are in English by project policy.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any, Optional, Tuple

from Retriever_Development.v4.hybrid_retriever_v4 import HybridRetrieverV4, RetrieverConfig


# ------------------------------- Defaults ---------------------------------- #

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "configs", "eval", "qrels_v4.jsonl")
DEFAULT_QUERIES_FILE = os.path.join(PROJECT_ROOT, "configs", "eval", "queries_v4.txt")

# Fallback built-in queries if queries_v4.txt is missing
BUILTIN_QUERIES = [
    "emergency fund",
    "50/30/20 budgeting rule",
    "debt snowball vs avalanche",
    "how to improve credit score fast",
    "index funds for beginners",
    "what is a sinking fund",
    "Roth IRA contribution limits",
    "how often to rebalance portfolio",
    "health insurance deductible vs out-of-pocket max",
    "build credit history from scratch",
    "how big should my emergency buffer be",
    "payday budgeting tips",
    "high-yield savings vs CD",
    "how to avoid overdraft fees",
    "should I pay off debt or invest first",
    "how to set financial goals SMART",
    "how much to save for retirement by age",
    "what is dollar-cost averaging",
    "expense tracking methods",
    "how to negotiate a lower interest rate",
]


# ------------------------------- Helpers ----------------------------------- #

def read_queries(path: str) -> List[str]:
    """Read queries from file if present; otherwise use built-in list."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            qs = [ln.strip() for ln in f if ln.strip()]
        if qs:
            return qs
    return BUILTIN_QUERIES


def ensure_dir_for_file(fpath: str) -> None:
    os.makedirs(os.path.dirname(fpath), exist_ok=True)


def autolabel_for_query(
    retriever: HybridRetrieverV4,
    query: str,
    k_dense: int,
    k_sparse: int,
    rrf_k: int,
    min_rel: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Produce silver labels for a single query.
    Returns (relevant_ids, debug_info).
    """
    # 1) Get candidates from both branches
    dense = retriever._dense_search(query, top_k=k_dense, filters=None)     # [(id, distance)]
    sparse = retriever._sparse_search(query, top_k=k_sparse)                # [(id, score)]

    dense_ids = [i for i, _ in dense]
    sparse_ids = [i for i, _ in sparse]

    # 2) RRF fuse for fallback
    fused_scores = retriever._rrf_fuse(dense_results=dense, sparse_results=sparse, k=rrf_k)
    fused_sorted = [doc_id for doc_id, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]

    # 3) Primary silver labels: intersection (preserve dense order)
    intersection = [i for i in dense_ids if i in set(sparse_ids)]

    # 4) Ensure at least min_rel by topping up from fused
    rel_ids: List[str] = []
    for i in intersection:
        if i not in rel_ids:
            rel_ids.append(i)
        if len(rel_ids) >= min_rel:
            break
    if len(rel_ids) < min_rel:
        for i in fused_sorted:
            if i not in rel_ids:
                rel_ids.append(i)
            if len(rel_ids) >= min_rel:
                break

    debug = {
        "dense_top": dense_ids[:10],
        "sparse_top": sparse_ids[:10],
        "fused_top": fused_sorted[:10],
    }
    return rel_ids, debug


# --------------------------------- Main ------------------------------------ #

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auto-generate weak qrels for v4 using hybrid retrieval.")
    p.add_argument("--queries-file", type=str, default=DEFAULT_QUERIES_FILE,
                   help="Path to a file with one query per line. If missing, a built-in list is used.")
    p.add_argument("--out", type=str, default=DEFAULT_OUT,
                   help="Output JSONL path (default: configs/eval/qrels_v4.jsonl).")

    p.add_argument("--k-dense", type=int, default=20, help="Top-K from dense branch.")
    p.add_argument("--k-sparse", type=int, default=30, help="Top-K from sparse branch.")
    p.add_argument("--rrf-k", type=int, default=60, help="RRF constant k.")
    p.add_argument("--min-rel", type=int, default=3, help="Minimum number of relevant IDs per query.")

    # Allow overriding retriever config if needed
    p.add_argument("--chroma-path", type=str, default=None)
    p.add_argument("--collection", type=str, default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    # Build retriever with optional overrides
    cfg = RetrieverConfig()
    if args.chroma_path:
        cfg.chroma_path = args.chroma_path
    if args.collection:
        cfg.chroma_collection = args.collection
    retriever = HybridRetrieverV4(cfg)

    # Read queries
    queries = read_queries(args.queries_file)
    ensure_dir_for_file(args.out)

    # Generate and write JSONL
    written = 0
    with open(args.out, "w", encoding="utf-8") as fw:
        for q in queries:
            rel_ids, _dbg = autolabel_for_query(
                retriever=retriever,
                query=q,
                k_dense=args.k_dense,
                k_sparse=args.k_sparse,
                rrf_k=args.rrf_k,
                min_rel=args.min_rel,
            )
            line = {"query": q, "relevant_ids": rel_ids, "filters": None}
            fw.write(json.dumps(line, ensure_ascii=False) + "\n")
            written += 1

    print(f"[OK] Wrote {written} qrels to: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
