# file: Retriever_Development/v4/cli_retrieve_v4.py
# -*- coding: utf-8 -*-

# How to Run:
# From Core Directory: python -m Retriever_Development.v4.cli_retrieve_v4 --q "emergency fund" -k 5 --format pretty --meta-keys chapter,position,category

"""
Command-line interface for HybridRetrieverV4.
- Runs hybrid retrieval (BM25s + Chroma -> RRF -> MMR) and prints results.
- No external deps beyond our project + stdlib.

Examples:
  # Single query, pretty print top-8
  python Retriever_Development/v4/cli_retrieve_v4.py --q "How to build an emergency fund?" -k 8

  # Multiple queries from file (one per line), JSONL output
  python Retriever_Development/v4/cli_retrieve_v4.py --q-file queries.txt --format json --k 10

  # Override fusion params and filter by metadata (JSON)
  python Retriever_Development/v4/cli_retrieve_v4.py --q "credit score" --rrf-k 60 --mmr-lambda 0.7 \
      --filters '{"chapter": "Chapter 2"}' --snippet 200

All comments are in English by project policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

# Import our retriever
from Retriever_Development.v4.hybrid_retriever_v4 import HybridRetrieverV4, RetrieverConfig


# ------------------------------- CLI utils --------------------------------- #

def _read_queries(q: Optional[str], q_file: Optional[str]) -> List[str]:
    """Normalize queries from --q and/or --q-file (one per line)."""
    queries: List[str] = []
    if q:
        queries.append(q.strip())
    if q_file:
        if not os.path.exists(q_file):
            raise FileNotFoundError(f"Query file not found: {q_file}")
        with open(q_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    queries.append(s)
    if not queries:
        raise ValueError("No query provided. Use --q or --q-file.")
    return queries


def _parse_filters(filters_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse --filters JSON safely."""
    if not filters_str:
        return None
    try:
        obj = json.loads(filters_str)
        if not isinstance(obj, dict):
            raise ValueError("Filters JSON must be an object, e.g. {\"chapter\": \"Intro\"}")
        return obj
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for --filters: {e}") from e


def _print_pretty(query: str, results: List[Dict[str, Any]], snippet: int, meta_keys: Optional[List[str]]) -> None:
    """Human-friendly printing."""
    print(f"\n=== Query: {query} ===")
    if not results:
        print("(no results)")
        return
    for rank, r in enumerate(results, start=1):
        rid = r.get("id", "")
        score = r.get("score", 0.0)
        meta = r.get("metadata", {}) or {}
        doc = r.get("document", "") or ""
        print(f"[{rank:02d}] id={rid}  score={score:.5f}")
        if meta_keys:
            # Print only requested meta keys if provided
            meta_view = {k: meta.get(k) for k in meta_keys}
        else:
            meta_view = meta
        if meta_view:
            try:
                print("     meta:", json.dumps(meta_view, ensure_ascii=False))
            except Exception:
                print("     meta:", str(meta_view))
        if snippet and doc:
            txt = doc.replace("\n", " ").strip()
            if len(txt) > snippet:
                txt = txt[:snippet - 1] + "…"
            print("     text:", txt)


def _print_json(query: str, results: List[Dict[str, Any]]) -> None:
    """Machine-friendly JSONL output (one line per hit)."""
    for r in results:
        payload = {
            "query": query,
            "id": r.get("id"),
            "score": r.get("score"),
            "metadata": r.get("metadata"),
            "document": r.get("document"),
        }
        print(json.dumps(payload, ensure_ascii=False))


def _print_ids(results: List[Dict[str, Any]]) -> None:
    """Only document IDs, space-separated on one line."""
    ids = [str(r.get("id")) for r in results]
    print(" ".join(ids))


# --------------------------------- Main ------------------------------------ #

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli_retrieve_v4",
        description="Hybrid retrieval (BM25s + Chroma -> RRF -> MMR) over finance_book_v4_cos.",
    )
    qg = p.add_argument_group("Query input")
    qg.add_argument("--q", type=str, help="Single query string.")
    qg.add_argument("--q-file", type=str, help="Path to a file with one query per line.")

    pg = p.add_argument_group("Retrieval parameters")
    pg.add_argument("-k", type=int, default=None, help="Final results to return. Default: config.final_k (10).")
    pg.add_argument("--candidates", type=int, default=None, help="Candidate pool per branch before fusion (default cfg).")
    pg.add_argument("--rrf-k", type=int, default=None, help="RRF constant k (default cfg=60).")
    pg.add_argument("--mmr-lambda", type=float, default=None, help="MMR lambda in [0,1] (default cfg=0.7).")
    pg.add_argument("--filters", type=str, default=None,
                    help='Metadata filter as JSON, e.g. {"chapter": "Chapter 2"}')

    og = p.add_argument_group("Output")
    og.add_argument("--format", choices=["pretty", "json", "ids"], default="pretty",
                    help="Output format: pretty (human), json (JSONL), ids (space-separated).")
    og.add_argument("--snippet", type=int, default=280, help="Max chars of text snippet in pretty mode (0 to disable).")
    og.add_argument("--meta-keys", type=str, default=None,
                    help="Comma-separated metadata keys to print in pretty mode, e.g. chapter,position,category")

    fg = p.add_argument_group("Advanced (rarely change)")
    fg.add_argument("--chroma-path", type=str, default=None, help="Override path to local Chroma store (default cfg).")
    fg.add_argument("--collection", type=str, default=None, help="Override Chroma collection name (default cfg).")

    # Placeholders for future switches; currently ignored by retriever skeleton
    fg.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking (if implemented).")
    fg.add_argument("--hyde", action="store_true", help="Enable HyDE synthetic query (if implemented).")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    # Prepare config overrides
    cfg = RetrieverConfig()
    if args.chroma_path:
        cfg.chroma_path = args.chroma_path
    if args.collection:
        cfg.chroma_collection = args.collection
    if args.rrf_k is not None:
        cfg.rrf_k = int(args.rrf_k)
    if args.candidates is not None:
        cfg.candidate_pool = int(args.candidates)
    if args.mmr_lambda is not None:
        # Clamp to [0,1]
        cfg.mmr_lambda = max(0.0, min(1.0, float(args.mmr_lambda)))
    if args.k is not None:
        cfg.final_k = int(args.k)

    # Build retriever once
    retriever = HybridRetrieverV4(cfg)

    # Queries
    queries = _read_queries(args.q, args.q_file)
    filters = _parse_filters(args.filters)

    # Output preferences
    meta_keys = [s.strip() for s in args.meta_keys.split(",")] if args.meta_keys else None

    for q in queries:
        results = retriever.retrieve(
            query=q,
            k=cfg.final_k,
            use_rerank=bool(args.rerank),
            use_hyde=bool(args.hyde),
            filters=filters,
        )
        if args.format == "pretty":
            _print_pretty(q, results, snippet=max(0, int(args.snippet)), meta_keys=meta_keys)
        elif args.format == "json":
            _print_json(q, results)
        else:
            _print_ids(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
