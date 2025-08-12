# file: Retriever_Development/v4/evaluate_retriever_v4.py
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# HOW TO RUN (from project root):
#
# Default: evaluate top-10 on qrels_v4.jsonl
# python -m Retriever_Development.v4.evaluate_retriever_v4 --qrels configs/eval/qrels_v4.jsonl --k 10

# Override fusion/diversity params and also save CSV + JSON
# python -m Retriever_Development.v4.evaluate_retriever_v4 --qrels configs/eval/qrels_v4.jsonl --k 10 --rrf-k 60 --mmr-lambda 0.7 --candidates 40 --out-json artifacts/v4/runs/last_run.json --out-csv artifacts/v4/runs/last_run.csv

# Custom Chroma path / collection
# python -m Retriever_Development.v4.evaluate_retriever_v4 --qrels configs/eval/qrels_v4.jsonl --chroma-path ./chroma_store --collection finance_book_v4_cos

#
# Notes:
# - Qrels format (JSONL, one line per query):
#     {"query": "...", "relevant_ids": ["fb-v4c-0133", "fb-v4c-0873"], "filters": null}
# - All comments are in English by project policy.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

from Retriever_Development.v4.hybrid_retriever_v4 import HybridRetrieverV4, RetrieverConfig


# ------------------------------- Data types -------------------------------- #

@dataclass
class QrelItem:
    query: str
    relevant_ids: List[str]
    filters: Optional[Dict[str, Any]] = None


@dataclass
class PerQueryMetrics:
    query: str
    rel_count: int
    hit_count: int
    recall_at_k: float
    ndcg_at_k: float
    mrr_at_k: float
    first_rel_rank: Optional[int]
    retrieved_ids: List[str]


# ------------------------------- Utilities --------------------------------- #

def ensure_dir_for_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def read_qrels(path: str) -> List[QrelItem]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Qrels file not found: {path}")
    items: List[QrelItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            q = str(obj.get("query", "")).strip()
            rel = obj.get("relevant_ids", []) or []
            flt = obj.get("filters")
            items.append(QrelItem(query=q, relevant_ids=[str(x) for x in rel], filters=flt))
    if not items:
        raise ValueError("Qrels file is empty.")
    return items


def dcg_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """Binary gains DCG@k: sum(1/log2(1+rank)) for hits within top-k."""
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(1.0 + rank)
    return dcg


def idcg_at_k(rel_count: int, k: int) -> float:
    """Ideal DCG@k for binary gains given rel_count relevant docs total."""
    m = min(rel_count, k)
    return sum(1.0 / math.log2(1.0 + r) for r in range(1, m + 1))


def ndcg_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    ideal = idcg_at_k(len(relevant), k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / ideal


def recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / float(len(relevant))


def mrr_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / float(rank)
    return 0.0


def first_rel_rank(retrieved: List[str], relevant: set, k: int) -> Optional[int]:
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return rank
    return None


def percentile(values: List[int], p: float) -> float:
    """
    Simple percentile (p in [0,100]). Inclusive method:
    idx = ceil(p/100 * n) - 1 on 1-based ranks.
    """
    if not values:
        return float("nan")
    xs = sorted(values)
    n = len(xs)
    pos = max(1, math.ceil(p / 100.0 * n))
    return float(xs[pos - 1])


# ------------------------------- Evaluation -------------------------------- #

def evaluate(
    retriever: HybridRetrieverV4,
    qrels: List[QrelItem],
    k: int,
) -> Tuple[List[PerQueryMetrics], Dict[str, Any]]:
    per_query: List[PerQueryMetrics] = []
    first_hits: List[int] = []
    recalls: List[float] = []
    ndcgs: List[float] = []
    mrrs: List[float] = []

    for item in qrels:
        rel_set = set(item.relevant_ids)
        # Retrieve results
        res = retriever.retrieve(query=item.query, k=k, filters=item.filters)
        ids = [r.get("id") for r in res]

        r_at_k = recall_at_k(ids, rel_set, k)
        n_at_k = ndcg_at_k(ids, rel_set, k)
        mrr = mrr_at_k(ids, rel_set, k)
        first = first_rel_rank(ids, rel_set, k)

        per_query.append(
            PerQueryMetrics(
                query=item.query,
                rel_count=len(rel_set),
                hit_count=sum(1 for i in ids[:k] if i in rel_set),
                recall_at_k=r_at_k,
                ndcg_at_k=n_at_k,
                mrr_at_k=mrr,
                first_rel_rank=first,
                retrieved_ids=ids[:k],
            )
        )

        recalls.append(r_at_k)
        ndcgs.append(n_at_k)
        mrrs.append(mrr)
        if first is not None:
            first_hits.append(first)
        else:
            # If no hit within top-k, treat as k+1 for percentiles of first relevant
            first_hits.append(k + 1)

    summary = {
        "queries": len(qrels),
        "k": k,
        "recall_at_k_mean": mean(recalls) if recalls else 0.0,
        "ndcg_at_k_mean": mean(ndcgs) if ndcgs else 0.0,
        "mrr_at_k_mean": mean(mrrs) if mrrs else 0.0,
        "first_rel_rank_p50": percentile(first_hits, 50.0),
        "first_rel_rank_p95": percentile(first_hits, 95.0),
    }
    return per_query, summary


# ----------------------------------- CLI ----------------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate HybridRetrieverV4 on qrels JSONL.")
    p.add_argument("--qrels", type=str, required=True, help="Path to qrels JSONL file.")
    p.add_argument("--k", type=int, default=10, help="Top-K to evaluate (default: 10).")

    # Optional overrides for retriever config
    p.add_argument("--candidates", type=int, default=None, help="Candidate pool per branch before fusion.")
    p.add_argument("--rrf-k", type=int, default=None, help="RRF constant k (default cfg=60).")
    p.add_argument("--mmr-lambda", type=float, default=None, help="MMR lambda in [0,1] (default cfg=0.7).")
    p.add_argument("--chroma-path", type=str, default=None, help="Override Chroma path.")
    p.add_argument("--collection", type=str, default=None, help="Override Chroma collection name.")

    # Outputs
    p.add_argument("--out-json", type=str, default="artifacts/v4/runs/last_run.json",
                   help="Where to write summary JSON (default: artifacts/v4/runs/last_run.json).")
    p.add_argument("--out-csv", type=str, default=None, help="Optional CSV with per-query metrics.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    # Build retriever with overrides
    cfg = RetrieverConfig()
    if args.candidates is not None:
        cfg.candidate_pool = int(args.candidates)
    if args.rrf_k is not None:
        cfg.rrf_k = int(args.rrf_k)
    if args.mmr_lambda is not None:
        cfg.mmr_lambda = max(0.0, min(1.0, float(args.mmr_lambda)))
    if args.chroma_path:
        cfg.chroma_path = args.chroma_path
    if args.collection:
        cfg.chroma_collection = args.collection

    retriever = HybridRetrieverV4(cfg)

    # Load qrels
    qrels = read_qrels(args.qrels)

    # Evaluate
    per_query, summary = evaluate(retriever, qrels, k=int(args.k))

    # Compose run info
    run_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "chroma_path": os.path.abspath(cfg.chroma_path),
        "collection": cfg.chroma_collection,
        "collection_size": retriever._collection.count(),
        "params": {
            "k": cfg.final_k if cfg.final_k else args.k,
            "candidate_pool": cfg.candidate_pool,
            "rrf_k": cfg.rrf_k,
            "mmr_lambda": cfg.mmr_lambda,
        },
        "summary": summary,
    }

    # Write JSON summary
    ensure_dir_for_file(args.out_json)
    with open(args.out_json, "w", encoding="utf-8") as fjson:
        json.dump(run_info, fjson, ensure_ascii=False, indent=2)

    # Optional CSV per-query
    if args.out_csv:
        ensure_dir_for_file(args.out_csv)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                "query", "rel_count", "hit_count",
                "recall@k", "nDCG@k", "MRR@k", "first_rel_rank",
                "retrieved_ids",
            ])
            for item in per_query:
                writer.writerow([
                    item.query,
                    item.rel_count,
                    item.hit_count,
                    f"{item.recall_at_k:.6f}",
                    f"{item.ndcg_at_k:.6f}",
                    f"{item.mrr_at_k:.6f}",
                    item.first_rel_rank if item.first_rel_rank is not None else "",
                    " ".join(item.retrieved_ids),
                ])

    # Pretty print to console
    print("\n=== Evaluation Summary ===")
    print(f"Queries:               {summary['queries']}")
    print(f"Recall@{args.k} (mean): {summary['recall_at_k_mean']:.4f}")
    print(f"nDCG@{args.k}   (mean): {summary['ndcg_at_k_mean']:.4f}")
    print(f"MRR@{args.k}    (mean): {summary['mrr_at_k_mean']:.4f}")
    print(f"FirstRelRank P50:      {summary['first_rel_rank_p50']:.1f}")
    print(f"FirstRelRank P95:      {summary['first_rel_rank_p95']:.1f}")
    print(f"Wrote JSON: {os.path.abspath(args.out_json)}")
    if args.out_csv:
        print(f"Wrote CSV:  {os.path.abspath(args.out_csv)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
