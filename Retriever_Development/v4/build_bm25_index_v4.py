# file: Retriever_Development/v4/build_bm25_index_v4.py
# -*- coding: utf-8 -*-
"""
Build BM25S index for v4 corpus and save it under artifacts/v4/bm25_index/.
By default it pulls (id, document) pairs from the existing Chroma collection
'finance_book_v4_cos' to guarantee ID alignment; alternatively it can read from
a JSON file if you pass --source json.

CLI:
    # Default: build from Chroma collection
    python Retriever_Development/v4/build_bm25_index_v4.py

    # Build from JSON
    python Retriever_Development/v4/build_bm25_index_v4.py \
        --source json --json-path Data_Processing_and_Indexing/book_metadata_with_chapters_v4.json

Requirements:
    pip install bm25s>=0.2 chromadb>=0.5
    # optional (slightly better lexical quality):
    pip install PyStemmer

What it writes:
    artifacts/v4/bm25_index/
        bm25_fb_v4.*          # BM25S serialized arrays (created by bm25s.save)
        index_meta.json       # {"ids": [...], "source": "...", ...}

All comments are in English by project policy.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple, Dict, Any

import numpy as np
import bm25s

# We import chromadb lazily (only if --source chroma), to keep JSON mode light.
try:
    import chromadb
    from chromadb.config import Settings
except Exception:  # pragma: no cover - optional dep in JSON mode
    chromadb = None
    Settings = None


# ------------------------------- Defaults ---------------------------------- #

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_store")
DEFAULT_COLLECTION = "finance_book_v4_cos"

OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "v4", "bm25_index")
DEFAULT_INDEX_BASENAME = "bm25_fb_v4"
META_FILENAME = "index_meta.json"


# ------------------------------- Utils ------------------------------------- #

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_meta(ids: List[str], out_dir: str, source: str, extra: Dict[str, Any]) -> None:
    meta_path = os.path.join(out_dir, META_FILENAME)
    payload = {"ids": ids, "source": source}
    payload.update(extra or {})
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_from_chroma(chroma_path: str, collection_name: str) -> Tuple[List[str], List[str]]:
    """
    Iterate the existing Chroma collection and return parallel lists (ids, docs).
    NOTE: Do NOT include "ids" in `include` — Chroma returns IDs by default.
    """
    if chromadb is None:
        raise RuntimeError("chromadb is not installed. Please `pip install chromadb` to use --source chroma.")
    client = chromadb.PersistentClient(path=os.path.abspath(chroma_path), settings=Settings())
    col = client.get_collection(collection_name)
    total = col.count()

    ids: List[str] = []
    docs: List[str] = []

    offset = 0
    batch = 512
    while offset < total:
        # VALID includes: "documents", "embeddings", "metadatas", "distances"
        got = col.get(include=["documents"], limit=batch, offset=offset)
        got_ids = got.get("ids", [])
        got_docs = got.get("documents", [])

        for i, d in zip(got_ids, got_docs):
            if d is None or str(d).strip() == "":
                continue
            ids.append(str(i))
            docs.append(str(d))
        offset += batch

    if not ids:
        raise RuntimeError(f"No documents found in Chroma collection '{collection_name}' at {chroma_path}")
    return ids, docs



def load_from_json(json_path: str) -> Tuple[List[str], List[str]]:
    """
    Load (ids, docs) from a JSON file.
    The JSON is expected to be a list[dict]. We try common keys for text: 'text', 'document', 'content'.
    For id: 'id' or enumerated fallback.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON must contain a list of records")

    ids: List[str] = []
    docs: List[str] = []
    for idx, rec in enumerate(data):
        if not isinstance(rec, dict):
            continue
        # Text extraction with fallbacks
        text = rec.get("text") or rec.get("document") or rec.get("content") or rec.get("chunk") or ""
        text = str(text).strip()
        if not text:
            continue
        doc_id = rec.get("id")
        if doc_id is None:
            doc_id = f"json-{idx}"
        ids.append(str(doc_id))
        docs.append(text)

    if not ids:
        raise RuntimeError("No usable records found in JSON (missing 'text'/'document'/'content').")
    return ids, docs


def tokenize_corpus(docs: List[str], stemming: bool) -> Any:
    """
    Tokenize the corpus using bm25s.tokenize. If stemming=True and PyStemmer is available,
    we apply English stemming. Otherwise we default to stopwords='en' only.
    """
    stemmer = None
    if stemming:
        try:
            import Stemmer  # type: ignore
            stemmer = Stemmer.Stemmer("english")
        except Exception:
            stemmer = None  # silently fall back
    # Returns integer token IDs by default, which is both faster and more compact
    return bm25s.tokenize(docs, stopwords="en", stemmer=stemmer)


# ------------------------------- Main build -------------------------------- #

def build_index(
    source: str,
    chroma_path: str,
    collection_name: str,
    json_path: str,
    out_dir: str,
    index_basename: str,
    method: str,
    k1: float,
    b: float,
    delta: float,
    stemming: bool,
) -> None:
    # 1) Load corpus
    if source == "chroma":
        ids, docs = load_from_chroma(chroma_path, collection_name)
    else:
        ids, docs = load_from_json(json_path)

    # 2) Tokenize
    corpus_tokens = tokenize_corpus(docs, stemming=stemming)

    # 3) Build BM25S retriever with chosen method/hyperparams
    #    - 'lucene' is the bm25s default; 'robertson' matches classic Okapi formulation.
    retriever = bm25s.BM25(method=method, k1=k1, b=b, delta=delta)

    # 4) Index corpus (token IDs)
    retriever.index(corpus_tokens)

    # 5) Persist index and aligned ID mapping
    ensure_dir(out_dir)
    basepath = os.path.join(out_dir, index_basename)
    # Save arrays + (optionally) the raw corpus for debugging
    retriever.save(basepath, corpus=docs)

    extra = {
        "collection": collection_name if source == "chroma" else None,
        "doc_count": len(ids),
        "method": method,
        "k1": float(k1),
        "b": float(b),
        "delta": float(delta),
        "stemming": bool(stemming),
    }
    save_meta(ids=ids, out_dir=out_dir, source=source, extra=extra)

    print(f"[OK] Saved BM25S index to: {basepath}* and meta to: {os.path.join(out_dir, META_FILENAME)}")
    print(f"      Documents indexed: {len(ids)} | method={method} k1={k1} b={b} delta={delta} stemming={stemming}")
    if source == "chroma":
        print(f"      Chroma path: {os.path.abspath(chroma_path)} | collection: {collection_name}")
    else:
        print(f"      JSON path: {os.path.abspath(json_path)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BM25S index for v4 corpus.")
    p.add_argument("--source", choices=["chroma", "json"], default="chroma",
                   help="Where to read documents from. Default: chroma.")
    p.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH, help="Path to local Chroma store.")
    p.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name.")
    p.add_argument("--json-path", default=os.path.join(PROJECT_ROOT, "Data_Processing_and_Indexing", "book_metadata_with_chapters_v4.json"),
                   help="Path to JSON with records (only used when --source json).")

    p.add_argument("--out-dir", default=OUT_DIR, help="Output directory for BM25S index.")
    p.add_argument("--index-name", default=DEFAULT_INDEX_BASENAME, help="Basename for saved index files.")

    p.add_argument("--method", default="lucene",
                   choices=["lucene", "robertson", "atire", "bm25l", "bm25+"],
                   help="BM25 variant (bm25s). Default: lucene.")
    p.add_argument("--k1", type=float, default=1.2, help="BM25 k1 parameter (1.2-2.0 typical).")
    p.add_argument("--b", type=float, default=0.75, help="BM25 b parameter (0-1, default 0.75).")
    p.add_argument("--delta", type=float, default=0.5, help="Delta for BM25+ / BM25L.")
    p.add_argument("--stemming", action="store_true", help="Enable English stemming if PyStemmer is available.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(
        source=args.source,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        json_path=args.json_path,
        out_dir=args.out_dir,
        index_basename=args.index_name,
        method=args.method,
        k1=args.k1,
        b=args.b,
        delta=args.delta,
        stemming=args.stemming,
    )
