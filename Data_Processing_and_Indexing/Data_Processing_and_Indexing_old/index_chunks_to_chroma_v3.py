# Data_Processing_and_Indexing/index_chunks_to_chroma_v3.py
"""
Index chunks from the fixed JSON (v3) into a Chroma collection, with safe metadata sanitization.

- Reads: Data_Processing_and_Indexing/finance_book_chunks_enriched.fixed.v3.json
- Writes to: ./chroma_store (PersistentClient, new API)
- Collection name: finance_book_v2
- Metadata: strictly primitives (str|int|float|bool) — no None

Deps:
    pip install "chromadb>=0.5.0" sentence-transformers
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

# ========== Config ==========
# Fixed input JSON (same folder as this script)
JSON_PATH = Path(__file__).resolve().parent / "finance_book_chunks_enriched.fixed.v3.json"

# Chroma persistent folder (repo root / chroma_store)
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_store"

# Collection name to use in Chroma
COLLECTION_NAME = "finance_book_v3"

# If True, drop existing collection before indexing
RESET_COLLECTION = False

# Embedding model and batch size
EMBED_MODEL = "all-MiniLM-L6-v2"  # 384-d embeddings
BATCH_SIZE = 64


# ========== Helpers ==========

def extract_chapter_num(label: str) -> int:
    """Parse 'Chapter N' -> N, else 0."""
    m = re.search(r"Chapter\s+(\d+)", label or "", flags=re.I)
    return int(m.group(1)) if m else 0


def is_primitive(v: Any) -> bool:
    """Check if value is a Chroma-safe primitive."""
    return isinstance(v, (str, int, float, bool))


def to_int_or_default(v: Any, default: int = -1) -> int:
    """Best-effort cast to int, else return default."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure metadata contains only primitives (str|int|float|bool) and no None.
    - Numeric fields coerced to int with defaults.
    - None converted to safe defaults.
    - Non-primitive values stringified.
    """
    numeric_keys = {"order", "tokens", "char_start", "char_end", "chapter_num"}
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            # Use numeric -1 for known numeric keys; empty string otherwise.
            clean[k] = -1 if k in numeric_keys else ""
            continue

        if k in numeric_keys:
            clean[k] = to_int_or_default(v, -1)
            continue

        if is_primitive(v):
            clean[k] = v
        else:
            # Fallback: stringify complex types (lists/dicts/etc.)
            clean[k] = str(v)
    return clean


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    """Load chunks from JSON and keep stable ordering by 'order' then numeric id."""
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    def id_num(x: Dict[str, Any]) -> int:
        try:
            return int(str(x.get("id", "")).split("_")[1])
        except Exception:
            return 10**9

    data.sort(key=lambda x: (x.get("order", id_num(x))))
    return data


def main() -> None:
    print(f"[Input] {JSON_PATH}")
    print(f"[Chroma dir] {CHROMA_DIR}")
    print(f"[Collection] {COLLECTION_NAME}")

    # -------- Load --------
    chunks = load_chunks(JSON_PATH)
    print(f"[Load] {len(chunks)} chunks")

    # -------- Embedder --------
    # all-MiniLM-L6-v2 -> 384-dim sentence embeddings (fast & solid baseline)
    model = SentenceTransformer(EMBED_MODEL)

    # -------- Chroma client --------
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # (Re)create collection if needed
    if RESET_COLLECTION:
        existing = {c.name for c in client.list_collections()}
        if COLLECTION_NAME in existing:
            print(f"[Drop] removing existing collection '{COLLECTION_NAME}'")
            client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # -------- Prepare arrays --------
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for idx, ch in enumerate(chunks, start=1):
        cid = str(ch.get("id") or f"chunk_{idx}")
        text = ch.get("text") or ""
        chapter_label = ch.get("chapter", "")

        raw_meta = {
            "chapter": chapter_label,
            "order": ch.get("order"),
            "tokens": ch.get("tokens"),
            "char_start": ch.get("char_start"),
            "char_end": ch.get("char_end"),
            "chapter_num": extract_chapter_num(chapter_label),
            "source_json": JSON_PATH.name,
        }
        meta = sanitize_meta(raw_meta)

        ids.append(cid)
        docs.append(text)
        metas.append(meta)

    # -------- Batch embed + add --------
    total = len(docs)
    for i in range(0, total, BATCH_SIZE):
        j = min(i + BATCH_SIZE, total)
        batch_texts = docs[i:j]

        # Encode to Python lists (not tensors); Chroma expects list[list[float]]
        vecs = model.encode(batch_texts, convert_to_tensor=False)
        # SentenceTransformer may return numpy arrays; .tolist() makes them JSON-serializable
        vecs = vecs.tolist() if hasattr(vecs, "tolist") else vecs

        # Add batch (docs + embeddings + sanitized metadatas)
        collection.add(
            ids=ids[i:j],
            documents=batch_texts,
            metadatas=metas[i:j],
            embeddings=vecs,
        )
        print(f"[Index] {j}/{total}")

    print(f"✅ Done. Items in '{COLLECTION_NAME}': {collection.count()}")


if __name__ == "__main__":
    main()
