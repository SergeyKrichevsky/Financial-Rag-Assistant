# Data_Processing_and_Indexing/embed_chroma_v4.py
# -------------------------------------------------------------------
# How to run (PowerShell, from repo root):
#   python Data_Processing_and_Indexing\embed_chroma_v4.py --json "Data_Processing_and_Indexing\book_metadata_with_chapters_v4.json" --persist "chroma_store" --collection "finance_book_v4"
#
# First-time dependencies (once):
#   pip install "chromadb>=0.5.5" "sentence-transformers>=5.1.0"
# -------------------------------------------------------------------

import argparse, json, os, sys
from typing import List, Dict, Any

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to enriched JSON (array of objects with 'text' etc.)")
    ap.add_argument("--persist", default="chroma_store", help="Directory for Chroma PersistentClient")
    ap.add_argument("--collection", default="finance_book_v4", help="Chroma collection name")
    ap.add_argument("--batch", type=int, default=128, help="Batch size")
    ap.add_argument("--reset", action="store_true", help="Drop and recreate the collection if it exists")
    ap.add_argument("--probe", default="How to reset my personal finances step by step?", help="Probe query")
    return ap.parse_args()

def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        print("[ERR] Input JSON must be a non-empty array."); sys.exit(1)
    return data

def normalize_category(value):
    """Return (primary_category:str|None, flags:dict[str,bool]) from raw 'category' (list|str|None)."""
    flags = {}
    if isinstance(value, list):
        cleaned = [str(x).strip() for x in value if x is not None and str(x).strip()]
        if cleaned:
            for tag in cleaned:
                flags[f"has_{tag.lower()}"] = True
            return cleaned[0], flags
        return None, {}
    if isinstance(value, str):
        s = value.strip()
        if s:
            flags[f"has_{s.lower()}"] = True
            return s, flags
    return None, {}

def sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys with None and any non-scalar values; keep only str/int/float/bool."""
    out = {}
    for k, v in meta.items():
        if v is None:  # drop None entirely (Chroma doesn't accept None in metadata)
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        # everything else (list/dict/etc.) is dropped
    return out

def main():
    args = parse_args()
    if not os.path.exists(args.json):
        print(f"[ERR] JSON not found: {args.json}"); sys.exit(1)

    try:
        import chromadb
        from chromadb.utils import embedding_functions
        from chromadb.config import Settings
    except Exception:
        print("[ERR] Missing Chroma. Install with: pip install chromadb"); raise

    client = chromadb.PersistentClient(path=args.persist, settings=Settings())
    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"[OK] Dropped existing collection: {args.collection}")
        except Exception:
            pass

    st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=st_ef,
        metadata={"collection_version": "v4", "created_by": "embed_chroma_v4.py"}
    )
    print(f"[OK] Using collection: {args.collection} (persist: {os.path.abspath(args.persist)})")

    rows = load_json(args.json)
    print(f"[..] Loaded records: {len(rows)}")

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for i, r in enumerate(rows):
        txt = r.get("text", "")
        chapter_title  = r.get("chapter_title") or r.get("chapter") or "Unknown"
        chapter_number = r.get("chapter_number") or r.get("chapter_no") or 0
        raw_category   = r.get("category") or r.get("mode")
        category, cat_flags = normalize_category(raw_category)
        position       = r.get("position", i)
        source_id      = r.get("source_id") or "finance_book_v4"

        meta = {
            "chapter_title": chapter_title,
            "chapter_number": int(chapter_number) if chapter_number is not None else 0,
            "position": int(position) if position is not None else i,
            "source_id": source_id,
        }
        if category:  # only add if not None
            meta["category"] = category
        # add boolean flags (already True; scalars)
        meta.update(cat_flags)

        meta = sanitize_meta(meta)  # drop None & non-scalars

        texts.append(txt)
        metadatas.append(meta)
        ids.append(f"fb-v4-{i:04d}")

    total = 0
    b = args.batch
    for start in range(0, len(texts), b):
        end = min(start + b, len(texts))
        collection.add(ids=ids[start:end], documents=texts[start:end], metadatas=metadatas[start:end])
        total += (end - start)
        print(f"[OK] Added {end - start} (total {total})")

    results = collection.query(query_texts=[args.probe], n_results=3)
    print("\n[Probe] Query:", args.probe)
    for rank, (doc, meta, doc_id) in enumerate(
        zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0], results.get("ids", [[]])[0]), 1
    ):
        ch = meta.get("chapter_title")
        pos = meta.get("position")
        cat = meta.get("category")
        print(f"  {rank}. id={doc_id} | chapter='{ch}' | pos={pos} | category={cat}")
    print("\n✅ Done. You can now query the collection in your app.")

if __name__ == "__main__":
    main()
