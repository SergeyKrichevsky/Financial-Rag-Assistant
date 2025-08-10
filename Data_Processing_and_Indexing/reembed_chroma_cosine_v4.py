# Data_Processing_and_Indexing/reembed_chroma_cosine_v4.py
# -------------------------------------------------------------------
# How to run (PowerShell, from repo root):
#
#   python Data_Processing_and_Indexing\reembed_chroma_cosine_v4.py --json "Data_Processing_and_Indexing\book_metadata_with_chapters_v4.json" --persist "chroma_store" --collection "finance_book_v4_cos" --model "sentence-transformers/all-MiniLM-L6-v2" --reset
#
#   python Data_Processing_and_Indexing\reembed_chroma_cosine_v4.py ^
#     --json "Data_Processing_and_Indexing\book_metadata_with_chapters_v4.json" ^
#     --persist "chroma_store" ^
#     --collection "finance_book_v4_cos" ^
#     --model "sentence-transformers/all-MiniLM-L6-v2" ^
#     --reset
#
# Что делает:
# - Грузит JSON (172 чанков)
# - Считает эмбеддинги локально через SentenceTransformers с L2-нормализацией
# - Создаёт НОВУЮ коллекцию с hnsw:space="cosine" и пишет туда (batch)
# - Пробный запрос для валидации
# -------------------------------------------------------------------

import argparse, json, os, sys
from typing import List, Dict, Any

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--persist", default="chroma_store")
    ap.add_argument("--collection", default="finance_book_v4_cos")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--probe", default="How to reset my personal finances step by step?")
    return ap.parse_args()

def load_rows(p: str) -> List[Dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        print("[ERR] JSON must be a non-empty array"); sys.exit(1)
    return data

def to_scalar_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if v is None: 
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
    return out

def main():
    args = parse_args()
    if not os.path.exists(args.json):
        print(f"[ERR] JSON not found: {args.json}"); sys.exit(1)

    # --- embed locally with L2 normalization
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print('[ERR] Install sentence-transformers: pip install "sentence-transformers>=5.1.0"'); raise

    model = SentenceTransformer(args.model)
    print(f"[OK] Model loaded: {args.model}")

    rows = load_rows(args.json)
    texts, metas, ids = [], [], []
    for i, r in enumerate(rows):
        txt = r.get("text", "")
        chapter_title  = r.get("chapter_title") or r.get("chapter") or "Unknown"
        chapter_number = int(r.get("chapter_number") or 0)
        position       = int(r.get("position") if r.get("position") is not None else i)
        source_id      = r.get("source_id") or "finance_book_v4"
        category       = r.get("category") if isinstance(r.get("category"), str) else None

        m = {"chapter_title": chapter_title, "chapter_number": chapter_number,
             "position": position, "source_id": source_id}
        if category: m["category"] = category
        metas.append(to_scalar_meta(m))
        texts.append(txt)
        ids.append(f"fb-v4c-{i:04d}")

    # batch encode with normalization
    import math
    embeddings: List[List[float]] = []
    B = args.batch
    for s in range(0, len(texts), B):
        e = min(s+B, len(texts))
        vecs = model.encode(texts[s:e], batch_size=min(64, B), normalize_embeddings=True, convert_to_numpy=True)
        embeddings.extend(v.tolist() for v in vecs)
        print(f"[EMB] {e-s} -> total {e}")

    # --- write to Chroma with cosine space
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception:
        print("[ERR] Install chromadb: pip install chromadb>=0.5.5"); raise

    client = chromadb.PersistentClient(path=args.persist, settings=Settings())

    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"[OK] Dropped existing: {args.collection}")
        except Exception:
            pass

    # Важно: задаём косинусную метрику на этапе создания
    coll = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine", "collection_version": "v4-cos"},
    )
    print(f"[OK] Using collection: {args.collection} (space=cosine)")

    # заливаем батчами
    total = 0
    for s in range(0, len(ids), B):
        e = min(s+B, len(ids))
        coll.add(ids=ids[s:e], embeddings=embeddings[s:e], metadatas=metas[s:e], documents=texts[s:e])
        total += (e - s)
        print(f"[OK] Added {e-s} (total {total})")

    # quick probe
    res = coll.query(query_texts=[args.probe], n_results=3)
    print("\n[Probe]", args.probe)
    for rank, (doc, meta, _id) in enumerate(
        zip(res.get("documents", [[]])[0], res.get("metadatas", [[]])[0], res.get("ids", [[]])[0]), 1
    ):
        print(f"  {rank}. id={_id} | ch={meta.get('chapter_title')} | pos={meta.get('position')} | cat={meta.get('category')}")
    print("\n✅ Re-embed done (cosine + L2-normalized).")

if __name__ == "__main__":
    main()
