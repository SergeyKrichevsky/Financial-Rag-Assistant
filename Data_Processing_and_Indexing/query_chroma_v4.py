# Data_Processing_and_Indexing/query_chroma_v4.py
# -------------------------------------------------------------------
# How to run (PowerShell, from repo root):
#   # range + category + boolean tag (AND)
#   python Data_Processing_and_Indexing\query_chroma_v4.py --q "how to build an emergency fund" --n 5 --category PRACTICAL --chapter-min 8 --chapter-max 12 --has PRACTICAL
#
#   # exact chapter
#   python Data_Processing_and_Indexing\query_chroma_v4.py --q "budgeting steps" --chapter 3
#
#   # no filters
#   python Data_Processing_and_Indexing\query_chroma_v4.py --q "debt snowball vs avalanche" --n 5
# -------------------------------------------------------------------

import argparse, os
from typing import Dict, Any, List

def build_where(args) -> Dict[str, Any]:
    """
    Build a Chroma 'where' filter.
    IMPORTANT: Chroma allows only ONE operator per field; to combine conditions use $and/$or.
    """
    clauses: List[Dict[str, Any]] = []

    # chapter filter
    if args.chapter is not None:
        clauses.append({"chapter_number": int(args.chapter)})
    else:
        if args.chapter_min is not None:
            clauses.append({"chapter_number": {"$gte": int(args.chapter_min)}})
        if args.chapter_max is not None:
            clauses.append({"chapter_number": {"$lte": int(args.chapter_max)}})

    # category (primary string)
    if args.category:
        clauses.append({"category": {"$in": [args.category]}})

    # boolean flags we added at indexing time (e.g., has_practical)
    if args.has:
        for tag in args.has:
            clauses.append({f"has_{tag.lower()}": True})

    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Query text")
    ap.add_argument("--n", type=int, default=5, help="n_results")
    ap.add_argument("--persist", default="chroma_store", help="Chroma persist dir")
    ap.add_argument("--collection", default="finance_book_v4", help="Collection name")
    ap.add_argument("--chapter", type=int, help="Exact chapter number")
    ap.add_argument("--chapter-min", type=int, help="Min chapter number (inclusive)")
    ap.add_argument("--chapter-max", type=int, help="Max chapter number (inclusive)")
    ap.add_argument("--category", type=str, help="Category string (e.g., PRACTICAL, MOTIVATION)")
    ap.add_argument("--has", nargs="*", help="Boolean tags (e.g., PRACTICAL MOTIVATION)")
    args = ap.parse_args()

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(path=args.persist, settings=Settings())
    col = client.get_collection(args.collection)

    where = build_where(args)
    print(f"[Query] {args.q}")
    if where:
        print(f"[Filter] where={where}")

    res = col.query(query_texts=[args.q], n_results=args.n, where=where)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]

    if not docs:
        print("No results.")
        return

    for i, (doc, meta, _id) in enumerate(zip(docs, metas, ids), 1):
        ch = meta.get("chapter_title")
        num = meta.get("chapter_number")
        cat = meta.get("category")
        pos = meta.get("position")
        print(f"\n{i}. id={_id} | chapter {num}: {ch} | category={cat} | pos={pos}")
        preview = (doc[:240] + "…") if len(doc) > 240 else doc
        print(preview)

if __name__ == "__main__":
    main()
