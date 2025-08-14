# file: llm_integration/retriever_bridge.py
# Adapter that converts the Phase 3 retriever output into:
# - context_text: compact text blob for the LLM
# - source_refs: normalized metadata for developers (hidden from end users)

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

# Import the hybrid retriever class from Phase 3
from Retriever_Development.v4.hybrid_retriever_v4 import (
    HybridRetrieverV4,
    RetrieverConfig,
)

@lru_cache(maxsize=1)
def _get_retriever(cfg: Optional[RetrieverConfig] = None) -> HybridRetrieverV4:
    """Create the retriever once and cache it (avoids reloading BM25/Chroma)."""
    return HybridRetrieverV4(cfg or RetrieverConfig())

def retrieve_context(question: str, k: int = 10) -> Tuple[str, List[Dict]]:
    """
    Run the hybrid retriever and return (context_text, source_refs).
    Post-filtering:
      - drop non-informational sections (e.g., "About the Author", "Final Words"),
      - deduplicate by id,
      - cap max items per chapter to improve diversity,
      - build a compact plain-text context for the LLM.
    """
    retriever = _get_retriever()

    EXCLUDE_CHAPTERS = {
        "About the Author",
        "Final Words",
        "Acknowledgments",
        "Table of Contents",
        "Foreword",
        "Preface",
        "Index",
    }
    MAX_PER_CHAPTER = 2  # keep at most 2 chunks from the same chapter

    # Pull a bit more to allow filtering/diversity
    results: List[Dict] = retriever.retrieve(question, k=max(k * 3, k))

    # 1) Drop excluded chapters and deduplicate by id
    seen_ids = set()
    prelim: List[Dict] = []
    for item in results:
        meta = item.get("metadata") or {}
        chapter = (meta.get("chapter_title") or "").strip()
        doc_id = item.get("id")
        if chapter in EXCLUDE_CHAPTERS:
            continue
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        prelim.append(item)

    # 2) Cap per chapter while preserving original order
    per_chapter_count: Dict[str, int] = {}
    filtered: List[Dict] = []
    for item in prelim:
        meta = item.get("metadata") or {}
        chapter = (meta.get("chapter_title") or "").strip() or "_unknown_"
        cnt = per_chapter_count.get(chapter, 0)
        if cnt >= MAX_PER_CHAPTER:
            continue
        per_chapter_count[chapter] = cnt + 1
        filtered.append(item)

    # 3) Keep top-k after filtering (preserve order)
    filtered = filtered[:k]

    # 4) Build compact context for the LLM (plain text only)
    texts: List[str] = []
    for item in filtered:
        txt = (item.get("document") or "").strip()
        if txt:
            texts.append(txt)
    context_text = "\n\n".join(texts)

    # 5) Developer-facing references
    refs: List[Dict] = []
    for item in filtered:
        meta = item.get("metadata") or {}
        refs.append({
            "id": item.get("id"),
            "score": float(item.get("score", 0.0)),
            "chapter": meta.get("chapter_title") or meta.get("chapter") or meta.get("chapter_name"),
            "position": meta.get("position"),
            "category": meta.get("category"),
            "source_id": meta.get("source_id"),
            "preview": (item.get("document") or "").strip()[:200],
        })

    return context_text, refs

