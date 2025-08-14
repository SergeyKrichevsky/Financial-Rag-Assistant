# file: llm_integration/retriever_bridge.py
# Adapter that converts the Phase 3 retriever output into:
# - context_text: compact text blob for the LLM
# - source_refs: normalized metadata for developers (hidden from end users)
# Parameterized by configs/rag_config.json via config_loader, including off-switch flags.

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

# Phase 3 retriever (adjust path if your project structure differs)
from Retriever_Development.v4.hybrid_retriever_v4 import (
    HybridRetrieverV4,
    RetrieverConfig,
)

# RAG config (retriever section)
from .config_loader import get_retriever_cfg

# ---- Load retriever config (with safe defaults) ----
_RCFG = get_retriever_cfg()  # {"k_default", "max_per_chapter", "exclude_chapters", "use_filters", "use_per_chapter_cap"}

K_DEFAULT: int = int(_RCFG.get("k_default", 5))
MAX_PER_CHAPTER: int = int(_RCFG.get("max_per_chapter", 2))
EXCLUDE_CHAPTERS = set(_RCFG.get("exclude_chapters", []))
USE_FILTERS: bool = bool(_RCFG.get("use_filters", True))
USE_PER_CHAPTER_CAP: bool = bool(_RCFG.get("use_per_chapter_cap", True))

@lru_cache(maxsize=1)
def _get_retriever(cfg: Optional[RetrieverConfig] = None) -> HybridRetrieverV4:
    """Create the retriever once and cache it (avoids reloading BM25/Chroma)."""
    return HybridRetrieverV4(cfg or RetrieverConfig())

def retrieve_context(question: str, k: Optional[int] = None) -> Tuple[str, List[Dict]]:
    """
    Run the hybrid retriever and return (context_text, source_refs).
    Post-filtering (configurable via rag_config.json):
      - drop non-informational sections (EXCLUDE_CHAPTERS) if USE_FILTERS = True,
      - deduplicate by document id,
      - cap max items per chapter if USE_PER_CHAPTER_CAP = True,
      - keep top-k items (k from arg or config),
      - build a compact plain-text context for the LLM.
    """
    retriever = _get_retriever()

    # Effective k and pull a bit more to allow filtering/diversity
    k_eff = int(k) if isinstance(k, int) and k > 0 else K_DEFAULT
    raw: List[Dict] = retriever.retrieve(question, k=max(k_eff * 3, k_eff))

    # 1) Drop excluded chapters and deduplicate by id (preserve order)
    seen_ids = set()
    prelim: List[Dict] = []
    for item in raw:
        meta = item.get("metadata") or {}
        chapter = (meta.get("chapter_title") or meta.get("chapter") or meta.get("chapter_name") or "").strip()
        doc_id = item.get("id")
        if USE_FILTERS and chapter in EXCLUDE_CHAPTERS:
            continue
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        prelim.append(item)

    # 2) Cap per chapter to improve diversity (preserve order)
    per_chapter_count: Dict[str, int] = {}
    filtered: List[Dict] = []
    for item in prelim:
        meta = item.get("metadata") or {}
        chapter = (meta.get("chapter_title") or meta.get("chapter") or meta.get("chapter_name") or "_unknown_").strip()
        cnt = per_chapter_count.get(chapter, 0)
        if USE_PER_CHAPTER_CAP and cnt >= MAX_PER_CHAPTER:
            continue
        per_chapter_count[chapter] = cnt + 1
        filtered.append(item)

    # 3) Keep top-k after filtering (preserve order)
    filtered = filtered[:k_eff]

    # 4) Build compact context for the LLM (plain text only)
    texts: List[str] = []
    for item in filtered:
        txt = (item.get("document") or "").strip()
        if txt:
            texts.append(txt)
    context_text = "\n\n".join(texts)

    # 5) Developer-facing references (safe scalar fields only)
    refs: List[Dict] = []
    for item in filtered:
        meta = item.get("metadata") or {}
        refs.append({
            "id": item.get("id"),
            "score": float(item.get("score", 0.0)),
            "chapter": (meta.get("chapter_title") or meta.get("chapter") or meta.get("chapter_name")),
            "position": meta.get("position"),
            "category": meta.get("category"),
            "source_id": meta.get("source_id"),
            "preview": (item.get("document") or "").strip()[:200],
        })

    return context_text, refs
