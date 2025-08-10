# ADR-001 — Embedding & Vector Store (v4)
**Date:** 2025-08-10  
**Owner:** AI Financial Assistant (RAG)  
**Status:** Accepted

## Context
We need a reproducible, local, low-friction embedding + vector store setup for the book “The 6-Step Personal Finance Reset”, to power hybrid retrieval (dense + sparse) and support future A/B tests.

**Corpus (v4):**
- Source dataset: `Data_Processing_and_Indexing/book_metadata_with_chapters_v4.json`
- Records: 172 chunks
- Chunking: 50–150 words, no overlap (coherent boundaries)
- Metadata per chunk: `chapter_title`, `chapter_number`, `position`, `source_id`, optional `category` (string) + boolean flags `has_<tag>` (for multi-labels)

## Decision
Use **Sentence-Transformers MiniLM** embeddings with **L2 normalization** and index in **Chroma HNSW (cosine)**.

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384d)
- **Normalization:** yes (L2) before indexing
- **Vector DB:** Chroma persistent client
- **Collection:** `finance_book_v4_cos`
- **HNSW space:** `cosine`
- **Persist dir:** `chroma_store`

## Rationale
- **Local & fast**: no external API costs; quick iterations.
- **Cosine + normalized vectors**: standard for SBERT-like models; more stable similarity than L2 without normalization.
- **Small corpus (172)**: MiniLM is sufficient to start; we’ll benchmark larger models before switching.
- **Clear metadata**: chapter mapping from DOCX headings enables explanations, filtering, and eval slices.

## Alternatives Considered
1) `thenlper/gte-large` (1024d) — higher quality, heavier; candidate for A/B.
2) `BAAI/bge-m3` (dense mode) — strong baseline; candidate for A/B.
3) OpenAI `text-embedding-3-large` — top quality via API; introduces cost and dependency.

## Retrieval Plan
- **Default:** dense (Chroma cosine) + **BM25** → **RRF fusion** (late fusion).
- **Rerank (opt.):** cross-encoder over top-30 for precision.
- **Diversification:** MMR or soft quotas to include concept + motivation + practical in the final context.
- **LLM prompt shape:** explain concept → motivate → give step-by-step plan (with chapter attributions).

## Quality & Evaluation
- **Metrics:** Recall@5, nDCG@10, response faithfulness (manual spot-check).
- **Eval set:** 20–30 user-like questions covering concept/motivation/practical.
- **Baselines to compare:** 
  - Dense (MiniLM, cosine)  
  - Hybrid (BM25 + dense via RRF)  
  - Hybrid + rerank  
  - (A/B) MiniLM vs GTE-Large / BGE-M3 / OpenAI

## Risks & Mitigations
- *Model ceiling (MiniLM):* may cap recall on nuanced queries → **A/B** with stronger embedders.
- *Metadata sparsity:* some chunks may lack `category` → rely on boolean flags + not filtering too hard; prefer diversification.
- *Heading detection noise:* DOCX-based mapping may miss rare cases → forward-fill + manual review of “Unknown”.
- *Index config drift:* HNSW space/params immutable post-create → **manifest + config** versioned (see below).

## Reproducibility
- **Config:** `configs/embedding_v4.json` (single source of truth)
- **Manifest:** `artifacts/v4/vectorstore_manifest.json` (what’s in the store now)
- **Scripts:**
  - Chapter enrichment:  
    `python Data_Processing_and_Indexing/chapters_enrich_v4.py --json "Data_Processing_and_Indexing/book_metadata_merged_fixed.json" --docx "Data_Processing_and_Indexing/Final - The_6-Step_Personal_Finance_Reset_6x9_my_hand_breaks.docx" --out "Data_Processing_and_Indexing/book_metadata_with_chapters_v4.json"`
  - Re-embed (cosine + L2):  
    `python Data_Processing_and_Indexing/reembed_chroma_cosine_v4.py --json "Data_Processing_and_Indexing/book_metadata_with_chapters_v4.json" --persist "chroma_store" --collection "finance_book_v4_cos" --model "sentence-transformers/all-MiniLM-L6-v2" --reset`
  - Probe query example:  
    `python Data_Processing_and_Indexing/query_chroma_v4.py --q "debt snowball vs avalanche" --n 5`

## Rollback
- Keep prior collection `finance_book_v4` (L2 default) alongside `finance_book_v4_cos`.
- To revert: point retriever to the older collection; embeddings can be re-built from the JSON using the config.

## Change Log
- **2025-08-10:** Initial acceptance — MiniLM + cosine + normalized; Chroma persistent collection `finance_book_v4_cos` created (172 vectors).
