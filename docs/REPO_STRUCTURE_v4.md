# Repository layout — v4 baseline (2025-08-12)

```text
financial-rag-assistant/
├─ Retriever_Development/
│  └─ v4/                           # Phase 3 (v4) retriever code
│     ├─ hybrid_retriever_v4.py     # BM25s + Chroma → RRF → MMR; public retrieve()
│     ├─ build_bm25_index_v4.py     # builds bm25s index from Chroma/JSON; saves to artifacts/v4/bm25_index/
│     ├─ cli_retrieve_v4.py         # CLI for manual probing (pretty/json/ids)
│     ├─ auto_make_qrels_v4.py      # auto "silver" qrels generator (intersection + RRF)
│     └─ evaluate_retriever_v4.py   # offline eval: Recall@K, nDCG@K, MRR@K; writes runs/last_run.*
│
├─ artifacts/
│  └─ v4/
│     ├─ bm25_index/                # persisted bm25s index (generated)
│     │  ├─ bm25_fb_v4.*            # multiple files created by bm25s.save(...)
│     │  └─ index_meta.json         # {"ids":[...], ...} aligned to Chroma IDs
│     └─ runs/                      # evaluation outputs (generated)
│        ├─ last_run.json           # summary (timestamp, params, metrics)
│        └─ last_run.csv            # per-query metrics (optional)
│
├─ chroma_store/                    # Chroma persistent DB (existing)
│  └─ (collection: finance_book_v4_cos)
│
├─ configs/
│  ├─ eval/
│  │  ├─ qrels_v4.jsonl             # auto-generated "silver" labels (query → relevant_ids)
│  │  └─ queries_v4.txt             # optional list of queries for auto qrels
│  └─ retriever_v1.json             # retriever config (extend with rrf_k/mmr_lambda/etc. if needed)
│
├─ Data_Processing_and_Indexing/
│  └─ book_metadata_with_chapters_v4.json   # v4 dataset (source for bm25/indexing)
│
├─ docs/
│  └─ REPO_STRUCTURE_v4.md          # this file with the tree snapshot
│
├─ Progress_Log_12.08.2025_14_43.md # latest progress log (this session’s snapshot)
└─ Progress_Log_10.08.2025_15_55.md # previous progress log
```

## Notes
- Dense store: `./chroma_store`, collection **finance_book_v4_cos**; IDs follow `fb-v4c-XXXX`.
- Hybrid: BM25s + Chroma → **RRF(k=60)** → **MMR(λ=0.7, final_k=10)**, `candidate_pool=40`.
- Generated folders/files: `artifacts/v4/bm25_index/*` и `artifacts/v4/runs/*` (обычно не коммитим; допустимо хранить `index_meta.json` и метрики).
