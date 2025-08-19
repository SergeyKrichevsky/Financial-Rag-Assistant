# Repository Structure — Phase 4–5 Snapshot (2025-08-17)

```text
financial-rag-assistant/
├── Data_Processing_and_Indexing/          # Raw book processing, enrichment, embeddings
│   ├── Archive/
│   ├── Data_Processing_and_Indexing_old/
│   ├── book_marked.txt
│   ├── book_metadata_merged_fixed.json
│   ├── book_metadata_with_chapters_v4.json
│   ├── chapters_enrich_v4.py
│   ├── embed_chroma_v4.py
│   ├── query_chroma_v4.py
│   ├── reembed_chroma_cosine_v4.py
│   ├── sanity_hist_words.png
│   ├── Final - The_6-Step_Personal_Finance_Reset-Book.docx
│   ├── Final - The_6-Step_Personal_Finance_Reset-Book.pdf
│   ├── Prompt to process data.txt
│   └── Prompt to process data 2.txt
│
├── Retriever_Development/                  # Retriever logic: BM25, Dense, RRF, MMR (v3 + v4)
│   ├── bm25_test.py
│   ├── bm25_test_v3.py
│   ├── eval_retriever_v3.py
│   ├── hybrid_rff_test.py
│   ├── hybrid_rff_test_v3.py
│   └── v4/
│       ├── auto_make_qrels_v4.py
│       ├── build_bm25_index_v4.py
│       ├── cli_retrieve_v4.py
│       ├── evaluate_retriever_v4.py
│       ├── hybrid_retriever_v4.py
│       ├── tests/
│       └── __pycache__/
│
├── llm_integration/                        # Phase 4: LLM-based generation pipeline
│   ├── answer_generator.py
│   ├── answer_generator_1.py
│   ├── answer_generator_2.py
│   ├── answer_generator_3_before_errors_undling.py
│   ├── answer_generator_4_before_config_parameters.py
│   ├── answer_generator_5_before_off-switch_filters.py
│   ├── cli_ask.py
│   ├── config_loader.py
│   ├── llm_router.py
│   ├── model.config.json
│   ├── retriever_bridge.py
│   ├── retriever_bridge_1.py
│   ├── retriever_bridge_2_before_off-switch_filters.py
│   ├── run_logger.py
│   ├── run_logger_1.py
│   ├── smoke_test_openai.py
│   ├── test_generate.py
│   └── __pycache__/
│
├── Interface/                  # Phase 5: User Interface (Streamlit app)
│   ├── app.py                  # Main UI entry point
│   └── requirements.txt        
│
├── artifacts/
│   └── v4/
│       ├── bm25_index/                     # Saved sparse index
│       │   ├── bm25_fb_v4
│       │   └── index_meta.json
│       ├── runs/                           # Logging output from generation
│       │   ├── last_run.json
│       │   ├── last_run_phase4.json
│       │   └── runs_history.jsonl
│       └── vectorstore_manifest.json
│
├── chroma_store/                           # Chroma vector DB (MiniLM, cosine, persisted)
│   ├── chroma.sqlite3
│   └── <uuid-based folders>
│
├── configs/
│   ├── eval/
│   │   └── qrels_v4.jsonl
│   ├── embedding_v4.json
│   ├── rag_config.json
│   ├── rag_config_1_before_off-switch_filters.json
│   ├── retriever_v1.json
│   ├── system_prompt.txt
│   ├── system_prompt_assistant_API.txt
│   └── models.ui.json             # Phase 5: UI models catalog (strict; required by Interface/app.py)

│
├── docs/
│   ├── HANDOFF/
│   │   ├── ENV_SETUP.md
│   │   ├── TECHNICAL_ASSIGNMENT.md
│   │   └── REPO_STRUCTURE.md  
│   └── decisions/
│       └── ADR-001-embedding-v4.md
