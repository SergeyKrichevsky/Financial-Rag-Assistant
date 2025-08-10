# 📊 Progress Log — AI Financial Assistant (RAG-Based)

> Goal: Build a RAG system that answers user questions based on the book *The 6-Step Personal Finance Reset*.

---

## ✅ Phase 1: Knowledge Base Preparation — **Completed**

- [x] 📘 **Created original source material** — *"The 6-Step Personal Finance Reset"*, developed specifically for this project as both:  
  1. A standalone commercial product.  
  2. An AI-ready structured knowledge base for RAG.  
  **Now available for purchase:** [Kindle eBook](https://kdp.amazon.com/amazon-dp-action/us/dualbookshelf.marketplacelink/B0FLVT9LQV) · [Paperback](https://kdp.amazon.com/amazon-dp-action/us/dualbookshelf.marketplacelink/B0FLWDCLF3)


  Subtasks:
  - [x] 🔍 Conducted market research to identify audience needs, niche opportunities, and bestseller potential.
  - [x] 🛠 Designed the book’s dual purpose from the outset — monetizable asset + structured AI dataset.
  - [x] 📑 Created a logical content structure — chapters, subchapters, checklists, frameworks — optimized for future semantic chunking.
  - [x] ✍ Used AI-assisted writing techniques, followed by human editing and restructuring for clarity and educational impact.
  - [x] 🧠 Applied principles of behavioral psychology to make advice practical, emotionally supportive, and easy to implement.
  - [x] ⏱ Completed the creation process in approximately 4–6 hours of focused work.

**Extended Commentary:**  
The knowledge base was not simply “prepared” in the technical sense — it was **conceived and built as a multi-purpose asset** from the ground up.  
Before writing, a targeted market analysis was conducted to ensure the content would address clear audience pain points and have strong commercial potential. The book’s structure was deliberately designed to serve **both as a retail-ready product** and **as a semantically organized dataset** for RAG integration.  

The content balances **technical guidance with emotional and behavioral insights**, ensuring that recommendations are easy to follow and sustainable in real life. This required combining skills in **information design, human behavior, and AI-assisted content creation**. By doing so, one creative effort produced **multiple benefits**: a knowledge base, a marketable product, and a reusable educational framework.

---

## ✅ Phase 2: Data Processing & Indexing — (**completed;** later issues found)

**Issues identified (after completion):**
- Chunk boundaries were incorrect: some chunks start mid-chapter and end in a different chapter/subchapter.
- As a result, certain embeddings represent mixed/unrelated content, degrading retrieval precision and ranking stability.

**Required corrections:**
- [x] Re-chunk strictly within chapter/subchapter boundaries.  
- [x] Target chunk size **350 ± 50 GPT tokens** with **~15% soft overlap**.  
  *Outcome (2025-08-10): effective semantic blocks finalized at ~**50–150 words** with neighbor-aware merging; no cross-chapter bleed.*  
- [x] Fix/normalize metadata; regenerate embeddings.  
- [x] Create a new Chroma collection **`finance_book_v4_cos`**; keep **`finance_book_v2`** and **`finance_book_v3`** for rollback and A/B comparison.

**Previously completed work (unchanged):**
- [x] 📄 **Extracted text from Word-Book** — converted the manuscript into a clean, machine-readable `.txt` file for consistent downstream processing.  
- [x] 🧩 **Semantic chunking** — split content into meaning-preserving segments using cosine similarity between sentence-transformer embeddings (`all-MiniLM-L6-v2`). This ensured that each chunk contained a coherent unit of thought.  
- [x] 🧮 **Token count & chunk size adjustment** — pre-calculated the number of tokens in each chunk using `tiktoken` (GPT-4.0 tokenizer) to enforce a **300–500 token range**. This was done *before* embeddings to:
  1. Guarantee that each chunk fits comfortably within ChatGPT-4.0’s context window.
  2. Avoid overly small or excessively large chunks, which could harm retrieval quality.  
  *(Note: This is not the same as the model’s own tokenization step — it was a preparatory measurement for optimal chunk sizing.)*  
- [x] 🔗 **Embedding generation (earlier iteration)** — encoded each chunk using the `thenlper/gte-large` model (1024-dimensional vectors).  
- [x] 🏷 **Metadata enrichment** — attached metadata to each chunk (chapter title, sequential order, token count, character offsets) to enable **filtering, chapter-level navigation, and neighbor-aware retrieval**.  
- [x] 📦 **Vector database creation** — stored all chunks with embeddings and metadata in a **persistent ChromaDB** instance (`./chroma_store`) for fast, local semantic search without API costs or latency.

**Outcome (2025-08-10):**
- **Corpus v4:** 172 chunks (~50–150 words), order preserved; enriched fields `chapter_title`, `chapter_number`, `position`, `source_id`.
- **Dense embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-d) with **L2 normalization**.
- **Vector store:** Chroma **`finance_book_v4_cos`**, HNSW (cosine), persist: `./chroma_store`.
- **Key artifacts/paths:**  
  - Dataset: `Data_Processing_and_Indexing/book_metadata_with_chapters_v4.json`  
  - Scripts: `chapters_enrich_v4.py`, `reembed_chroma_cosine_v4.py`, `query_chroma_v4.py`  
  - Configs/manifest: `configs/embedding_v4.json`, `configs/retriever_v1.json`, `artifacts/v4/vectorstore_manifest.json`

**Extended Commentary:**  
Phase 2 transformed a raw manuscript into an **LLM-ready knowledge index**. The process began with semantic chunking — identifying natural conceptual boundaries rather than cutting text blindly by character length. Pre-tokenization measurement ensured chunks fit the target window. The resulting persistent Chroma index enables millisecond semantic search and forms the backbone for Phase 3.

---

## 📅 Phase 3: Retriever Development — **Planned**

## Phase 3 — Retriever Development (**in progress; paused pending Phase 2.1**)

**Work completed so far:**
- [x] Implemented **BM25** retriever.
- [x] Implemented **Hybrid** retriever (**BM25 + dense embeddings**) with **Reciprocal Rank Fusion (RRF)**.
- [x] Ran initial retrieval tests on collections **`finance_book_v2`** / **`finance_book_v3`**.

**Issue discovered:**
- Retrieval quality was limited by flawed chunking from Phase 2 (cross-chapter chunks; mixed-content embeddings). *(Resolved in v4.)*

**Decision (update 2025-08-10):**
- **Resume Phase 3 on `finance_book_v4_cos`** with Hybrid (BM25 + Dense), **RRF merge**, optional **HyDE** and **cross-encoder rerank**, then **MMR** to assemble **8–12** diversified context chunks.

**Implementation checklist (v4):**
- [ ] **3.1 Sparse (BM25):** lexical index over `book_metadata_with_chapters_v4.json`; lowercase, remove accents, English stopwords; keep `doc_ids = fb-v4c-XXXX`.  
- [ ] **3.2 Dense (Chroma cosine):** use collection `finance_book_v4_cos`; normalize query vectors if encoded externally.  
- [ ] **3.3 RRF merge:** `k = 60`.  
- [ ] **3.4 (Opt.) HyDE:** 1 hypothetical paragraph → search expansion.  
- [ ] **3.5 (Opt.) Rerank:** cross-encoder on top-30; final **N = 8–12**.  
- [ ] **3.6 MMR:** `λ = 0.7`, `final_k = 10`, cosine on dense vectors.  
- [ ] **3.7 Context assembly:** neighbor-merge within chapter if total length < 220–250 words; attach `chapter/position`.  
- [ ] **3.8 One function & CLI:** `retrieve(query, filters?) → List[ContextChunk]` + CLI for debugging.

**Metrics & control:**
- **Recall@5**, **nDCG@10**, **Latency (P50/P95)**; manual **Faithfulness** checks.
- Gold annotations per query (chapter/position or “needle” phrases).
- One-liners:  
  - Check Chroma:  
    ```
    python Data_Processing_and_Indexing\query_chroma_v4.py --q "debt snowball vs avalanche" --n 5
    ```
  - Reindex (if needed):  
    ```
    python Data_Processing_and_Indexing\reembed_chroma_cosine_v4.py --json "Data_Processing_and_Indexing\book_metadata_with_chapters_v4.json" --persist "chroma_store" --collection "finance_book_v4_cos" --model "sentence-transformers/all-MiniLM-L6-v2" --reset
    ```

**Constants to carry into code:**
- **Embedding:** MiniLM-L6-v2 (384d), L2-norm, cosine, collection `finance_book_v4_cos`.  
- **RRF:** `k=60`; **MMR:** `λ=0.7`, `final_k=10`; **HyDE:** 1 paragraph; filters “soft” by default.  
- **Paths:** see artifacts above.

**Definition of Done (retriever):**
- [ ] `retrieve()` returns **8–12** chunks with metadata + brief summaries.  
- [ ] **Recall@5 ≥ baseline dense**; **nDCG@10** ≥ baseline.  
- [ ] ≥70% queries include diverse types (concept/motivation/practical).  
- [ ] Last-run log saved (query, params, pool size, final context).

---

## 📅 Phase 4: LLM Integration — **Planned**
- [ ] Choose a suitable LLM (GPT-4, Mistral, Claude, etc.).  
- [ ] Build a RAG pipeline (LangChain / Haystack).  
- [ ] Design system prompts, tone, and style guidelines.  
- [ ] Test retrieval + generation flow with real financial queries.

---

## 📅 Phase 5: User Interface / API — **Planned**
- [ ] Develop a simple interface (Streamlit / Gradio / FastAPI).  
- [ ] Add question input and generated answer display.  
- [ ] Show retrieved context chunks for transparency.  
- [ ] Allow follow-up / clarifying questions.

---

## 📅 Phase 6: Testing & Optimization — **Planned**
- [ ] Test system with real-world scenarios based on book content.  
- [ ] Measure retrieval accuracy, relevance, and response time.  
- [ ] Optimize retriever, embeddings, and prompts.  
- [ ] Collect feedback and iterate.

---

## 📅 Phase 7: Documentation & Release — **Planned**
- [ ] Write README.md with project architecture and usage instructions.  
- [ ] Provide Colab / Jupyter notebook for local testing.  
- [ ] Package and demo the final system.

---

## 📦 Context Format for LLM (final recommendation)
Each retrieved fragment must follow:

{
  "chapter": "Chapter X: ...",
  "position": <int>,
  "category": "PRACTICAL" | "MOTIVATION" | "CONCEPT" | null,
  "brief": "One-sentence gist.",
  "text": "~60–120 words"
}

Total context budget: **≤ 1,200 tokens**; when over budget, trim `text` but keep metadata.

## 🧱 Chroma Metadata & IDs (v4_cos)
- ID template: `fb-v4c-{index:04d}`
- Required metadata keys:
  - `chapter_title: str`
  - `chapter_number: int`
  - `position: int`
  - `source_id: str`
  - `category: str | null`
  - boolean flags `has_<tag>` (e.g., `has_practical: true`)
- Query format: `col.query(query_texts=[...], n_results=k, where={...})`
- Filters: prefer *soft* filters; for ranges/labels combine `$and`/`$or`; avoid hard intersections that kill recall.

## 🧾 New Chat Hand-Off (paste this at the top of a fresh chat)
> Context: We’re continuing an AI Financial Assistant (RAG) project. Use MiniLM (384d) with L2-norm + cosine in Chroma (`finance_book_v4_cos`). Data file: `Data_Processing_and_Indexing/book_metadata_with_chapters_v4.json`. Next task: build a **hybrid retriever** (BM25 + Dense, RRF), then MMR diversification to return 8–12 mixed-type chunks (concept/motivation/practical). Metrics: Recall@5, nDCG@10; keep latency (P50/P95). Follow `configs/retriever_v1.json`. Use existing scripts for dense; add BM25 + RRF + MMR. Context format and DoD are in this log.

## 🧭 Phase 4 prompt note (LLM Integration)
Use the answer pattern: **explain the concept → motivate → give actionable steps**, and **print sources (Chapter, position)** in every answer.

## 🧩 Dataset lineage (for clarity)
- 2025-08-10 (day log): `book_metadata_merged_fixed.json` → consolidated and renamed to **`Data_Processing_and_Indexing/book_metadata_with_chapters_v4.json`** (source of truth).
- Keep v2/v3 only for rollback/A/B; main collection: **`finance_book_v4_cos`**.

## 📚 Reference artifacts (docs)
- `configs/embedding_v4.json`, `configs/retriever_v1.json`, `artifacts/v4/vectorstore_manifest.json`
- (If present) `docs/decisions/ADR-001-embedding-v4.md`

---

## 🔧 ADDENDUM — Operational Defaults for Phase 3 (Retriever)

> Why this block exists: to lock in concrete defaults so any new chat can continue immediately with zero ambiguity.

### 1) Dense Embeddings (already in place)
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-d), **L2-normalized**, cosine (Chroma HNSW).
- Collection: `finance_book_v4_cos` in `./chroma_store`.
- Query rule: if you encode queries **outside** Chroma, normalize the query vector; if you call `query_texts=[...]`, Chroma handles it.

### 2) Sparse Retriever (BM25) — **explicit defaults**
- Text prep: lowercase = true; remove accents = true; stopwords = **english**.
- Scoring params: **k1 = 1.2**, **b = 0.75** (initial defaults; tune only after baseline).
- Document IDs: reuse dataset IDs `fb-v4c-XXXX` to align with dense side.

### 3) Fusion & Diversification (carry these constants)
- **RRF**: `k = 60`, weights: dense = 1.0, sparse = 1.0.
- **MMR**: `lambda = 0.7`, `final_k = 10`, similarity = cosine on dense vectors.
- Target context: **8–12** chunks, diverse by type (concept / motivation / practical).

### 4) HyDE (optional but recommended for recall)
- Generation LLM: `gpt-4o-mini` (or your smallest available capable model).
- Prompt (single paragraph):  
  *“Write a concise, helpful paragraph that directly answers the user’s question.”*
- Use: run dense search **also** on this synthetic paragraph and union candidates before fusion/rerank.

### 5) Cross-Encoder Reranker (optional polish)
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Pipeline: take top-30 after RRF → rerank → keep **N = 8–12**.

### 6) Chroma metadata & filters — **operational rules**
- **Scalars only** in `metadata` (`str/int/float/bool`); **no lists / no None**. For multi-labels, expose boolean flags `has_<tag>` (e.g., `has_practical: true`).
- **One operator per field** in `where`. For ranges or multiple conditions, combine with `$and` / `$or` / `$in`.
- Prefer **soft** filtering (or none) and rely on RRF + MMR for diversity.

### 7) PowerShell commands — **one-liners**
- Always provide commands as **single-line** PowerShell (no line-continuations).

### 8) Eval & DoD (unchanged; to measure progress)
- Metrics: **Recall@5**, **nDCG@10**, latency **P50/P95**, manual **faithfulness** check.
- Definition of Done (retriever):
  - `retrieve()` returns **8–12** chunks with `chapter/position/category` and a one-line `brief`.
  - Recall@5 / nDCG@10 **≥ dense baseline**.
  - ≥70% queries include a mix of concept/motivation/practical.
  - Save a last-run log (query, params, pool size, final context).

### 9) “Start-in-new-chat” one-liner (copy/paste)
> Context: Continue our AI Financial Assistant (RAG). Dense = MiniLM (384d) with L2-norm + cosine in Chroma (`finance_book_v4_cos`), dataset `Data_Processing_and_Indexing/book_metadata_with_chapters_v4.json`. Build **hybrid retriever** (BM25 k1=1.2, b=0.75 + Dense) with **RRF k=60**, then **MMR (λ=0.7, final_k=10)** to return **8–12** mixed-type chunks. Use **HyDE** (1 paragraph) and optional **cross-encoder/ms-marco-MiniLM-L-6-v2** rerank on top-30. Respect Chroma rules (scalars-only metadata; one operator per field; combine with $and/$or). Provide PowerShell commands as one-liners. Measure **Recall@5 / nDCG@10 / P50/P95**; keep last-run log.
