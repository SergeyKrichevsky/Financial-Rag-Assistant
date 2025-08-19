# Handover — Financial RAG Assistant
**UTC timestamp:** 2025-08-19 22:00

---

## 1) Context / Purpose
This handover transfers operational ownership of **Financial RAG Assistant**, a dual-mode Q&A system built as a **Bootcamp capstone** at Developers Institute (Israel).  
- **Default mode:** OpenAI **Assistants API + File Search** (Vector Store). The book is uploaded once and reused across chats.  
- **Alternate mode:** **Local RAG** (hybrid retriever + generator) implemented earlier.  
The book (knowledge base) was **written specifically for this project** and is already available on **Amazon (Kindle & Paperback)**.

---

## 2) Actions Taken (step-by-step)
1. Authored the book as the project’s knowledge base; exported as PDF.  
2. Created an **Assistant** on OpenAI Platform and attached a **Vector Store**; uploaded/attached the PDF (indexed once).  
3. Implemented a **Streamlit UI** (`Interface/app.py`) with a mode toggle:  
   - Assistants API (default): threads → messages → runs (`create_and_poll`).  
   - Local RAG: calls `llm_integration.retriever_bridge.retrieve_context` → `llm_integration.answer_generator.generate_answer`.  
4. Added **API-key handling** (ENV or paste-in-session) and Assistant ID sourcing (ENV → `configs/assistant.meta.json` → UI override).  
5. Fixed SDK compatibility (`client.threads` and `client.beta.threads`), import paths, and error reporting.

---

## 3) Decisions Made (with rationale)
- **Pivot to Assistants API by default.** For a single, well-scoped book, server-side File Search yields higher reliability and lower operational overhead than maintaining a fully custom RAG stack.  
- **Keep Local RAG as a second mode.** Demonstrates stack knowledge and preserves transparency/control when needed.  
- **Do not hard-code secrets/IDs.** Keys via ENV or paste; Assistant ID via ENV/config for environment portability (DEV/TEST/PROD).  
- **Minimal UI, maximum clarity.** One file, explicit toggles, and friendly errors.

---

## 4) Results & Current State
- UI runs and returns answers in both modes.  
- Assistants mode shows `model: … | thread: …` footer; Local RAG triggers the project retriever/generator when available on `PYTHONPATH`.  
- The system prompt for the Assistant is stored/managed on the OpenAI Platform (a local copy also exists as `configs/system_prompt_assistant_API.txt` for reference).

---

## 5) Parameters & Metadata
### Environment
- **OS:** Windows 11 (tested), macOS/Linux also fine.  
- **Python:** 3.11+ (3.13 OK).  
- **Core libs:** `openai >= 1.100.0`, `streamlit >= 1.38`.  
- **Launch:** `streamlit run Interface/app.py`.

### Assistants mode
- **Model:** `gpt-4.1-mini` (change on Platform without code edits).  
- **Artifacts:** Assistant (with File Search tool) + Vector Store (book attached, status **Indexed/Completed**).  
- **IDs:** `assistant_id`, `vector_store_id` (non-secret; kept in `configs/assistant.meta.json` or ENV).  
- **Threads:** one per browser session (persistent until page reload).

### Local RAG mode
- **Retriever:** `llm_integration.retriever_bridge.retrieve_context(q, k=5)` (BM25 + dense hybrid).  
- **Generator:** `llm_integration.answer_generator.generate_answer(ctx, q)` → returns `final_output`.  
- **Discovery:** repo root is injected into `sys.path` so `llm_integration` can be imported even when running from `/Interface`.

### Security & Privacy
- **No secrets in repo.** Keys provided via ENV or pasted into the UI (session-only; not written to disk).  
- **Project scoping.** Use Project-scoped keys/objects on the OpenAI Platform.  
- **Key rotation.** Replace ENV or paste a new key; no code changes required.  
- **Data pruning.** You can delete files/vector stores/assistants from the Platform when not needed.

---

## 6) Artifacts & File Tree
```
financial-rag-assistant/
├─ Interface/
│  └─ app.py                   # Streamlit UI with mode toggle (Assistants / Local RAG)
├─ llm_integration/            # Local pipeline (retriever/generator)
├─ configs/
│  ├─ assistant.meta.json      # non-secret IDs (assistant, vector store, model)
│  └─ system_prompt_assistant_API.txt
├─ app_my_RAG.py               # earlier UI (kept for reference)
├─ requirements.txt
└─ docs/ (optional)            # handover/readme, diagrams, screenshots
```

---

## 7) Risks / Open Questions
- **Assistant indexing stuck / empty Vector Store.** Sometimes UI shows “Processing/Indexing…” for too long; re-attach or re-create the Vector Store if needed; use REST to attach as a fallback.  
- **API evolution.** OpenAI is progressively moving focus to Responses/Agents; Assistants continue to work, but future migration may be recommended.  
- **Local RAG visibility.** Ensure `llm_integration` is on `PYTHONPATH` or run from repo root.  
- **Cost awareness.** Vector Store storage + model tokens; monitor usage on the Platform.

---

## 8) Next Steps (actionable checklist)
- **Finalize Platform setup**
  - Ensure the book file shows **Indexed/Completed** under *Vector Store → Files attached*.  
  - Confirm the same Vector Store is attached under *Assistant → Tools: File Search*.  
- **Lock the configuration**
  - Put `assistant_id` in `configs/assistant.meta.json` (and/or set ENV `ASSISTANT_ID`).  
  - Keep `OPENAI_API_KEY` in ENV for demos to avoid pasting every time.
- **Demo prep (video 1–3 min)**
  1) State purpose; dual-mode; book authored for the project (available on Amazon).  
  2) Show Assistant → Tools: File Search → Indexed.  
  3) Run UI; Assistants mode → ask a unique book fact; show footer `model|thread`.  
  4) Switch to Local RAG → ask similar question; note trade-offs.  
  5) Close with Security (no secrets in repo) + quick roadmap.
- **Roadmap**
  - UI **citations** from Assistants annotations (clickable “sources”).  
  - **Eval harness** (10–20 golden questions) for regression across modes.  
  - **Telemetry** (latency & rough cost per query).  
  - Optional **Colab** one-click demo (install → set ENV → launch).

---

## 9) Runbooks (Operations)
### R1. Change the Assistant model
- On the OpenAI Platform, open the Assistant → change **Model** (e.g., `gpt-4.1` or `gpt-4o-mini`).  
- No code changes required.

### R2. Update the system prompt
- Edit **Instructions** in the Assistant on the Platform (you may keep a local copy under `configs/system_prompt_assistant_API.txt`).

### R3. Replace / add a new book file
- Platform → **Vector stores → Your store → Add files → Attach**.  
- Wait for **Indexed** status. If UI misbehaves, attach via REST (Create Vector Store File) and recheck.

### R4. Rotate/replace API key
- Update ENV `OPENAI_API_KEY` (recommended) or paste in UI.  
- Never commit keys; repository contains no secrets.

### R5. Run a clean session
- Reload the Streamlit page (new browser session → new thread).  
- Or restart the app: `streamlit run Interface/app.py`.

---

## 10) Source References (links + verification date)
- **OpenAI Platform (Assistants & File Search)** — quickstart & tools (verified **2025-08-19**).  
- **OpenAI Python SDK (1.x)** — threads/messages/runs (`create_and_poll`) (verified **2025-08-19**).  
- **API Keys / Projects** — project-scoped resources (verified **2025-08-19**).  
*(Direct links intentionally general here; use the Platform Docs navigation for the latest pages.)*
