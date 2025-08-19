# Financial RAG Assistant — README  
**Version:** 1.0 • **Owner:** Sergey Krichevskiy • **Bootcamp Capstone:** Developers Institute (Israel)

---

## 1) Name & Summary
**Financial RAG Assistant** is a dual-mode Q&A system that answers strictly from a single book used as the knowledge base.  
- **Default mode:** OpenAI **Assistants API + File Search** (Vector Store). The book is uploaded once and reused across chats.  
- **Alternate mode:** **Local RAG** (hybrid retriever + generator) built earlier during the project.

> Note: The book was **written specifically for this project** and is already available on **Amazon** ( [Kindle eBook](https://kdp.amazon.com/amazon-dp-action/us/dualbookshelf.marketplacelink/B0FLVT9LQV) · [Paperback](https://kdp.amazon.com/amazon-dp-action/us/dualbookshelf.marketplacelink/B0FLWDCLF3) )

---

## 2) Features
- 🔁 **Two backends** switchable in UI: Assistants API (default) / Local RAG.  
- 🔐 **Key management**: use server `OPENAI_API_KEY` or paste manually (session-only).  
- 🧠 **Knowledge base**: the full book in a Vector Store (no need to resend on each request).  
- 🧵 **Threaded chat**: per-session thread continuity in Assistants mode.  
- ⚠️ **Graceful errors**: friendly messages if config/modules are missing.

---

## 3) Architecture Overview
```
User (Browser)
   │
   └── Streamlit UI (Interface/app.py)
        ├─ Mode A: Assistants API (default)
        │    ├─ Thread → Messages → Run (create_and_poll)
        │    └─ File Search → Vector Store (book indexed once)
        └─ Mode B: Local RAG (existing)
             ├─ retriever_bridge.retrieve_context(q, k)
             └─ answer_generator.generate_answer(context, q)
```
- **Config priority for Assistant ID:** ENV → `configs/assistant.meta.json` → UI override.

---

## 4) Setup (prereqs, install, env)
### 4.1 Prerequisites
- Python **3.11+** (3.13 OK), pip, git.  
- OpenAI account + **API key** (project-scoped).

### 4.2 Create & activate venv, install deps
```bash
# Windows (PowerShell)
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
pip install -U openai streamlit
```
```bash
# macOS / Linux
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -U openai streamlit
```

### 4.3 Environment variables (recommended)
```bash
# Option A: set globally (example)
# Windows
setx OPENAI_API_KEY "sk-xxxx"
# macOS/Linux
echo 'export OPENAI_API_KEY=sk-xxxx' >> ~/.bashrc
```
Optional (for no-typing demo):
```bash
# Optional: pin assistant id
# Windows
setx ASSISTANT_ID "asst_xxx"
# macOS/Linux
echo 'export ASSISTANT_ID=asst_xxx' >> ~/.bashrc
```

---

## 5) Configuration
Create **`configs/assistant.meta.json`** (no secrets inside):
```json
{
  "name": "Finance Assistant",
  "assistant_id": "asst_XXXXXXXXXXXXXXX",
  "vector_store_id": "vs_XXXXXXXXXXXXXXX",
  "model": "gpt-4.1-mini"
}
```
Priority at runtime: **ENV → this file → UI field**.

---

## 6) Usage (UI)
```bash
streamlit run Interface/app.py
```
- **Context Source:** choose backend  
  - **Assistant API (default):** paste API key or use ENV; ensure `assistant_id` is filled (auto-read from config if present).  
  - **Local RAG (existing):** uses your modules:
    - `llm_integration.retriever_bridge.retrieve_context(q, k=5)`
    - `llm_integration.answer_generator.generate_answer(ctx, q)`
- Ask a question. In Assistants mode you’ll see a footer like:  
  `model: gpt-4.1-mini | thread: thd_...`

---

## 7) RAG / Training specifics (high level)
- **Dataset:** author’s book (created specifically for this project).  
- **Chunking & Embedding:** experimented with chunk sizes/overlap; produced embeddings and indexes for the local mode.  
- **Retriever (local):** hybrid (BM25 + dense) with simple re-ranking; configurable top-k.  
- **Generation:** prompt-controlled answer synthesis; final text returned to UI.

---

## 8) Evaluation (how to validate quickly)
- Ask **unique facts** from the book (not guessable).  
- On the OpenAI Platform:  
  - Vector Store → **Files attached** → status **Indexed/Completed**.  
  - Assistant → **Tools → File Search** → same store attached.  
  - Logs → last Run → step shows **file_search** (retrieval occurred).

*(No time for formal metrics in this version.)*

---

## 9) Deployment
- **Local demo:** Streamlit as described above.  
- **Keys:** provide via ENV (preferred) or paste in UI (session only).  
- **Multiple environments:** keep separate Assistants/Vector Stores per DEV/TEST/PROD; select via ENV or `assistant.meta.json`.

---

## 10) Troubleshooting
- **“OpenAI object has no attribute 'threads'”** → update SDK: `pip install -U openai`.  
- **No answers in Local RAG** → ensure project root on `PYTHONPATH` or run from repo root; verify `llm_integration/*` exists.  
- **Vector Store is empty / endless indexing** → add file via Platform “Attach” or via API; recreate the store if UI glitches.  
- **Assistant not using book** → confirm file is **Indexed** and assistant has File Search tool attached to that store.

---

## 11) Security & Privacy
- **No secrets in repo.** API keys are provided via ENV or typed in UI; manual keys are kept **only in the Streamlit session**.  
- **IDs as config.** `assistant_id`/`vector_store_id` stored in a JSON config (non-secret), or supplied via ENV.  
- **Data isolation.** Use project-scoped keys/objects on the OpenAI Platform; delete files/assistants when not needed.  
- **Rotation.** Keys/IDs can be changed without code edits (12-factor principle).

---

## 12) Roadmap
- **Citations UI:** extract Assistants annotations and display clickable sources (file/page/snippet).  
- **Eval harness:** a tiny script with 10–20 golden questions for regression checks (per-mode accuracy).  
- **Telemetry:** basic latency & cost per query (Assistants vs Local RAG).  
- **Colab demo (Phase 7 optional):** one-click try (install → set env → launch).  
- **Model/Prompt mgmt:** small JSON for per-env model choice and prompt presets.

---

## 13) Troubleshooting (Quick FAQ)
- *Why not hard-code `assistant_id`?* — IDs differ per environment; config/ENV is safer and standard practice.  
- *Why Assistants by default?* — accuracy/maintenance/cost benefits vs building full RAG infra for one book.  
- *When to use Local RAG?* — when you need maximum control/visibility over retrieval and ranking.

---

## 14) Acknowledgments
- Developers Institute (Israel) — **Bootcamp Capstone** context.  
- The book (knowledge base) authored for this project; currently available on **Amazon (Kindle & Paperback)**.

---



## Operations / Handover
For operational details, configuration tips, troubleshooting, and demo guidelines,  
please refer to the **handover document**:

[📄 Open Handover Documentation](docs/HANDOVER.md)