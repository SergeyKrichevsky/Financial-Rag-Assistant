# TECHNICAL ASSIGNMENT — Phase 4 Handoff Docs (for a fresh chat/session)

**Purpose:** Enable a new ChatGPT session (with no prior memory) to generate all required handoff documents for the project, using only:
1) this Technical Assignment,
2) the current **REPO_STRUCTURE_17.08.2025.md**, and
3) the current **Progress_Log_*.md** (latest).

Everything below is authoritative for **Phase 4 (RAG Core)**. Phase 5 (UI/API) is planned but **not** executed yet (we explicitly defer API hookups & UI).

> **Formatting rule for the next chat:**  
> Output **each document as a single code block** with `markdown` language tag.  
> Inside those documents, if you must show inline code, use **tildes `~~~`** instead of triple backticks to avoid nested-code-block breaks.

---

## 0) Project Snapshot (must appear consistently across all docs)

- **Goal:** Personal-finance RAG assistant that answers user questions using a curated book as the knowledge base.
- Status (Aug 17, 2025): Phase 4 completed and tested offline. Phase 5 delivered the initial Streamlit UI (local-only; no external API). API integration is explicitly deferred to Phase 6. The UI strictly requires `configs/models.ui.json`.
- **Principle:** end users see **only the final answer**; developer metadata (retrieval refs, prompt, caps) is separate, hidden by default.

**Core components (implemented in code):**
- `llm_integration/llm_router.py` — model router (backends: `local_stub` now; `openai` later).
- `llm_integration/answer_generator.py` — system prompt loader, context sanitizer, token cap, LLM call, final/dev outputs.
- `llm_integration/retriever_bridge.py` — adapter that post-filters retriever results and builds `context_text` + `source_refs`.
- `llm_integration/config_loader.py` — safe defaults + loaders for `retriever` / `generator` / `logging`.
- `llm_integration/run_logger.py` — writes `artifacts/v4/runs/last_run_phase4.json` and optionally appends `runs_history.jsonl`.
- `llm_integration/test_generate.py` — E2E test (retrieve → generate → log).
- `llm_integration/cli_ask.py` — CLI Q&A: `--q "..." [--k N] [--show-dev]`.
- `llm_integration/smoke_test_openai.py` — smoke test file exists; **actual run deferred to Phase 5**.

**Configs & artifacts:**
- `model.config.json` — backend/model selection; keep `"backend": "local_stub"` in Phase 4.
- `configs/rag_config.json` — all knobs (retriever/generator/logging) incl. **off-switch flags**.
- `configs/system_prompt.txt` — editable system instruction.
- `artifacts/v4/runs/` — logs: `last_run_phase4.json`, optional `runs_history.jsonl`.

**Retriever dependency (Phase 3 asset):**
- `Retriever_Development/v4/hybrid_retriever_v4.py` — exposes `HybridRetrieverV4(...).retrieve(question, k)`.

---

## 1) Inputs the next chat will receive

- The **latest** `REPO_STRUCTURE_v4.md` (current repo tree).
- The **latest** `Progress_Log_*.md` (project log; this is the curator-facing report).
- This file: **TECHNICAL_ASSIGNMENT.md** (the current document).

The next chat must **read REPO_STRUCTURE** to confirm paths and **read Progress_Log** to reflect dates/decisions in the docs it generates.

---

## 2) Required Deliverables (documents to produce)

All files go under `docs/` and must be written in **one markdown code block each**:

1. `docs/HANDOFF/README_HANDOFF.md` — entry point: what’s implemented, how to run, where to read more.
2. `docs/HANDOFF/ARCHITECTURE_PHASE4.md` — end-to-end pipeline, components, data contracts, control flow, configs, error handling.
3. `docs/HANDOFF/CONFIG_REFERENCE.md` — definitive spec for `model.config.json`, `configs/rag_config.json`, and `system_prompt.txt`.
4. `docs/HANDOFF/RUN_GUIDE_PHASE4.md` — how to run CLI/E2E locally; where logs go; minimal troubleshooting.
5. `docs/HANDOFF/LOGGING_SPEC.md` — schema & examples for `last_run_phase4.json` and `runs_history.jsonl`, plus locations.
6. `docs/HANDOFF/DECISIONS_PHASE4.md` — ADR digest: alternatives considered → final choice → rationale → impact.
7. `docs/HANDOFF/NEW_CHAT_KICKOFF.md` — short copy-paste intro for a **brand-new** chat to get up to speed in 1 minute.
8. *(Optional but desired)* `docs/REPORTS/retriever_eval_v4.md` — current retriever metrics (if available) or a placeholder with how to compute.
9. *(Optional but desired)* `docs/REPORTS/phase4_status.md` — checklist of done/left, risks, and Phase-5 entry criteria.
10. **Update** existing: `REPO_STRUCTURE_v4.md` — ensure it lists all new docs and any new code files created during Phase 4.
11. **Update** existing: latest `Progress_Log_*.md` — add a concise Phase-4 wrap-up section and Phase-5 next steps.

> If metrics aren’t available yet, produce the Reports as **templates** with “TBD” placeholders and instructions to fill in later.

---

## 3) Canonical technical content the next chat must include

### 3.1 Architecture essentials (repeat across relevant docs)
- **Pipeline:**  
  Retrieve top-K → Bridge filters/dedup/cap-per-chapter → `context_text` + `source_refs` → Generator loads system prompt → sanitize (dedupe + char cap) → token cap → Router `.complete(system,user)` → outputs: **Final** (user) and **Developer** (hidden).
- **Separation of outputs:** end-user **never** sees sources/IDs by default; developer info is logged and can be toggled in future UI.
- **Developer Panel (Phase 5, DEV-only):**  
  - Visible only when `APP_ENV=DEV`.  
  - Shows Phase-4 logs (`artifacts/v4/runs/*`, `runs_history.jsonl`) and config snapshots; provides an E2E test button.  
  - Intended for local development; not for public deployments.
- **Router contract:** unified `.complete(system_prompt, user_prompt) -> str`; model/backends switched via `model.config.json` and env (Phase 5).
- **Sanitizer & caps:** character cap **and** token cap; both are togglable via off-switch flags.
- **Retriever diversity controls:** `exclude_chapters`, `max_per_chapter` (both configurable; off-switch supported).
- **Configs are the single source of truth** (no code edits required for tuning common parameters).

### 3.2 Configs (must be documented precisely)
- `model.config.json` (Location: llm_integration/model.config.json) fields:
  - `backend`: `"local_stub"` (Phase 4) or `"openai"` (Phase 5),
  - `model`: e.g., `"gpt-5-mini"` (used when backend = openai),
  - `temperature` (optional).
- `configs/rag_config.json` sections:
  - `retriever`: `k_default`, `max_per_chapter`, `exclude_chapters`, `use_filters`, `use_per_chapter_cap`.
  - `generator`: `system_prompt_path`, `max_context_chars`, `max_context_tokens`, `token_encoding`, `use_sanitizer`, `use_token_cap`.
  - `logging`: `runs_dir`, `history_file`, `enable_history`.
- `configs/system_prompt.txt`: file-based system instruction; editable without code changes.
- **ENV (Phase 5 only):** `OPENAI_API_KEY` (deferred).

- `configs/models.ui.json`: new Phase-5 file defining **available UI models**. 
  - Example content:  
    ~~~json
    {
      "models": [
        { "id": "chatgpt-5-micro", "label": "ChatGPT 5.0 Micro", "desc": "for testing / lowest cost", "enabled": true },
        { "id": "chatgpt-5-mini",  "label": "ChatGPT 5.0 Mini",  "desc": "balance of quality & cost", "enabled": true },
        { "id": "chatgpt-5",       "label": "ChatGPT 5.0",       "desc": "highest quality",          "enabled": true }
      ]
    }

    ~~~
  - Purpose: UI reads available models from here, instead of hard-coding.  
  - If file is missing or invalid → the UI must raise an explicit error and stop.



### 3.3 Commands (must appear in Run Guide)
- CLI ask (with/without developer output):
  - `python -m llm_integration.cli_ask --q "Как начать вести бюджет?" --k 5`
  - `python -m llm_integration.cli_ask --q "Как начать вести бюджет?" --k 5 --show-dev`
- E2E test:
  - `python -m llm_integration.test_generate`
- **UI run (Phase 5 local-only)**  
  - PowerShell (Windows):  
    ~~~powershell
    $Env:APP_ENV = "DEV"
    streamlit run Interface/app.py
    ~~~
  - bash/zsh (macOS/Linux):  
    ~~~bash
    APP_ENV=DEV streamlit run Interface/app.py
    ~~~
  - Install UI deps:  
    ~~~bash
    pip install -r Interface/requirements.txt
    ~~~
  - Notes: `APP_ENV=DEV` shows the Developer Panel; `APP_ENV=PROD` hides it. The UI strictly requires `configs/models.ui.json`.

- (Phase 5) Smoke test:
  - `python -m llm_integration.smoke_test_openai` (after switching backend to `"openai"` and setting `OPENAI_API_KEY`).

### 3.4 Logging (must be specified with fields)
- **`artifacts/v4/runs/last_run_phase4.json`** — compact snapshot of the latest run, containing at minimum:
  - `ts_unix` (float), `model` (str), `question_preview` (first ~200 chars),  
    `context_chars` (int), `num_refs` (int), `refs` (list of shallow refs), `answer_chars` (int).
- **`artifacts/v4/runs/runs_history.jsonl`** — optional append-only history (one JSON per line) with the same payload structure; enabled via `logging.enable_history`.
- artifacts/v4/runs/runs_history.jsonl — append-only

### 3.5 Data contracts (exact signatures)
- `retrieve_context(question: str, k: Optional[int]) -> Tuple[str, List[Dict]]`
  - returns **plain** `context_text` and `source_refs` list with fields:
    - `id`, `score`, `chapter`, `position`, `category`, `source_id`, `preview` (first ~200 chars).
- `generate_answer(context: str, question: str) -> Dict[str, object]`
  - returns:
    - `"final_output": str` (for user),
    - `"developer_output": { "context_used": str, "full_prompt": str, "limits": {...} | "error": {...} }`.

### 3.6 Error handling (Phase 4 scope)
- Friendly fallback messages for RuntimeError (e.g., missing key) and generic exceptions in `answer_generator`.
- Missing prompt file → explicit `FileNotFoundError`.
- Windows “resource module…” message from BM25 is harmless.

---

## 4) Document-by-document outlines & acceptance criteria

> For each doc, the next chat must generate **one** markdown file with the sections below, concise but complete. Use `~~~` for inline code blocks **inside** the document.

### 4.1 `docs/HANDOFF/README_HANDOFF.md`
- Project at a glance; what Phase 4 includes; Phase 5 deferred scope.
- Bullet list of implemented modules and where they live.
- Quick Run (CLI + E2E) with commands.
- Where outputs/logs appear.
- Pointers to other docs (Architecture, Config Reference, Run Guide, Logging Spec, ADR, New Chat Kickoff).
- “What’s next” for Phase 5 (high level).

**Acceptance:** a new engineer can run the pipeline in <5 minutes and knows where to read further.

---

### 4.2 `docs/HANDOFF/ARCHITECTURE_PHASE4.md`
- High-level overview of the pipeline.
- Components & files (with responsibilities).
- Data contracts and control flow (step-by-step).
- Configs (single source of truth), limits & guards.
- Error handling and logging.
- Security & privacy (keys not in code; hidden sources by default).
- Extension points for Phase 5.

**Acceptance:** contains enough detail to modify any single component without asking for prior chat context.

---

### 4.3 `docs/HANDOFF/CONFIG_REFERENCE.md`
- Load order (config loader merges user JSON over safe defaults).
- `model.config.json` fields + example.
- `configs/rag_config.json` full schema + field reference.
- How values are consumed in code (bridge/generator/logger).
- Safe change checklist (e.g., tune `k_default`, adjust caps, toggle off-switches).

**Acceptance:** all knobs are discoverable and unambiguous; no need to read code to tune behavior.

---

### 4.4 `docs/HANDOFF/RUN_GUIDE_PHASE4.md`
- Prereqs; repo root assumption.
- CLI commands (with/without `--show-dev`) and E2E test.
- Where logs go; how to enable/disable history.
- Basic troubleshooting (common Python/module path issues; BM25 Windows message; missing prompt).
- Phase 5 preview: how to switch to OpenAI + run smoke test.

**Acceptance:** copy-paste runnable, accurate paths, and matches the actual repo.

---

### 4.5 `docs/HANDOFF/LOGGING_SPEC.md`
- Exact path locations for logs.
- JSON field definitions for `last_run_phase4.json` and `runs_history.jsonl`.
- Size/PII considerations (trim previews).
- Example payloads (use `~~~json`), plus note that history never blocks main flow.

**Acceptance:** allows an external observer to parse and analyze runs without reading code.

---

### 4.6 `docs/HANDOFF/DECISIONS_PHASE4.md` (ADR digest)
For each decision, include:
- **Context/problem**, **Options considered**, **Decision**, **Why**, **Trade-offs**, **Follow-ups**.

**Decisions to include at minimum:**
- Provider-agnostic router + config-first model selection.
- Split Final vs Developer outputs; sources hidden by default.
- Retriever filters (`exclude_chapters`) and per-chapter diversity cap.
- Sanitizer + char cap + token cap; off-switch flags.
- Externalized system prompt as a file.
- Deferring API/UI to Phase 5.

**Acceptance:** a maintainer understands *why* we made current choices and what to revisit later.

---

### 4.7 `docs/HANDOFF/NEW_CHAT_KICKOFF.md`
- 10–15 lines max: project goal, what’s already implemented (Phase 4), how to run one command, where to read more, what’s next (Phase 5).

**Acceptance:** can be pasted into a brand-new chat to bootstrap context immediately.

---

### 4.8 (Optional) `docs/REPORTS/retriever_eval_v4.md`
- If metrics exist: Recall@K / nDCG@K / MRR; latency notes; dataset description.
- If not: provide a **template** with commands and placeholders to fill later.

---

### 4.9 (Optional) `docs/REPORTS/phase4_status.md`
- Checklist of completed items, items moved to Phase 5, risks, and entry criteria for Phase 5 (e.g., smoke test OK, API key set, UI scaffold done).

---

### 4.10 **Update** `REPO_STRUCTURE_v4.md`
- Ensure it lists:
  - All **llm_integration** files above,
  - `configs/*` and `model.config.json`,
  - `artifacts/v4/runs/*`,
  - **new** docs under `docs/HANDOFF` and `docs/REPORTS`.
- Keep notes where “not all files are included yet” if that’s the repo convention.

---

### 4.11 **Update** `Progress_Log_*.md`
- Add a **Phase 4 wrap-up** summary:
  - What was implemented (with file paths),
  - What was explicitly deferred to Phase 5 (API hookup, smoke test run, UI),
  - Any open TODOs,
  - Testing done (CLI/E2E),
  - Current defaults (`backend: local_stub`, `k_default`, caps).
- Add **Next steps (Phase 5)**:
  - Switch to `openai`, set `OPENAI_API_KEY`, run smoke test, build UI with hidden Developer Output by default, etc.

**Acceptance:** the curator can rely on the log as the primary report.

### 4.12 Phase-6 extension: User-driven E2E script
- Add new script (planned for Phase 6) alongside `test_generate.py`.  
- Instead of a fixed hard-coded question, it must accept the **user’s input** from the UI and run the full RAG → LLM pipeline.  
- Purpose: connect Developer Panel with true end-to-end tests using live user input.  
- The existing `test_generate.py` remains unchanged (keeps hard-coded default).  
- The Developer Panel's "Run E2E" button must call this script to use the current **user-entered** question from the UI.



---

## 5) Style & Quality Bar

- Language: **English** in all docs (consistent with code comments rule).
- One **single** code block per file (language tag `markdown`).  
  Use `~~~` for inner examples and code snippets to avoid nested-block breaks.
- Keep sections short and scannable: bullets > long prose.
- Verify all paths and commands against **REPO_STRUCTURE_v4.md**.
- No secrets or API keys anywhere.
- Include dates where relevant (use the date from Progress_Log).
- UI labels and in-app texts must be **English**.  
- Brand header: **Sergey Krichevskiy Group** (Phase 5 UI).

---

## 6) Acceptance Checklist (for the person producing docs)

- [ ] All 7 core docs under `docs/HANDOFF` exist and pass the quality bar.  
- [ ] Optional reports exist (or templates with “TBD + how-to”).  
- [ ] `REPO_STRUCTURE_v4.md` updated with new docs and any new code files.  
- [ ] `Progress_Log_*.md` updated with Phase-4 wrap-up + Phase-5 next steps.  
- [ ] Every doc delivered as a **single** `markdown` code block (no splits).  
- [ ] Commands copy-paste and run from repo root (`python -m llm_integration.*`).  
- [ ] Logging spec matches fields actually written by `run_logger.py`.

---

## 7) Known Constraints & Assumptions

- Windows: info message “resource module not available…” from BM25 is harmless.
- If `tiktoken` isn’t installed, token cap degrades to word-based trimming (safe).
- The knowledge base is the curated **finance_book_v4** (see retriever metadata fields).
- UI defaults must **hide** developer metadata; provide a toggle only for maintainers (Phase 5).
- The Phase-5 UI **requires** `configs/models.ui.json` to exist and be valid (strict). Missing/invalid file must stop the UI with a clear error.
- Developer Panel is **local-only** (DEV); do not expose it on public deployments.
- Deployment is **deferred** to late Phase 6 (after tests), candidate: Streamlit Community Cloud or Render.

---

## 8) Hand-off Notes for the next chat

- If something in `REPO_STRUCTURE_v4.md` is missing or differs, **ask for the updated file** before proceeding.
- If any file paths in this TA conflict with the repo, treat the repo as ground truth and reflect the correct paths across all docs.
- If metrics aren’t available, create the templates and clearly mark all **TBD** areas.

---

## 9) Minimal reference snippets (reuse across docs)

> Use these inside docs with `~~~` fences (not triple backticks) to avoid breaking the single-block requirement.

**CLI (with and without developer output)**

~~~bash
python -m llm_integration.cli_ask --q "Как начать вести бюджет?" --k 5
python -m llm_integration.cli_ask --q "Как начать вести бюджет?" --k 5 --show-dev
~~~

**E2E test**

~~~bash
python -m llm_integration.test_generate
~~~

**(Phase 5) Smoke test**

~~~bash
# After switching model.config.json backend to "openai" and setting OPENAI_API_KEY:
python -m llm_integration.smoke_test_openai
~~~

**Config pointers**

- `model.config.json` → backend/model/temperature (Phase 4: `"local_stub"`).  
- `configs/rag_config.json` → `retriever`, `generator`, `logging` + off-switch flags.  
- `configs/system_prompt.txt` → editable system instruction.

**Logs**

- Latest run: `artifacts/v4/runs/last_run_phase4.json`  
- Optional history: `artifacts/v4/runs/runs_history.jsonl`

---

**End of Technical Assignment**


## ATTACHMENTS REQUIRED (ship with this Technical Assignment)

To ensure a fresh chat can continue work **without guessing any values**, this Technical Assignment **must be accompanied** by three snapshot files. If any of these are missing, the next chat **must request them before producing the docs**.

### A) Required attachments (3)
1) **SNAPSHOT_model.config.json** — the exact, current contents of `model.config.json`.  
   - Verify fields: `backend`, `model`, `temperature` (if present).  
   - Note: No secrets are expected here. If `backend` is `"openai"` (Phase 5), keys stay in ENV, not in this file.

2) **SNAPSHOT_rag_config.json** — the exact, current contents of `configs/rag_config.json`.  
   Must include the live values for:
   - `retriever`: `k_default`, `max_per_chapter`, `exclude_chapters` (full list), `use_filters`, `use_per_chapter_cap`
   - `generator`: `system_prompt_path`, `max_context_chars`, `max_context_tokens`, `token_encoding`, `use_sanitizer`, `use_token_cap`
   - `logging`: `runs_dir`, `history_file`, `enable_history`

3) **SNAPSHOT_system_prompt.txt** — the full text of `configs/system_prompt.txt` used in Phase 4.

### B) Optional but strongly recommended
4) **EXAMPLE_last_run_phase4.json** — the current `artifacts/v4/runs/last_run_phase4.json` as a concrete example of the log schema and values.

### C) If attachments are missing — required action
Before drafting any handoff documents, the next chat must ask for the exact snapshots. Use this template:

> **Please provide Phase-4 config snapshots:**  
> 1) `SNAPSHOT_model.config.json` (exact current contents),  
> 2) `SNAPSHOT_rag_config.json` (exact current contents),  
> 3) `SNAPSHOT_system_prompt.txt` (full text).  
> *(Optional)* `EXAMPLE_last_run_phase4.json`.  
> You can paste the file contents directly into the chat or upload the files. These snapshots are needed to avoid mismatches in parameters, flags, and prompt text.

### D) Acceptance checklist for attachments
- [ ] `backend` in `SNAPSHOT_model.config.json` matches the intended Phase (Phase 4 → `"local_stub"`).  
- [ ] All `retriever`/`generator`/`logging` keys in `SNAPSHOT_rag_config.json` are present and consistent with the repo and log.  
- [ ] `system_prompt_path` in the snapshot points to an existing file and matches `SNAPSHOT_system_prompt.txt`.  
- [ ] (If provided) `EXAMPLE_last_run_phase4.json` is located under `artifacts/v4/runs/` and contains:  
      `ts_unix`, `model`, `question_preview`, `context_chars`, `num_refs`, `refs` (trimmed), `answer_chars`.  
- [ ] No secrets (API keys) are present in any snapshot. If any sensitive value appears, redact the key **only**, keep structure unchanged.

> With these attachments plus `REPO_STRUCTURE_v4.md` and the latest `Progress_Log_*.md`, the new chat has all exact parameters, the live prompt text, and an example log — sufficient to generate the full handoff docs and continue with Phase 5 without ambiguity.
