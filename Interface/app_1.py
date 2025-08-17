# =========================================
# HOW TO RUN (copy-paste into your terminal)
# =========================================
# macOS / Linux (bash/zsh):
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -U pip
#   pip install -r requirements.txt
#   # DEV (with Developer Panel):
#   APP_ENV=DEV UI_DEFAULT_MODEL=chatgpt-5-micro streamlit run Interface/app.py
#   # PROD (no Developer Panel):
#   APP_ENV=PROD UI_DEFAULT_MODEL=chatgpt-5-mini streamlit run Interface/app.py
#
# Windows PowerShell (current session only):
#   python -m venv .venv ; .\.venv\Scripts\Activate.ps1
#   python -m pip install -U pip
#   pip install -r requirements.txt
#   # DEV (with Developer Panel):
#   $Env:APP_ENV="DEV" ; $Env:UI_DEFAULT_MODEL="chatgpt-5-micro"
#   streamlit run Interface/app.py
#   # PROD (no Developer Panel):
#   $Env:APP_ENV="PROD" ; $Env:UI_DEFAULT_MODEL="chatgpt-5-mini"
#   streamlit run Interface/app.py
#
# Stop server:
#   - In terminal: Ctrl+C (or Kill Terminal 🗑 in VS Code)
#
# Notes:
# - UI_DEFAULT_MODEL is optional; falls back to the first list item if not set.
# - Phase 5 does NOT call any external APIs and does NOT modify any JSON files.
# - Developer Panel reads server-side logs; E2E test runs local Phase-4 pipeline.
# =========================================

# Streamlit MVP UI for Phase 5 (no real external API calls yet)
# Brand: Sergey Krichevskiy Group
# Internal model IDs: chatgpt-5-micro, chatgpt-5-mini, chatgpt-5
# Developer Panel only appears if APP_ENV=DEV

import sys
from pathlib import Path

# Add the project root (one level above Interface/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import streamlit as st

# Optional imports for E2E run (Phase 5 local pipeline).
# If modules are missing or project structure differs, we'll show a clear message in the UI.
try:
    from llm_integration.answer_generator import generate_answer as _gen_answer
    from llm_integration.retriever_bridge import retrieve_context as _retrieve_context
    from llm_integration.llm_router import get_llm as _get_llm
    from llm_integration.run_logger import log_phase4_run as _log_run
    _E2E_AVAILABLE = True
except Exception as _e:
    _E2E_AVAILABLE = False
    _E2E_IMPORT_ERROR = repr(_e)

# -----------------------------
# Basic constants and config
# -----------------------------
BRAND_NAME = "Sergey Krichevskiy Group"
APP_ENV = os.getenv("APP_ENV", "PROD").upper()  # "DEV" or "PROD"
UI_DEFAULT_MODEL = os.getenv("UI_DEFAULT_MODEL", "").strip()

# Neutral internal model IDs (mapped to provider-specific IDs in Phase 6)
MODEL_OPTIONS = [
    {"id": "chatgpt-5-micro", "label": "ChatGPT 5.0 Micro", "desc": "for testing / lowest cost"},
    {"id": "chatgpt-5-mini",  "label": "ChatGPT 5.0 Mini",  "desc": "balance of quality & cost"},
    {"id": "chatgpt-5",       "label": "ChatGPT 5.0",       "desc": "highest quality"},
]

# Default paths where Phase 4 logs might live (can be adjusted later)
DEFAULT_RUNS_DIR = Path("artifacts/v4/runs")
DEFAULT_HISTORY_FILE = Path("runs_history.jsonl")

# -----------------------------
# Helpers (Phase 5 stubs)
# -----------------------------
def read_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    """Try to read a JSON file; return None if not found or invalid. Never writes."""
    try:
        if path.exists() and path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def read_optional_text(path: Path, max_bytes: int = 200_000) -> Optional[str]:
    """Read small text file if present; cap size for safety."""
    try:
        if path.exists() and path.is_file():
            data = path.read_bytes()[:max_bytes]
            return data.decode("utf-8", errors="replace")
    except Exception:
        return None
    return None


def list_recent_run_files(runs_dir: Path, limit: int = 10) -> List[Path]:
    """List recent run files in artifacts folder (best-effort)."""
    if not runs_dir.exists() or not runs_dir.is_dir():
        return []
    files = [p for p in runs_dir.glob("**/*") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def read_history_lines(history_file: Path, limit: int = 50) -> List[str]:
    """Read last N lines from runs_history.jsonl if exists."""
    if not history_file.exists() or not history_file.is_file():
        return []
    try:
        content = history_file.read_text(encoding="utf-8", errors="replace").splitlines()
        return content[-limit:]
    except Exception:
        return []


def generate_answer_stub(query: str, model_id: str) -> Dict[str, Any]:
    """
    Placeholder "local stub" generator to simulate a response in Phase 5.
    No external API calls. No file modifications.
    """
    start = time.time()
    time.sleep(0.2)

    quality_hint = {
        "chatgpt-5-micro": "Test-mode response (lowest fidelity).",
        "chatgpt-5-mini": "Balanced response (mid fidelity).",
        "chatgpt-5": "High-quality response (highest fidelity).",
    }.get(model_id, "Generic response.")

    answer = (
        f"{quality_hint} This is a placeholder answer for your query:\n\n"
        f"Q: {query}\n\n"
        "In Phase 6, this will be replaced with real API-backed output."
    )
    elapsed = time.time() - start
    return {
        "answer": answer,
        "elapsed_sec": round(elapsed, 3),
        "model_id": model_id,
    }


def get_model_label(model_id: str) -> str:
    for m in MODEL_OPTIONS:
        if m["id"] == model_id:
            return m["label"]
    return model_id


def default_model_index() -> int:
    """Resolve default model from UI_DEFAULT_MODEL env var; fallback to 0."""
    if UI_DEFAULT_MODEL:
        for i, m in enumerate(MODEL_OPTIONS):
            if m["id"] == UI_DEFAULT_MODEL:
                return i
    return 0


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="MVP UI", page_icon="🤖", layout="wide")

# Header / brand
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:12px;">
      <div style="font-size:1.2rem;font-weight:600;">{BRAND_NAME}</div>
      <div style="opacity:0.6;">MVP User Interface (Phase 5)</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# Sidebar: global controls
with st.sidebar:
    st.subheader("Mode")
    mode = st.radio(
        "Select how to run:",
        ["Test Mode (local stub)", "Connect API"],
        index=0,
        help="Phase 5: Test Mode only. API wiring comes in Phase 6.",
    )

    # Model picker (always visible; runtime-only, never writes to JSON)
    st.subheader("Model")
    model_labels = [f'{m["label"]} — {m["desc"]}' for m in MODEL_OPTIONS]
    model_choice_label = st.selectbox(
        "Choose a model:",
        model_labels,
        index=default_model_index(),
        help="Runtime-only selection. Does not modify any config files.",
    )
    model_id = MODEL_OPTIONS[model_labels.index(model_choice_label)]["id"]
    # Keep in session state (runtime param only)
    st.session_state["current_model_id"] = model_id

    if mode == "Connect API":
        st.subheader("API Settings")
        api_key_mode = st.radio(
            "API key source:",
            ["Use default server key", "Paste API key"],
            index=0,
            help="One key works for all models. Actual API calls will be enabled in Phase 6.",
        )
        if api_key_mode == "Paste API key":
            _ = st.text_input("Enter API key:", type="password")  # stored in Streamlit widget state only

    st.markdown("---")
    st.caption(f"Environment: **{APP_ENV}**")

# Main content
st.title("Chat")
user_query = st.text_area("Your question", height=120, placeholder="Type your question here...")
run_clicked = st.button("Ask")

# Placeholder containers
answer_container = st.empty()
meta_container = st.empty()

# Run logic (no external API in Phase 5)
if run_clicked:
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking (stub)..."):
            result = generate_answer_stub(user_query.strip(), model_id=st.session_state["current_model_id"])
        answer_container.markdown(
            f"**Model:** {get_model_label(result['model_id'])}\n\n"
            f"**Answer (stub):**\n\n{result['answer']}"
        )
        meta_container.info(f"Elapsed: {result['elapsed_sec']} sec | Mode: {mode}")

# Developer Panel (DEV only)
if APP_ENV == "DEV":
    st.markdown("---")
    dev_expander = st.expander("Developer Panel (DEV only)", expanded=False)
    with dev_expander:
        enable_dev = st.checkbox("Enable Developer Panel")
        if enable_dev:
            st.write("Server-side logs (best-effort):")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Recent run files**")
                files = list_recent_run_files(DEFAULT_RUNS_DIR, limit=10)
                if not files:
                    st.caption("No recent run files found.")
                else:
                    for p in files:
                        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
                        st.write(f"- {p}  \n  _modified: {ts}_")

            with col2:
                st.markdown("**runs_history.jsonl (last 50 lines)**")
                lines = read_history_lines(DEFAULT_HISTORY_FILE, limit=50)
                if not lines:
                    st.caption("No history file or empty.")
                else:
                    st.code("\n".join(lines), language="json")

            # Optional snapshots for DEV convenience (read-only)
            st.markdown("---")
            st.markdown("**Optional snapshots (if present)**")
            system_prompt_text = read_optional_text(Path("system_prompt.txt"))
            if system_prompt_text:
                st.markdown("`system_prompt.txt` (first 400 chars):")
                st.code(system_prompt_text[:400] + ("..." if len(system_prompt_text) > 400 else ""), language="markdown")

            model_cfg = read_optional_json(Path("model.config.json"))
            if model_cfg:
                st.markdown("`model.config.json` (trimmed):")
                st.json(model_cfg)

            rag_cfg = read_optional_json(Path("rag_config.json"))
            if rag_cfg:
                st.markdown("`rag_config.json` (trimmed):")
                st.json(rag_cfg)

            # ---------------------------------------------
            # NEW: End-to-end test runner (Phase 5 local)
            # ---------------------------------------------
            st.markdown("---")
            st.markdown("### End-to-end test (local Phase-4 pipeline)")
            if not _E2E_AVAILABLE:
                st.error(
                    "E2E runner is unavailable: failed to import Phase-4 modules "
                    f"(llm_integration.*). Import error: {_E2E_IMPORT_ERROR}"
                )
            else:
                test_q = st.text_input(
                    "Test question",
                    value="Как мне начать вести личный бюджет?",
                    help="This will run: retrieve_context → generate_answer → log_phase4_run."
                )
                run_e2e = st.button("Run end-to-end test")
                if run_e2e:
                    if not test_q.strip():
                        st.warning("Please enter a test question.")
                    else:
                        with st.spinner("Running local pipeline..."):
                            try:
                                # 1) Retrieve
                                context_text, refs = _retrieve_context(test_q.strip(), k=5)
                                # 2) Generate
                                gen_result = _gen_answer(context_text, test_q.strip())
                                # 3) Resolve model & log
                                llm = _get_llm()
                                model_name = getattr(llm, "model", "unknown-model")
                                _log_run(
                                    model_name=model_name,
                                    question=test_q.strip(),
                                    context_text=context_text,
                                    refs=refs,
                                    answer_text=gen_result["final_output"]
                                )
                                # 4) Show outputs
                                st.success("E2E run completed.")
                                st.markdown("**Final Output**")
                                st.write(gen_result["final_output"])

                                st.markdown("**Developer Output**")
                                st.json({
                                    "refs": refs,
                                    "llm_debug": gen_result.get("developer_output")
                                })

                            except Exception as e:
                                st.exception(e)
else:
    # In PROD, keep panel hidden
    pass

# Footer
st.markdown("---")
st.caption("Phase 5 MVP • No external API calls • Ready for Phase 6 wiring & deploy prep.")
