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
# - This UI REQUIRES configs/models.ui.json. If missing/invalid → hard error.
# - No fallbacks or built-in defaults are used.
# - UI_DEFAULT_MODEL is optional; must match an ID from models.ui.json if set.
# - Phase 5 does NOT call any external APIs and does NOT modify any JSON files.
# - Developer Panel reads server-side logs; E2E test runs local Phase-4 pipeline.
# =========================================

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import streamlit as st

# Ensure project root on sys.path (so llm_integration is importable regardless of CWD)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional imports for E2E run (Phase 5 local pipeline).
try:
    from llm_integration.answer_generator import generate_answer as _gen_answer
    from llm_integration.retriever_bridge import retrieve_context as _retrieve_context
    from llm_integration.llm_router import get_llm as _get_llm
    from llm_integration.run_logger import log_phase4_run as _log_run
    _E2E_AVAILABLE = True
except Exception as _e:
    _E2E_AVAILABLE = False
    _E2E_IMPORT_ERROR = repr(_e)

BRAND_NAME = "Sergey Krichevskiy Group"
APP_ENV = os.getenv("APP_ENV", "PROD").upper()  # "DEV" or "PROD"
UI_DEFAULT_MODEL = os.getenv("UI_DEFAULT_MODEL", "").strip()

# Strict models source (JSON file). No fallback allowed.
MODELS_JSON_PATH = ROOT / "configs" / "models.ui.json"

# Paths for logs (Phase 4)
DEFAULT_RUNS_DIR = Path("artifacts/v4/runs")
DEFAULT_HISTORY_FILE = Path("runs_history.jsonl")

# -----------------------------
# Safe file helpers
# -----------------------------
def read_required_json(path: Path) -> Dict[str, Any]:
    """Read mandatory JSON file or raise ValueError with a clear message."""
    if not path.exists() or not path.is_file():
        raise ValueError(f"Required file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {path} ({e})") from e

def read_optional_text(path: Path, max_bytes: int = 200_000) -> Optional[str]:
    try:
        if path.exists() and path.is_file():
            data = path.read_bytes()[:max_bytes]
            return data.decode("utf-8", errors="replace")
    except Exception:
        return None
    return None

def list_recent_run_files(runs_dir: Path, limit: int = 10) -> List[Path]:
    if not runs_dir.exists() or not runs_dir.is_dir():
        return []
    files = [p for p in runs_dir.glob("**/*") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]

def read_history_lines(history_file: Path, limit: int = 50) -> List[str]:
    if not history_file.exists() or not history_file.is_file():
        return []
    try:
        content = history_file.read_text(encoding="utf-8", errors="replace").splitlines()
        return content[-limit:]
    except Exception:
        return []

# -----------------------------
# Load models from JSON (STRICT)
# -----------------------------
def load_model_options_strict(path: Path) -> List[Dict[str, Any]]:
    """
    Load and validate models list from models.ui.json.
    Expected schema:
      {
        "models": [
          {"id": "...", "label": "...", "desc": "...", "enabled": true},
          ...
        ]
      }
    Rules:
      - File must exist and be valid JSON → otherwise raise.
      - Use ONLY items with enabled != false and with non-empty id & label.
      - If no valid items remain → raise.
    """
    data = read_required_json(path)
    if not isinstance(data, dict) or "models" not in data or not isinstance(data["models"], list):
        raise ValueError(f"Invalid schema in {path}: expected object with 'models' array")

    cleaned: List[Dict[str, Any]] = []
    for item in data["models"]:
        if not isinstance(item, dict):
            continue
        if item.get("enabled", True) is False:
            continue
        mid = str(item.get("id", "")).strip()
        lbl = str(item.get("label", "")).strip()
        dsc = str(item.get("desc", "")).strip()
        if not mid or not lbl:
            continue
        cleaned.append({"id": mid[:200], "label": lbl[:200], "desc": dsc[:500]})

    if not cleaned:
        raise ValueError(f"No enabled models with valid 'id' and 'label' in {path}")
    return cleaned

# Phase 5 stub generator (for regular Chat tab)
def generate_answer_stub(query: str, model_id: str) -> Dict[str, Any]:
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
    return {"answer": answer, "elapsed_sec": round(elapsed, 3), "model_id": model_id}

def get_model_label(model_id: str, options: List[Dict[str, Any]]) -> str:
    for m in options:
        if m["id"] == model_id:
            return m["label"]
    return model_id

def default_model_index(options: List[Dict[str, Any]]) -> int:
    if UI_DEFAULT_MODEL:
        for i, m in enumerate(options):
            if m["id"] == UI_DEFAULT_MODEL:
                return i
    return 0

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="MVP UI", page_icon="🤖", layout="wide")

# Load models strictly; fail fast with a clear UI error if file missing/invalid
try:
    MODEL_OPTIONS = load_model_options_strict(MODELS_JSON_PATH)
except Exception as err:
    st.error(
        f"Models configuration error: {err}\n\n"
        f"Please create and validate the file:\n**{MODELS_JSON_PATH}**"
    )
    st.stop()

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

# Sidebar controls
with st.sidebar:
    st.subheader("Mode")
    mode = st.radio(
        "Select how to run:",
        ["Test Mode (local stub)", "Connect API"],
        index=0,
        help="Phase 5: Test Mode only. API wiring comes in Phase 6.",
    )

    st.subheader("Model")
    model_labels = [
        f'{m["label"]} — {m["desc"]}' if m.get("desc") else m["label"]
        for m in MODEL_OPTIONS
    ]
    model_choice_label = st.selectbox(
        "Choose a model:",
        model_labels,
        index=min(default_model_index(MODEL_OPTIONS), max(0, len(model_labels) - 1)),
        help="Runtime-only selection. Does not modify any config files.",
    )
    model_id = MODEL_OPTIONS[model_labels.index(model_choice_label)]["id"]
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
            _ = st.text_input("Enter API key:", type="password")

    st.markdown("---")
    st.caption(f"Environment: **{APP_ENV}**")

# Main chat tab (stubbed)
st.title("Chat")
user_query = st.text_area("Your question", height=120, placeholder="Type your question here...")
run_clicked = st.button("Ask")

answer_container = st.empty()
meta_container = st.empty()

if run_clicked:
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking (stub)..."):
            result = generate_answer_stub(user_query.strip(), model_id=st.session_state["current_model_id"])
        answer_container.markdown(
            f"**Model:** {get_model_label(result['model_id'], MODEL_OPTIONS)}\n\n"
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

            # End-to-end test (local Phase-4 pipeline)
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
                    help="This runs: retrieve_context → generate_answer → log_phase4_run."
                )
                run_e2e = st.button("Run end-to-end test")
                if run_e2e:
                    if not test_q.strip():
                        st.warning("Please enter a test question.")
                    else:
                        with st.spinner("Running local pipeline..."):
                            try:
                                context_text, refs = _retrieve_context(test_q.strip(), k=5)
                                gen_result = _gen_answer(context_text, test_q.strip())
                                llm = _get_llm()
                                model_name = getattr(llm, "model", "unknown-model")
                                _log_run(
                                    model_name=model_name,
                                    question=test_q.strip(),
                                    context_text=context_text,
                                    refs=refs,
                                    answer_text=gen_result["final_output"]
                                )
                                st.success("E2E run completed.")
                                st.markdown("**Final Output**")
                                st.write(gen_result["final_output"])

                                st.markdown("**Developer Output**")
                                st.json({"refs": refs, "llm_debug": gen_result.get("developer_output")})
                            except Exception as e:
                                st.exception(e)

# Footer
st.markdown("---")
st.caption("Phase 5 MVP • No external API calls • Ready for Phase 6 wiring & deploy prep.")
