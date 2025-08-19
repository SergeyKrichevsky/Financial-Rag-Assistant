"""
Streamlit UI with two modes:
  1) Assistant API (default) — uses OpenAI Assistants API + File Search
  2) Local RAG (existing) — calls your legacy pipeline if importable

Features:
  - API key handling: "Use server key (ENV)" or "Paste manually"
  - Assistant ID autodetect from configs/assistant.meta.json (editable)
  - Thread continuity within a browser session
  - Clean error reporting

Run examples:
  APP_ENV=DEV streamlit run Interface/app.py
  APP_ENV=PROD streamlit run Interface/app.py

Dependencies:
  pip install -U streamlit openai

Notes:
  - This file is self-contained: it does not require the earlier helper module.
  - If your legacy RAG functions are available, the code will use them; otherwise it shows a friendly notice.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
import sys
from pathlib import Path as _P
# Ensure project root is importable so `llm_integration` can be found when running from /Interface
_project_root = _P(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from openai import OpenAI  # official SDK v1.x
    import openai as openai_module  # to report SDK version if needed
except Exception:
    OpenAI = None  # we will validate later and show a helpful message
    openai_module = None

# ---------- Constants & paths ----------
ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).name == "app.py") else Path.cwd()
CONFIGS_DIR = ROOT / "configs"
ASSISTANT_META_PATH = CONFIGS_DIR / "assistant.meta.json"

# ---------- Helpers ----------

def load_assistant_meta() -> dict:
    if ASSISTANT_META_PATH.exists():
        try:
            return json.loads(ASSISTANT_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def get_client(api_key: Optional[str]) -> Optional[OpenAI]:
    if OpenAI is None:
        return None
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None


def _get_threads_api(client: OpenAI):
    """Return (threads_api, track) where track is 'stable' or 'beta'.
    Supports both modern `client.threads` and legacy `client.beta.threads`.
    """
    threads_api = getattr(client, "threads", None)
    if threads_api is not None:
        return threads_api, "stable"
    beta = getattr(client, "beta", None)
    if beta is not None and getattr(beta, "threads", None) is not None:
        return beta.threads, "beta"
    return None, None

def _threads_api(client):
    """Return Threads API namespace, supporting both stable and beta paths."""
    if hasattr(client, "threads"):
        return client.threads  # new SDKs
    beta = getattr(client, "beta", None)
    if beta is not None and hasattr(beta, "threads"):
        return beta.threads   # older SDKs
    raise AttributeError("OpenAI SDK is too old: no 'threads' or 'beta.threads'. Update: pip install -U openai")

def ensure_thread(client) -> str:
    """Create a thread once per browser session; reuse thereafter."""
    if "thread_id" not in st.session_state:
        thread = _threads_api(client).create()
        st.session_state.thread_id = thread.id
    return st.session_state.thread_id


def run_assistant(client, assistant_id: str, user_text: str):
    """Send a user message to the assistant and return (answer_text, meta)."""
    thread_id = ensure_thread(client)

    # 1) Add the user message
    _threads_api(client).messages.create(
        thread_id=thread_id,
        role="user",
        content=user_text,
    )

    # 2) Run the assistant and poll until completion
    run = _threads_api(client).runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )

    # 3) Read the latest assistant message
    messages = _threads_api(client).messages.list(thread_id=thread_id, order="desc", limit=10)
    answer_text = ""
    for m in messages.data:
        if getattr(m, "role", None) != "assistant":
            continue
        for part in getattr(m, "content", []) or []:
            if getattr(part, "type", None) == "text":
                answer_text = getattr(getattr(part, "text", None), "value", "") or ""
                if answer_text:
                    break
        if answer_text:
            break

    meta = {
        "thread_id": thread_id,
        "run_status": getattr(run, "status", None),
        "model": getattr(run, "model", None),
    }
    return (answer_text or "[No assistant text found]"), meta


# ---- Optional: hook to local RAG if present ----

def answer_via_local_rag(question: str) -> str:
    """Call the same Phase-4 pipeline your original UI used:
    llm_integration.retriever_bridge.retrieve_context -> llm_integration.answer_generator.generate_answer
    Returns final text. If modules are not available, returns a short friendly hint.
    """
    import importlib
    try:
        retrieve_context = importlib.import_module(
            "llm_integration.retriever_bridge"
        ).retrieve_context
        generate_answer = importlib.import_module(
            "llm_integration.answer_generator"
        ).generate_answer

        ctx_text, _refs = retrieve_context(question.strip(), k=5)
        res = generate_answer(ctx_text, question.strip())
        return res.get("final_output") or res.get("answer") or str(res)
    except Exception as e:
        # Keep the fallback message one-line to avoid syntax issues in some editors
        return f"Local RAG selected, but pipeline modules not found or failed: {e}. Ensure 'llm_integration' is on PYTHONPATH."


# ---------- UI ----------

st.set_page_config(page_title="Financial RAG Assistant", page_icon="💬", layout="wide")

st.title("Financial Assistant — Two Modes")

with st.sidebar:
    st.subheader("Context Source")
    mode = st.radio(
        "Choose backend",
        options=["Assistant API (default)", "Local RAG (existing)"],
        index=0,
        help="Switch between OpenAI Assistants (with File Search) and your legacy RAG.",
    )

    st.markdown("---")
    st.subheader("OpenAI API key")
    key_mode = st.radio("Key source", ["Use server ENV", "Paste manually"], index=0)
    manual_key = None
    if key_mode == "Paste manually":
        manual_key = st.text_input("Enter API key", type="password", placeholder="sk-...", help="The key is kept in session only.")

    st.markdown("---")
    st.subheader("Assistant")
    meta = load_assistant_meta()
    default_assistant_id = meta.get("assistant_id", "") if isinstance(meta, dict) else ""
    assistant_id = st.text_input("assistant_id", value=default_assistant_id, placeholder="asst_...")

    if mode.startswith("Assistant API") and not assistant_id:
        st.info("Tip: create an Assistant in OpenAI Platform and paste its ID here. If configs/assistant.meta.json exists, it's pre-filled.")

# Conversation history box
if "chat" not in st.session_state:
    st.session_state.chat = []  # list[(role, text)]

# Input area
user_text = st.chat_input("Ask a question about the book…")

# Display existing transcripts
for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

# Handle new input
if user_text:
    st.session_state.chat.append(("user", user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    if mode.startswith("Assistant API"):
        # Validate SDK and key
        key_in_use = manual_key if (key_mode == "Paste manually" and manual_key) else os.getenv("OPENAI_API_KEY")
        if OpenAI is None:
            with st.chat_message("assistant"):
                st.error("Python package 'openai' is not installed. Run: pip install -U openai")
        elif not key_in_use:
            with st.chat_message("assistant"):
                st.error("No API key. Set OPENAI_API_KEY in ENV or paste it in the sidebar.")
        elif not assistant_id:
            with st.chat_message("assistant"):
                st.error("No assistant_id. Paste the ID in the sidebar.")
        else:
            try:
                client = get_client(key_in_use)
                if client is None:
                    raise RuntimeError("Failed to initialize OpenAI client.")
                with st.chat_message("assistant"):
                    with st.spinner("Running assistant…"):
                        answer, meta = run_assistant(client, assistant_id, user_text)
                        st.session_state.chat.append(("assistant", answer))
                        st.markdown(answer)
                        st.caption(f"model: {meta.get('model')} | thread: {meta.get('thread_id')}")
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"Assistant API error: {e}")
    else:
        # Local RAG mode
        with st.chat_message("assistant"):
            with st.spinner("Querying local RAG…"):
                try:
                    answer = answer_via_local_rag(user_text)
                except Exception as e:
                    answer = f"Local RAG error: {e}"
                st.session_state.chat.append(("assistant", answer))
                st.markdown(answer)

# Footer
st.markdown("---")
st.caption(
    "Backend: OpenAI Assistants API (threads/runs) when selected; otherwise attempts to call your legacy RAG pipeline.\n"
    "No keys are stored on disk; manual keys live only in the current Streamlit session."
)
