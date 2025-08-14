# file: llm_integration/answer_generator.py
# Generator wired to: router + system prompt loader + context sanitizer + token cap.
# Reads parameters (including off-switch flags) from configs/rag_config.json.

from __future__ import annotations
import os
from typing import Dict, Iterable, List

from .llm_router import get_llm
from .config_loader import get_generator_cfg

# ---- Load generator config (with safe defaults) ----
_GEN = get_generator_cfg()  # {system_prompt_path, max_context_chars, max_context_tokens, token_encoding, use_sanitizer, use_token_cap}

PROMPT_PATH: str = _GEN.get("system_prompt_path", "configs/system_prompt.txt")
MAX_CONTEXT_CHARS: int = int(_GEN.get("max_context_chars", 6000))
MAX_CONTEXT_TOKENS: int = int(_GEN.get("max_context_tokens", 900))
TOKEN_ENCODING: str = str(_GEN.get("token_encoding", "cl100k_base"))
USE_SANITIZER: bool = bool(_GEN.get("use_sanitizer", True))
USE_TOKEN_CAP: bool = bool(_GEN.get("use_token_cap", True))

def load_system_prompt(path: str = PROMPT_PATH) -> str:
    """Load the system prompt from a text file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"System prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def _dedupe_keep_order(lines: Iterable[str]) -> List[str]:
    """Remove exact-duplicate lines while preserving order."""
    seen = set()
    out: List[str] = []
    for ln in lines:
        key = ln.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(ln)
    return out

def sanitize_context(raw_context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Cleanup to avoid sending noisy/oversized context to the LLM.
    - Split by blank lines → dedupe → rejoin.
    - Trim whitespace.
    - Hard-cap by characters to keep the prompt budget reasonable.
    """
    parts = [p.strip() for p in (raw_context or "").split("\n\n")]
    parts = [p for p in parts if p]  # drop empties
    parts = _dedupe_keep_order(parts)

    text = "\n\n".join(parts).strip()
    if len(text) > max_chars:
        # Cut on paragraph boundary if possible
        cut = text[:max_chars]
        text = (cut.rsplit("\n\n", 1)[0] or cut).strip()
    return text

def _trim_to_tokens(text: str, max_tokens: int, encoding_name: str = TOKEN_ENCODING) -> str:
    """
    Trim text to at most `max_tokens`. Uses `tiktoken` if available; otherwise
    falls back to approximate word-based trimming.
    """
    if max_tokens <= 0 or not text:
        return ""
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding(encoding_name)
        ids = enc.encode(text)
        if len(ids) <= max_tokens:
            return text
        return enc.decode(ids[:max_tokens])
    except Exception:
        # Fallback: 1 word ≈ 1 token (rough but safe)
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])

def generate_answer(context: str, question: str) -> Dict[str, object]:
    """
    Generate answer using the selected LLM (configured via model.config.json / env).
    On errors, return a friendly fallback for the end user and include diagnostics
    in 'developer_output'.
    """
    llm = get_llm()  # central router (local_stub or openai)
    system_prompt = load_system_prompt()

    # 1) Optional sanitizer (char cap + dedupe)
    clean_context = sanitize_context(context, max_chars=MAX_CONTEXT_CHARS) if USE_SANITIZER else (context or "")

    # 2) Optional token cap
    capped_context = _trim_to_tokens(clean_context, max_tokens=MAX_CONTEXT_TOKENS, encoding_name=TOKEN_ENCODING) if USE_TOKEN_CAP else clean_context

    # 3) Build user prompt (keep sources hidden; pass only plain context)
    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Relevant context (do NOT mention its origin in the answer):\n{capped_context}\n\n"
        "Now answer following the system instructions."
    )

    # 4) Call LLM with basic error handling
    try:
        answer_text = llm.complete(system_prompt, user_prompt)
        return {
            "final_output": answer_text,
            "developer_output": {
                "context_used": capped_context,
                "full_prompt": f"{system_prompt}\n\n{user_prompt}",
                "limits": {
                    "use_sanitizer": USE_SANITIZER,
                    "use_token_cap": USE_TOKEN_CAP,
                    "max_context_chars": MAX_CONTEXT_CHARS,
                    "max_context_tokens": MAX_CONTEXT_TOKENS,
                    "token_encoding": TOKEN_ENCODING,
                },
            },
        }
    except RuntimeError as e:
        msg = str(e)
        user_fallback = (
            "The assistant is temporarily unavailable.\n"
            "Please configure the OpenAI API key or switch the backend to 'local_stub' in model.config.json."
        ) if "OPENAI_API_KEY" in msg else (
            "The assistant encountered a temporary issue. Please try again in a moment."
        )
        return {
            "final_output": user_fallback,
            "developer_output": {
                "error": {"type": "RuntimeError", "message": msg},
                "context_used": capped_context,
                "full_prompt": f"{system_prompt}\n\n{user_prompt}",
            },
        }
    except Exception as e:
        return {
            "final_output": "The assistant hit an unexpected error. Please try again shortly.",
            "developer_output": {
                "error": {"type": type(e).__name__, "message": str(e)},
                "context_used": capped_context,
                "full_prompt": f"{system_prompt}\n\n{user_prompt}",
            },
        }
