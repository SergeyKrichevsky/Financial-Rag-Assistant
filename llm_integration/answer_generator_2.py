# file: llm_integration/answer_generator.py
# Minimal generator wired to the router + system prompt loader + context sanitizer.

from __future__ import annotations
import os
from typing import Dict, Iterable, List
from .llm_router import get_llm  # relative import inside package

PROMPT_PATH = "configs/system_prompt.txt"  # adjust if your file lives elsewhere

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

def sanitize_context(raw_context: str, max_chars: int = 6000) -> str:
    """
    Light cleanup to avoid sending noisy/oversized context to the LLM.
    - Split by blank lines → dedupe → rejoin.
    - Trim trailing/leading whitespace.
    - Hard-cap by characters to keep the prompt budget reasonable.
    """
    # Split paragraphs by double newlines
    parts = [p.strip() for p in raw_context.split("\n\n")]
    parts = [p for p in parts if p]  # drop empties
    parts = _dedupe_keep_order(parts)

    # Rejoin and cap
    text = "\n\n".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n\n", 1)[0].strip()  # cut on paragraph boundary if possible
    return text

def generate_answer(context: str, question: str) -> Dict[str, object]:
    """
    Generate answer using the selected LLM (configured via model.config.json / env).
    Returns:
      {
        "final_output": <text for end user>,
        "developer_output": {"context_used": ..., "full_prompt": ...}
      }
    """
    llm = get_llm()  # central router (local_stub or openai)
    system_prompt = load_system_prompt()

    # Sanitize context before sending to the model
    clean_context = sanitize_context(context)

    # Build user prompt (keep sources hidden; we pass only plain context)
    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Relevant context (do NOT mention its origin in the answer):\n{clean_context}\n\n"
        "Now answer following the system instructions."
    )

    # IMPORTANT: call .complete(system_prompt, user_prompt)
    answer_text = llm.complete(system_prompt, user_prompt)

    return {
        "final_output": answer_text,
        "developer_output": {
            "context_used": clean_context,
            "full_prompt": f"{system_prompt}\n\n{user_prompt}",
        },
    }
