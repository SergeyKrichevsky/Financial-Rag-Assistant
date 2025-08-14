# file: llm_integration/answer_generator.py
# Minimal generator wired to the router + system prompt loader.

import os
from typing import Dict
from .llm_router import get_llm

PROMPT_PATH = "configs/system_prompt.txt"  # change to "configs/system_prompt.txt" if you kept that folder


def load_system_prompt(path: str = PROMPT_PATH) -> str:
    """Load the system prompt from a text file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"System prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

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

    # Build user prompt (keep sources hidden; we pass only plain context)
    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Relevant context (do NOT mention its origin in the answer):\n{context}\n\n"
        "Now answer following the system instructions."
    )

    # IMPORTANT: call .complete(system_prompt, user_prompt)
    answer_text = llm.complete(system_prompt, user_prompt)

    return {
        "final_output": answer_text,
        "developer_output": {
            "context_used": context,
            "full_prompt": f"{system_prompt}\n\n{user_prompt}",
        },
    }
