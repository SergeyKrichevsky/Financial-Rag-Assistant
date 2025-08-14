# file: llm_integration/smoke_test_openai.py
# Minimal OpenAI smoke test: verifies env key, model availability, network path.
# It does NOT touch the retriever or book context — just LLM call.
#
# Usage:
#   python -m llm_integration.smoke_test_openai
#
# Requirements (when backend=openai in model.config.json):
#   - OPENAI_API_KEY must be set in your OS environment.

from __future__ import annotations
import time
from .llm_router import get_llm

def main() -> None:
    llm = get_llm()  # will raise if OPENAI_API_KEY is missing and backend=openai

    system = (
        "You are a concise assistant. Reply in one short paragraph (~2–3 sentences)."
    )
    user = "Smoke test: just confirm you can respond to this ping."

    t0 = time.time()
    try:
        out = llm.complete(system, user)
        ms = int((time.time() - t0) * 1000)
        print("SMOKETEST STATUS: OK")
        print(f"MODEL: {getattr(llm, 'model', 'unknown')}")
        print(f"LATENCY_MS: {ms}")
        print("RESPONSE:", (out or "").strip()[:300].replace("\n", " "))
    except Exception as e:
        ms = int((time.time() - t0) * 1000)
        print("SMOKETEST STATUS: FAIL")
        print(f"MODEL: {getattr(llm, 'model', 'unknown')}")
        print(f"LATENCY_MS: {ms}")
        print(f"ERROR: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
