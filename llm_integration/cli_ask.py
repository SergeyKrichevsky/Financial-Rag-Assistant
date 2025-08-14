# file: llm_integration/cli_ask.py
# Simple CLI for Phase 4: retrieve → generate → print answer (+dev option).
# Usage examples:
#   python -m llm_integration.cli_ask --q "Как начать вести бюджет?"
#   python -m llm_integration.cli_ask --q "How to build an emergency fund?" --k 6 --show-dev

# The parameter `k` defines how many top-matching context chunks 
# the retriever should return for the given question.
# 
# - The retriever searches the vector database and finds the `k` most relevant text fragments.
# - These fragments are combined and passed to the LLM as "Relevant context".
# 
# Tips:
# - Small values (k=3–5) work well for short, focused questions.
# - Larger values (k=8–10) may be better for broader questions that require more context.
# - Very high values can increase token usage and may add noise, reducing accuracy.

from __future__ import annotations
import argparse, json
from typing import Dict
from .retriever_bridge import retrieve_context
from .answer_generator import generate_answer
from .llm_router import get_llm
from .run_logger import log_phase4_run

def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the RAG assistant (Phase 4).")
    parser.add_argument("--q", "--query", dest="query", required=True, help="User question")
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks to use after filtering")
    parser.add_argument("--show-dev", action="store_true", help="Print developer metadata")
    args = parser.parse_args()

    # 1) Retrieve context and refs
    context_text, refs = retrieve_context(args.query, k=args.k)

    # 2) Generate answer
    result: Dict[str, object] = generate_answer(context_text, args.query)

    # 3) Log run
    llm = get_llm()
    model_name = getattr(llm, "model", "unknown-model")
    log_phase4_run(
        model_name=model_name,
        question=args.query,
        context_text=context_text,
        refs=refs,
        answer_text=str(result.get("final_output", "")),
    )

    # 4) Print outputs
    print("\n=== Final Answer ===")
    print(result["final_output"])

    if args.show_dev:
        print("\n=== Developer Output ===")
        dev = {
            "refs": refs,
            "llm_debug": result["developer_output"],
        }
        print(json.dumps(dev, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
