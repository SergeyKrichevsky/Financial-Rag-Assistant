# file: llm_integration/test_generate.py
# Minimal end-to-end test: retrieve → generate → log

# Running comand:
#   python -m llm_integration.test_generate


from llm_integration.answer_generator import generate_answer
from llm_integration.retriever_bridge import retrieve_context
from llm_integration.llm_router import get_llm
from llm_integration.run_logger import log_phase4_run

# 1) Question to test
question = "Как мне начать вести личный бюджет?"

# 2) Retrieve context + developer refs
context_text, refs = retrieve_context(question, k=5)

# 3) Generate answer using current LLM (whatever is set in model.config.json / env)
result = generate_answer(context_text, question)

# 4) Resolve model name for logging (from the same router)
llm = get_llm()
model_name = getattr(llm, "model", "unknown-model")

# 5) Log Phase 4 run (writes artifacts/v4/runs/last_run_phase4.json)
log_phase4_run(
    model_name=model_name,
    question=question,
    context_text=context_text,
    refs=refs,
    answer_text=result["final_output"]
)

# 6) Show results in console
print("\n=== Final Output ===")
print(result["final_output"])

print("\n=== Developer Output ===")
print({
    "refs": refs,
    "llm_debug": result["developer_output"]
})
