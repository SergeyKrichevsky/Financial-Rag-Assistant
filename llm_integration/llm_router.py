# file: llm_integration/llm_router.py
# Single switchboard to choose which LLM the pipeline will use.
# - Keep pipeline code stable: always call get_llm().complete(system, user).
# - Start free: local stub (no keys). Later: flip to OpenAI (e.g., gpt-5-mini).
# - Reads config from model.config.json (optional) and env vars.
#
# JSON config schema (model.config.json):
#   {
#     "backend": "local_stub" | "openai",
#     "model": "gpt-5-mini",
#     "temperature": 0.3
#   }
#
# Environment variables (override file/defaults):
#   LLM_BACKEND = local_stub | openai
#   LLM_MODEL   = e.g. gpt-5-mini | gpt-5-nano | gpt-4.1
#   OPENAI_API_KEY = <your key when using openai>

# hint (How to use):
# from llm_integration.llm_router import get_llm
# llm = get_llm()  # no changes later; switch models via model.config.json/env
# answer = llm.complete(system_prompt, user_prompt)

from __future__ import annotations
from typing import Optional, Dict, Any
import os, json, importlib

# ---------- Base interface ----------

class BaseLLM:
    """Uniform interface used by the pipeline."""
    def __init__(self, model: str, temperature: float = 0.3):
        self.model = model
        self.temperature = float(temperature)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return plain-text completion."""
        raise NotImplementedError

# ---------- Free default: Local Stub ----------
# Tries to use a very small HF model if available; otherwise returns a template.
class LocalStubLLM(BaseLLM):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return (
                "This is a local stub response (no API key required).\n\n"
                "Concept: Explain the idea clearly based on the provided context.\n"
                "Motivation: Encourage consistent, realistic financial habits.\n"
                "Action steps: Provide 3–5 practical steps.\n\n"
                "Note: Switch to OpenAI by changing model.config.json or env vars."
            )

# ---------- OpenAI backend ----------

class OpenAILLM(BaseLLM):
    def __init__(self, model: str, temperature: float = 0.3, api_key: Optional[str] = None):
        super().__init__(model=model, temperature=temperature)
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Set it in your environment.")
        # Lazy import to avoid dependency during local_stub use.
        from openai import OpenAI
        self._client = OpenAI(api_key=self._api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_client()
        resp = self._client.chat.completions.create(
            model=self.model,  # e.g., "gpt-5-mini" for MVP
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

# ---------- Factory & config loading ----------

DEFAULTS: Dict[str, Any] = {
    "backend": "local_stub",   # free-by-default for development
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "temperature": 0.3,
}

def _normalize_model_name(backend: str, model: str) -> str:
    """Optional alias normalization to avoid typos."""
    if backend == "openai":
        aliases = {
            "gpt5-mini": "gpt-5-mini",
            "gpt5-nano": "gpt-5-nano",
            "gpt4.1": "gpt-4.1",
            "gpt5": "gpt-5",
        }
        return aliases.get(model, model)
    return model

def _read_file_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise RuntimeError(f"Failed to read '{path}': {e}")

def get_llm(config_path: Optional[str] = "model.config.json") -> BaseLLM:
    """
    Central entry point for the pipeline.
    1) Load defaults → overlay file config → overlay env vars.
    2) Instantiate backend client with a uniform interface.
    """
    cfg = {**DEFAULTS, **_read_file_config(config_path)}
    # Environment overrides (keep lower-case for backend)
    backend = os.getenv("LLM_BACKEND", str(cfg.get("backend", DEFAULTS["backend"]))).lower()
    model = os.getenv("LLM_MODEL", str(cfg.get("model", DEFAULTS["model"])))
    temperature = float(os.getenv("LLM_TEMPERATURE", cfg.get("temperature", DEFAULTS["temperature"])))
    model = _normalize_model_name(backend, model)

    if backend == "local_stub":
        return LocalStubLLM(model=model, temperature=temperature)
    if backend == "openai":
        return OpenAILLM(model=model, temperature=temperature)
    raise ValueError(f"Unsupported backend: {backend}")

# ---------- Minimal self-test (optional) ----------
if __name__ == "__main__":
    # Example usage. In production, your pipeline will import get_llm() instead.
    client = get_llm()  # reads model.config.json and/or env
    system = (
        "You are a helpful personal finance assistant. "
        "Answer with: concept → motivation → actionable steps. "
        "Do not mention internal sources."
    )
    user = "How can I start an emergency fund if my income is irregular?"
    print(client.complete(system, user))
