# file: llm_integration/config_loader.py
# Simple JSON config loader for Phase 4 (RAG parameters).
# - Reads configs/rag_config.json
# - Merges user values over safe defaults
# - Exposes helpers to access "retriever" and "generator" sections

from __future__ import annotations
import json
import os
from typing import Any, Dict

_DEFAULTS: Dict[str, Any] = {
    "retriever": {
        "k_default": 5,
        "max_per_chapter": 2,
        "exclude_chapters": [
            "About the Author",
            "Final Words",
            "Acknowledgments",
            "Table of Contents",
            "Foreword",
            "Preface",
            "Index"
        ]
    },
    "generator": {
        "system_prompt_path": "configs/system_prompt.txt",
        "max_context_chars": 6000,
        "max_context_tokens": 900,
        "token_encoding": "cl100k_base"
    }
}

def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge src into dst (dicts only)."""
    out = dict(dst)
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_rag_config(path: str = "configs/rag_config.json") -> Dict[str, Any]:
    """Load JSON config and merge over defaults. Missing/invalid file → defaults."""
    cfg = dict(_DEFAULTS)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            if isinstance(user_cfg, dict):
                cfg = _deep_merge(cfg, user_cfg)
    except Exception:
        # Fail-safe: return defaults if file is malformed
        cfg = dict(_DEFAULTS)
    return cfg

def get_retriever_cfg(path: str = "configs/rag_config.json") -> Dict[str, Any]:
    """Return the 'retriever' section (with defaults)."""
    return load_rag_config(path).get("retriever", dict(_DEFAULTS["retriever"]))

def get_generator_cfg(path: str = "configs/rag_config.json") -> Dict[str, Any]:
    """Return the 'generator' section (with defaults)."""
    return load_rag_config(path).get("generator", dict(_DEFAULTS["generator"]))



def get_logging_cfg(path: str = "configs/rag_config.json") -> Dict[str, Any]:
    """Return the 'logging' section (with defaults if missing)."""
    cfg = load_rag_config(path)
    return cfg.get("logging", {
        "runs_dir": "artifacts/v4/runs",
        "history_file": "runs_history.jsonl",
        "enable_history": True
    })
