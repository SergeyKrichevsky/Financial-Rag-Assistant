# file: llm_integration/run_logger.py
# Minimal run logger for Phase 4.
# - Writes the latest run to artifacts/v4/runs/last_run_phase4.json
# - Optionally appends every run to runs_history.jsonl (controlled via configs/rag_config.json)

from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List

from .config_loader import get_logging_cfg

# ---- Load logging config (with safe defaults) ----
_LOG = get_logging_cfg()
RUNS_DIR = _LOG.get("runs_dir", "artifacts/v4/runs")
LAST_RUN_FILE = os.path.join(RUNS_DIR, "last_run_phase4.json")

HISTORY_FILE_NAME = _LOG.get("history_file", "runs_history.jsonl")
ENABLE_HISTORY = bool(_LOG.get("enable_history", True))
HISTORY_PATH = os.path.join(RUNS_DIR, HISTORY_FILE_NAME)

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _shallow_refs(refs: List[Dict[str, Any]], max_len: int = 50) -> List[Dict[str, Any]]:
    """Keep only safe, short fields for logging."""
    out: List[Dict[str, Any]] = []
    for r in refs or []:
        out.append({
            "id": r.get("id"),
            "chapter": r.get("chapter"),
            "position": r.get("position"),
            "score": r.get("score"),
            "preview": (r.get("preview") or "")[:max_len]
        })
    return out

def append_history(record: Dict[str, Any]) -> None:
    """Append a single JSON line to runs_history.jsonl if enabled."""
    if not ENABLE_HISTORY:
        return
    _ensure_dir(RUNS_DIR)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def log_phase4_run(
    model_name: str,
    question: str,
    context_text: str,
    refs: List[Dict[str, Any]],
    answer_text: str,
    extra: Dict[str, Any] | None = None,
) -> str:
    """
    Write a compact JSON log for a single Phase 4 run.
    Returns the absolute path to the written 'last_run_phase4.json'.
    Also appends a history line to 'runs_history.jsonl' if enabled.
    """
    _ensure_dir(RUNS_DIR)
    payload: Dict[str, Any] = {
        "ts_unix": time.time(),
        "model": model_name,
        "question_preview": (question or "")[:200],
        "context_chars": len(context_text or ""),
        "num_refs": len(refs or []),
        "refs": _shallow_refs(refs or []),
        "answer_chars": len(answer_text or ""),
    }
    if extra:
        payload["extra"] = extra

    # Write last run (pretty JSON)
    with open(LAST_RUN_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Append to history (one-line JSON)
    try:
        append_history(payload)
    except Exception:
        # Never fail the main flow because of history
        pass

    return os.path.abspath(LAST_RUN_FILE)
