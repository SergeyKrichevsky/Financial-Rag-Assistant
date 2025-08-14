# file: llm_integration/run_logger.py
# Minimal run logger for Phase 4. Writes JSON to artifacts/v4/runs/last_run_phase4.json

from __future__ import annotations
import json, os, time
from typing import Any, Dict, List

RUNS_DIR = os.path.join("artifacts", "v4", "runs")
RUN_FILE = os.path.join(RUNS_DIR, "last_run_phase4.json")

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _shallow_refs(refs: List[Dict[str, Any]], max_len: int = 50) -> List[Dict[str, Any]]:
    """Keep only safe, short fields for logging."""
    out: List[Dict[str, Any]] = []
    for r in refs:
        out.append({
            "id": r.get("id"),
            "chapter": r.get("chapter"),
            "position": r.get("position"),
            "score": r.get("score"),
            "preview": (r.get("preview") or "")[:max_len]
        })
    return out

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
    Returns the absolute path to the written file.
    """
    _ensure_dir(RUNS_DIR)
    payload: Dict[str, Any] = {
        "ts_unix": time.time(),
        "model": model_name,
        "question": question,
        "context_chars": len(context_text or ""),
        "num_refs": len(refs or []),
        "refs": _shallow_refs(refs or []),
        "answer_chars": len(answer_text or ""),
    }
    if extra:
        payload["extra"] = extra

    with open(RUN_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return os.path.abspath(RUN_FILE)
