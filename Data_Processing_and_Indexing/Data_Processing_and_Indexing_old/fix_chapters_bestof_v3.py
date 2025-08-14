# Data_Processing_and_Indexing/fix_chapters_bestof_v3.py
"""
Best-of merge for chapter labels across two JSONs.

- Base:  finance_book_chunks_enriched.fixed.v2.json
- Patch: finance_book_chunks_enriched.fixed.json     (only 'chapter' field for selected chunks)
- Output: finance_book_chunks_enriched.fixed.v3.json

What it does:
1) Loads both JSON arrays (each item has fields like: id, text, chapter, tokens, order, etc.).
2) For specific chunk ids, copies the 'chapter' value from v1 into the v2 item.
3) Keeps improved labels from v2 for several chunk ids.
4) Marks chunk_19 as a cross-chapter span: "Chapter 8 → Chapter 9 (span)".
5) Writes a new JSON and prints a short diff-style report.

It does NOT touch Chroma. Pure JSON rewrite.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# ---- Config ----
DATA_DIR = Path(".")  # adjust if you run from a different cwd

INPUT_V1 = DATA_DIR / "finance_book_chunks_enriched.fixed.json"
INPUT_V2 = DATA_DIR / "finance_book_chunks_enriched.fixed.v2.json"
OUTPUT_V3 = DATA_DIR / "finance_book_chunks_enriched.fixed.v3.json"

# Chunks whose 'chapter' label we prefer from v1
TAKE_FROM_V1 = {2, 5, 6, 9, 10, 12, 14, 15, 16, 17, 20}

# Chunks that improved in v2 (keep as-is explicitly)
KEEP_FROM_V2 = {8, 11, 18, 21}

# Optional: mark cross-chapter spans here (id -> label)
SPAN_OVERRIDES = {
    19: "Chapter 8 → Chapter 9 (span)"
}


@dataclass
class Chunk:
    raw: dict

    @property
    def id_num(self) -> int:
        # expects id like "chunk_12"
        try:
            return int(str(self.raw.get("id", "")).split("_")[1])
        except Exception:
            return -1

    @property
    def chapter(self) -> str:
        return self.raw.get("chapter", "")

    @chapter.setter
    def chapter(self, value: str) -> None:
        self.raw["chapter"] = value


def load_index(path: Path) -> Dict[int, Chunk]:
    items: List[dict] = json.loads(path.read_text(encoding="utf-8"))
    index: Dict[int, Chunk] = {}
    for obj in items:
        ch = Chunk(obj)
        index[ch.id_num] = ch
    return index


def extract_chapter_number(label: str) -> int:
    """
    Returns chapter number if present (e.g., 'Chapter 6: ...' -> 6),
    else 0 for intro/front matter/etc.
    """
    m = re.search(r"Chapter\s+(\d+)", label or "", flags=re.I)
    return int(m.group(1)) if m else 0


def main() -> None:
    # Load
    idx_v1 = load_index(INPUT_V1)
    idx_v2 = load_index(INPUT_V2)

    # Sanity check: ensure same id set
    ids_v1 = set(idx_v1.keys())
    ids_v2 = set(idx_v2.keys())
    if ids_v1 != ids_v2:
        missing_in_v1 = sorted(ids_v2 - ids_v1)
        missing_in_v2 = sorted(ids_v1 - ids_v2)
        print("[WARN] ID sets differ.")
        if missing_in_v1:
            print("  Missing in v1:", missing_in_v1)
        if missing_in_v2:
            print("  Missing in v2:", missing_in_v2)

    # Build final list following v2 order
    ordered_ids = sorted(ids_v2)
    final_chunks: List[dict] = []
    changes: List[Tuple[int, str, str]] = []

    for cid in ordered_ids:
        base = idx_v2[cid]
        original = base.chapter

        # Decide which label to use
        if cid in TAKE_FROM_V1 and cid in idx_v1:
            new_label = idx_v1[cid].chapter
        elif cid in SPAN_OVERRIDES:
            new_label = SPAN_OVERRIDES[cid]
        else:
            new_label = base.chapter  # keep from v2 (including KEEP_FROM_V2)

        # Apply if different
        if new_label != original:
            changes.append((cid, original, new_label))
            base.chapter = new_label

        final_chunks.append(base.raw)

    # Simple quality check: detect big backward jumps in chapter numbering
    jumps = []
    prev_num = extract_chapter_number(final_chunks[0].get("chapter", ""))
    for i, obj in enumerate(final_chunks[1:], start=1):
        cur_num = extract_chapter_number(obj.get("chapter", ""))
        label = (obj.get("chapter") or "").lower()
        is_span = "span" in label
        if (cur_num < prev_num) and not is_span:
            jumps.append((final_chunks[i - 1]["id"], final_chunks[i]["id"], prev_num, cur_num))
        prev_num = cur_num if not is_span else prev_num  # keep prev on span

    # Write output
    OUTPUT_V3.write_text(json.dumps(final_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    # Report
    print(f"Saved -> {OUTPUT_V3}")
    print(f"Total chunks: {len(final_chunks)}")
    print(f"Chapter updates applied: {len(changes)}")
    if changes:
        print("- Examples (up to 10):")
        for cid, old, new in changes[:10]:
            print(f"  {cid:02d}. chunk_{cid}: '{old}' -> '{new}'")

    if jumps:
        print("\n[NOTE] Backward chapter jumps detected (could be fine if front matter or span):")
        for prev_id, cur_id, prev_num, cur_num in jumps:
            print(f"  {prev_id} -> {cur_id}: {prev_num} → {cur_num}")


if __name__ == "__main__":
    main()
