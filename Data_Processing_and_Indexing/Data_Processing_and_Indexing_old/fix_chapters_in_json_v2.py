# fix_chapters_in_json_v2.py
# Purpose: Fix chapter labels more robustly.
# Rules:
# 1) If a chunk text contains an explicit "Chapter N: ..." heading, trust it.
# 2) Else, locate the chunk in the original book with safer anchoring (no lossy normalization),
#    then map its start to chapter spans.

import os
import re
import json
from typing import Optional, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
BOOK_PATH = os.path.join(ROOT, "finance_book_clean.txt")
SRC_JSON = os.path.join(ROOT, "finance_book_chunks_enriched.json")
DST_JSON = os.path.join(ROOT, "finance_book_chunks_enriched.fixed.v2.json")

# --- Load data ---
with open(BOOK_PATH, "r", encoding="utf-8") as f:
    book = f.read()

with open(SRC_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# --- Build chapter spans from the book ---
chap_pat = re.compile(r"(^|\n)(Chapter\s+\d+\s*:[^\n]*|Chapter\s+\d+[^\n]*)", re.IGNORECASE)
hits = []
for m in chap_pat.finditer(book):
    start = m.start(2) if m.group(2) else m.start()
    title = m.group(2).strip()
    hits.append((start, title))

spans = []
if hits:
    if hits[0][0] > 0:
        spans.append((0, hits[0][0], "Introduction"))
    for i, (start, title) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(book)
        spans.append((start, end, title))
else:
    spans.append((0, len(book), "Whole Book"))

def chapter_for_pos(pos: int) -> str:
    """Map a character offset to its chapter title by spans."""
    for s, e, title in spans:
        if s <= pos < e:
            return title
    return spans[-1][2]

# --- Helpers ---

heading_in_chunk = re.compile(r"(Chapter\s+\d+\s*:[^\n]*|Chapter\s+\d+[^\n]*)", re.IGNORECASE)

def extract_heading_from_chunk(text: str) -> Optional[str]:
    """If the chunk itself contains a 'Chapter N: ...' heading, return it."""
    m = heading_in_chunk.search(text)
    return m.group(1).strip() if m else None

def locate_chunk_start(haystack: str, chunk_text: str, cursor: int) -> int:
    """
    Try to find chunk_text in haystack reliably:
    1) Raw .find from a rolling cursor (fast/precise).
    2) If not found, use a compact 'anchor' = first 180 chars (or len//2) to reduce false positives.
    Returns start index or -1.
    """
    # Try exact match first
    idx = haystack.find(chunk_text, cursor)
    if idx != -1:
        return idx

    # Build a shorter, still-distinctive anchor (avoid normalizing away signals)
    head = chunk_text.strip()[:180]
    # If head is too short (rare), extend up to half of the chunk
    if len(head) < 80:
        head = chunk_text.strip()[: max(80, len(chunk_text) // 2)]

    # Try to find the anchor near the cursor forward
    idx = haystack.find(head, cursor)
    if idx != -1:
        return idx

    # As a final attempt, allow a broader scan from the beginning to handle cursor drift
    idx = haystack.find(head)
    return idx

# --- Fix process ---

cursor = 0
fixed = 0
samples = []

for d in chunks:
    cid = d["id"]
    text = d["text"]
    stored = d.get("chapter")

    # 1) If the chunk carries its own heading, trust it directly
    own = extract_heading_from_chunk(text)
    if own:
        real = own
        # Keep cursor strategy: if we can also locate, advance cursor for next chunks
        pos = locate_chunk_start(book, text, cursor)
        if pos != -1:
            cursor = max(cursor, pos + max(1, len(text) // 2))
    else:
        # 2) Otherwise, locate in the book and map to spans
        pos = locate_chunk_start(book, text, cursor)
        if pos == -1:
            # Could not locate; keep existing label
            real = stored
        else:
            real = chapter_for_pos(pos)
            cursor = max(cursor, pos + max(1, len(text) // 2))

    if real != stored:
        d["chapter"] = real
        fixed += 1
        if len(samples) < 10:
            samples.append((cid, stored, real))

# --- Save result ---
with open(DST_JSON, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"Total chunks: {len(chunks)}")
print(f"Chapter spans: {len(spans)}")
print(f"Updated chapter labels: {fixed}")
print(f"Saved: {DST_JSON}\n")
print("Examples (up to 10):")
for i, (cid, old, new) in enumerate(samples, start=1):
    print(f"{i:02d}. {cid}: '{old}' -> '{new}'")
