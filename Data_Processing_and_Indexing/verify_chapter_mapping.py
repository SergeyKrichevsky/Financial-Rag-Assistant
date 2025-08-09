# verify_chapter_mapping.py
# Goal: Verify (dry-run) that each chunk's chapter label matches its real chapter in the original book text.
# Strategy:
# 1) Parse chapter headings and build spans (start_char, end_char, title) from finance_book_clean.txt.
# 2) For each chunk (in order), find its first occurrence in the book starting from a rolling cursor.
# 3) Map the chunk's start_char to the chapter span and compare with stored metadata.
# NOTE: This script does not modify JSON or Chroma – it only reports mismatches.

import os
import re
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
BOOK_PATH = os.path.join(ROOT, "finance_book_clean.txt")
JSON_PATH = os.path.join(ROOT, "finance_book_chunks_enriched.json")

# --- Load data ---
with open(BOOK_PATH, "r", encoding="utf-8") as f:
    book = f.read()

with open(JSON_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# --- 1) Detect chapter headings and build spans ---
# Accept headings like: "Chapter 2: The 10-Minute Money Checkup — ...", "Chapter 3:", etc.
chap_pat = re.compile(r"(^|\n)(Chapter\s+\d+\s*:[^\n]*|Chapter\s+\d+[^\n]*)", re.IGNORECASE)

chapters = []
for m in chap_pat.finditer(book):
    start = m.start(2) if m.group(2) else m.start()
    title = m.group(2).strip()
    chapters.append((start, title))

# Add a synthetic "Prelude/Front" span for text before the first matched chapter
spans = []
if chapters:
    # prelude
    if chapters[0][0] > 0:
        spans.append((0, chapters[0][0], "Introduction"))  # you can rename this label if needed
    # chapters with ends
    for i, (start, title) in enumerate(chapters):
        end = chapters[i + 1][0] if i + 1 < len(chapters) else len(book)
        spans.append((start, end, title))
else:
    # Fallback: no chapters detected
    spans.append((0, len(book), "Whole Book"))

# --- 2) For each chunk, locate its position in the book (rolling search to avoid collisions) ---
def find_rolling(haystack: str, needle: str, start_pos: int) -> int:
    """Find 'needle' in 'haystack' at or after 'start_pos'. Return index or -1."""
    # Tolerate minor whitespace differences by collapsing runs of whitespace
    norm = lambda s: re.sub(r"\s+", " ", s.strip())
    n_needle = norm(needle)
    # We scan a window of the haystack to avoid normalizing entire book repeatedly
    # Simple approach: search raw first; if not found, try normalized search.
    idx_raw = haystack.find(needle, start_pos)
    if idx_raw != -1:
        return idx_raw

    # Normalized search fallback
    # Build a limited slice to keep it efficient
    window = haystack[start_pos:start_pos + max(200000, len(needle) * 5)]
    n_window = norm(window)
    pos = n_window.find(n_needle)
    if pos == -1:
        return -1
    # Map back to original index approximately (best-effort)
    # We re-scan a small window to refine the raw index
    approx_global = start_pos + pos
    return approx_global

def chapter_for_pos(pos: int) -> str:
    for s, e, title in spans:
        if s <= pos < e:
            return title
    return spans[-1][2]

cursor = 0
mismatches = []
unlocated = []

for d in chunks:
    cid = d["id"]
    text = d["text"]
    stored_ch = d.get("chapter")
    found_at = find_rolling(book, text, cursor)
    if found_at == -1:
        unlocated.append(cid)
        continue
    real_ch = chapter_for_pos(found_at)
    if stored_ch != real_ch:
        mismatches.append((cid, stored_ch, real_ch))
    # advance cursor a bit before the end of this match to keep order
    cursor = max(cursor, found_at + max(1, len(text) // 2))

# --- 3) Report ---
print(f"Total chunks: {len(chunks)}")
print(f"Chapter spans detected: {len(spans)}")
print(f"Mismatches: {len(mismatches)} | Unlocated: {len(unlocated)}")
print("-" * 80)

print("Examples of mismatches (up to 10):")
for i, (cid, old, new) in enumerate(mismatches[:10], start=1):
    print(f"{i:02d}. {cid}: stored='{old}' -> real='{new}'")

if unlocated:
    print("\nUnlocated chunks (up to 10):")
    for cid in unlocated[:10]:
        print(" -", cid)

# Exit hints
if mismatches:
    print("\nHint: We can fix JSON and update Chroma metadatas in-place in a follow-up step.")
else:
    print("\nNo mismatches found. Metadata looks consistent.")
