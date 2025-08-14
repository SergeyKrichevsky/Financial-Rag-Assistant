# fix_chapters_in_json.py
# Purpose: Recompute correct chapter labels for each chunk and write a fixed JSON copy.
# This DOES NOT touch Chroma; it only produces a corrected JSON file.

import os
import re
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
BOOK_PATH = os.path.join(ROOT, "finance_book_clean.txt")
SRC_JSON = os.path.join(ROOT, "finance_book_chunks_enriched.json")
DST_JSON = os.path.join(ROOT, "finance_book_chunks_enriched.fixed.json")

# --- Load data ---
with open(BOOK_PATH, "r", encoding="utf-8") as f:
    book = f.read()

with open(SRC_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# --- Detect chapter spans from the book ---
chap_pat = re.compile(r"(^|\n)(Chapter\s+\d+\s*:[^\n]*|Chapter\s+\d+[^\n]*)", re.IGNORECASE)
hits = []
for m in chap_pat.finditer(book):
    start = m.start(2) if m.group(2) else m.start()
    title = m.group(2).strip()
    hits.append((start, title))

spans = []
if hits:
    # text before first chapter -> label as "Introduction"
    if hits[0][0] > 0:
        spans.append((0, hits[0][0], "Introduction"))
    for i, (start, title) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(book)
        spans.append((start, end, title))
else:
    spans.append((0, len(book), "Whole Book"))

def chapter_for_pos(pos: int) -> str:
    for s, e, title in spans:
        if s <= pos < e:
            return title
    return spans[-1][2]

# --- Locate each chunk in the book and recompute its chapter ---
def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

cursor = 0
fixed = 0
mismatches_samples = []

for d in chunks:
    cid = d["id"]
    text = d["text"]
    stored = d.get("chapter", None)

    # Fast raw search from rolling cursor
    idx = book.find(text, cursor)
    if idx == -1:
        # Fallback: normalized search in a window
        window = book[cursor:cursor + max(200000, len(text) * 5)]
        n_text = norm_space(text)
        n_win = norm_space(window)
        pos = n_win.find(n_text)
        if pos == -1:
            # Give up; keep stored chapter
            continue
        idx = cursor + pos

    real = chapter_for_pos(idx)
    if real != stored:
        mismatches_samples.append((cid, stored, real))
        d["chapter"] = real
        fixed += 1

    # advance cursor roughly forward
    cursor = max(cursor, idx + max(1, len(text) // 2))

# --- Save fixed copy ---
with open(DST_JSON, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"Total chunks: {len(chunks)}")
print(f"Chapter spans: {len(spans)}")
print(f"Updated chapter labels: {fixed}")
print(f"Saved: {DST_JSON}")

print("\nExamples (up to 10):")
for i, (cid, old, new) in enumerate(mismatches_samples[:10], start=1):
    print(f"{i:02d}. {cid}: '{old}' -> '{new}'")
