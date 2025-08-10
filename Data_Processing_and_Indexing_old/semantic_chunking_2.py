# Step 1: Enrich chunks with metadata (chapter, order, char_span)
import re
import json

# Load raw book and chunks
with open("finance_book_clean.txt", "r", encoding="utf-8") as f:
    book = f.read()
with open("finance_book_chunks_300_500.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Pre-compute chapter spans
chapter_regex = re.compile(r"(Chapter\s+\d+:[^\n]+)", re.IGNORECASE)
chapters = []
for m in chapter_regex.finditer(book):
    chapters.append({"title": m.group(1).strip(), "start": m.start()})
for i in range(len(chapters)):
    chapters[i]["end"] = chapters[i+1]["start"] if i+1 < len(chapters) else len(book)

def find_chapter_for_span(start_idx):
    for ch in chapters:
        if ch["start"] <= start_idx < ch["end"]:
            return ch["title"]
    return "Front matter / Introduction"

# Map chunk text back to book to get span and chapter (best-effort)
cursor = 0
enriched = []
for order, ch in enumerate(chunks, start=1):
    # naive search from cursor forward to avoid matching earlier duplicates
    idx = book.find(ch["text"][:120], cursor)  # anchor by prefix
    if idx == -1:
        idx = book.find(ch["text"][:60], 0)    # fallback broader search
    chapter_title = find_chapter_for_span(max(idx, 0))
    enriched.append({
        "id": ch["id"],
        "text": ch["text"],
        "tokens": ch["tokens"],
        "order": order,
        "chapter": chapter_title,
        "char_start": idx if idx != -1 else None,
        "char_end": (idx + len(ch["text"])) if idx != -1 else None
    })
    if idx != -1:
        cursor = idx + len(ch["text"])  # advance cursor if found

with open("finance_book_chunks_enriched.json", "w", encoding="utf-8") as f:
    json.dump(enriched, f, ensure_ascii=False, indent=2)

print("✅ Saved: finance_book_chunks_enriched.json")
