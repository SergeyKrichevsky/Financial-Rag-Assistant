import re
import tiktoken
import statistics
import matplotlib.pyplot as plt
import json

# ===== Step 1 — Load text =====
with open("finance_book_clean.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ===== Step 2 — Init GPT-4o tokenizer =====
MODEL_NAME = "gpt-4o"
enc = tiktoken.encoding_for_model(MODEL_NAME)

# ===== Step 3 — Split into chapters (parent chunks) =====
# Assume chapters are marked by "Chapter", "CHAPTER", or numbered headings
chapter_pattern = r"(?:^|\n)(Chapter\s+\d+.*|CHAPTER\s+\d+.*|\d+\.\s+[^\n]+)"
chapters = re.split(chapter_pattern, text)

# Re-split preserves both headings and text, so we need to pair them
structured_book = []
current_chapter = None

for part in chapters:
    if not part.strip():
        continue
    if re.match(r"(Chapter|CHAPTER|\d+\.)", part.strip()):
        # New chapter starts
        current_chapter = {
            "chapter_title": part.strip(),
            "subsections": []
        }
        structured_book.append(current_chapter)
    else:
        # Add text to last chapter
        if current_chapter:
            current_chapter["subsections"].append(part.strip())

# ===== Step 4 — Split each chapter into subsections (child chunks) =====
# We use headings, bullet points, or empty lines as subsection markers
for chapter in structured_book:
    merged_subsections = []
    buffer = ""
    for para in re.split(r"\n\s*\n", "\n".join(chapter["subsections"])):
        para = para.strip()
        if not para:
            continue
        # Simple heuristic: if it's a heading-like line, start a new subsection
        if re.match(r"^#+\s|\d+\)|\d+\.", para) or len(buffer) > 500:
            if buffer:
                merged_subsections.append(buffer.strip())
                buffer = ""
        buffer += " " + para
    if buffer:
        merged_subsections.append(buffer.strip())
    chapter["subsections"] = merged_subsections

# ===== Step 5 — Token count =====
token_stats = []
for chapter in structured_book:
    for idx, sub in enumerate(chapter["subsections"], start=1):
        tokens = len(enc.encode(sub))
        token_stats.append(tokens)
        print(f"[{chapter['chapter_title']} → Section {idx}] Tokens: {tokens}")

# ===== Step 6 — Stats summary =====
print("\n=== Token Statistics ===")
print(f"Total sections: {len(token_stats)}")
print(f"Average tokens: {statistics.mean(token_stats):.1f}")
print(f"Median tokens: {statistics.median(token_stats):.1f}")
print(f"Min tokens: {min(token_stats)}")
print(f"Max tokens: {max(token_stats)}")

# ===== Step 7 — Plot distribution =====
plt.hist(token_stats, bins=20)
plt.xlabel("Tokens per semantic block")
plt.ylabel("Number of blocks")
plt.title(f"Semantic block length distribution ({MODEL_NAME})")
plt.show()

# ===== Step 8 — Save structure to JSON =====
with open("semantic_structure.json", "w", encoding="utf-8") as f:
    json.dump(structured_book, f, ensure_ascii=False, indent=2)
