import re
import json
import tiktoken
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ===== PARAMETERS =====
TARGET_MIN_TOKENS = 300       # Minimum tokens per chunk
TARGET_MAX_TOKENS = 500       # Maximum tokens per chunk
THRESHOLD = 0.3               # Semantic drift threshold for initial split
MODEL_NAME = "gpt-4o"         # Tokenization model (GPT-4.0 compatible)

# ===== LOAD CLEANED BOOK =====
with open("finance_book_clean.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# ===== STEP 1 — Split into paragraphs =====
paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]

# ===== STEP 2 — Encode paragraphs using SentenceTransformer =====
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(paragraphs, convert_to_tensor=False)

# ===== STEP 3 — Calculate semantic drift between adjacent paragraphs =====
similarities = [
    cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
    for i in range(len(embeddings) - 1)
]
semantic_drift = [1 - s for s in similarities]

# ===== STEP 4 — Initial semantic chunking =====
chunks = []
current_chunk = [paragraphs[0]]

for i, drift in enumerate(semantic_drift):
    if drift > THRESHOLD:
        chunks.append(" ".join(current_chunk))
        current_chunk = []
    current_chunk.append(paragraphs[i + 1])

if current_chunk:
    chunks.append(" ".join(current_chunk))

# ===== STEP 5 — Tokenization function =====
enc = tiktoken.encoding_for_model(MODEL_NAME)

def count_tokens(text):
    """Count the number of tokens in a given text."""
    return len(enc.encode(text))

# ===== STEP 6 — Merge small chunks / Split large chunks =====
final_chunks = []
buffer = ""

for ch in chunks:
    tokens = count_tokens(ch)

    if tokens < TARGET_MIN_TOKENS:
        # Merge with buffer if available
        if buffer:
            merged = buffer + " " + ch
            if count_tokens(merged) <= TARGET_MAX_TOKENS:
                buffer = merged
                continue
            else:
                final_chunks.append(buffer.strip())
                buffer = ch
        else:
            buffer = ch

    elif tokens > TARGET_MAX_TOKENS:
        # Split large chunk into smaller ones by sentences
        sentences = re.split(r'(?<=[.!?]) +', ch)
        sub_buffer = ""
        for sent in sentences:
            if count_tokens(sub_buffer + " " + sent) > TARGET_MAX_TOKENS:
                final_chunks.append(sub_buffer.strip())
                sub_buffer = sent
            else:
                sub_buffer += " " + sent
        if sub_buffer.strip():
            final_chunks.append(sub_buffer.strip())

    else:
        # Chunk is within desired range
        if buffer:
            merged = buffer + " " + ch
            if count_tokens(merged) <= TARGET_MAX_TOKENS:
                final_chunks.append(merged.strip())
                buffer = ""
            else:
                final_chunks.append(buffer.strip())
                final_chunks.append(ch.strip())
                buffer = ""
        else:
            final_chunks.append(ch.strip())

# Add remaining buffer if exists
if buffer:
    final_chunks.append(buffer.strip())

# ===== STEP 7 — Save final chunks with metadata =====
output_data = []
for idx, text in enumerate(final_chunks, start=1):
    output_data.append({
        "id": f"chunk_{idx}",
        "text": text,
        "tokens": count_tokens(text)
    })

with open("finance_book_chunks_300_500.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"✅ Total chunks: {len(final_chunks)}")
print(f"📊 Average tokens per chunk: {sum(c['tokens'] for c in output_data) / len(output_data):.1f}")
print(f"📁 File saved: finance_book_chunks_300_500.json")
