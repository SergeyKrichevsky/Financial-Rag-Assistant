# test_chroma_with_neighbors.py
# Purpose: query Chroma with the same embedding model and expand the top hit with ±1 neighbors by chunk order.

from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import re

# --- Load persistent DB and embedding model ---
client = PersistentClient(path="./chroma_store", settings=Settings(anonymized_telemetry=False))
collection = client.get_collection(name="finance_book")
embedder = SentenceTransformer("thenlper/gte-large")

def id_to_order(chunk_id: str) -> int:
    """Extract numeric order from ids like 'chunk_12'."""
    m = re.match(r"chunk_(\d+)$", chunk_id)
    return int(m.group(1)) if m else -1

def order_to_id(order: int) -> str:
    """Reconstruct id string from numeric order."""
    return f"chunk_{order}"

# --- Make a query embedding with the SAME model used for indexing ---
query_text = "How can I quickly stop money leaks without making a full budget?"
query_emb = embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)

# --- Get top-1 semantic result ---
top = collection.query(query_embeddings=query_emb, n_results=1)

top_id = top["ids"][0][0]
top_meta = top["metadatas"][0][0]
top_text = top["documents"][0][0]
top_order = top_meta.get("order", id_to_order(top_id))

# --- Build neighbor ids (±1) and fetch them in one call ---
neighbor_orders = [o for o in [top_order - 1, top_order, top_order + 1] if o >= 1]
neighbor_ids = [order_to_id(o) for o in neighbor_orders]

neighbors = collection.get(ids=neighbor_ids)

# --- Pretty print results in reading order ---
print("🔍 Query:", query_text)
print("=" * 80)
# Sort by order to keep context flow
items = list(zip(neighbors["ids"], neighbors["documents"], neighbors["metadatas"]))
items.sort(key=lambda x: x[2].get("order", id_to_order(x[0])))

for i, (cid, ctext, cmeta) in enumerate(items, start=1):
    print(f"[{i}/{len(items)}] ID: {cid} | Chapter: {cmeta.get('chapter','N/A')} | Tokens: {cmeta.get('tokens','N/A')} | Order: {cmeta.get('order','?')}")
    print("Text snippet:", ctext[:300].replace("\n", " "), "...\n")
