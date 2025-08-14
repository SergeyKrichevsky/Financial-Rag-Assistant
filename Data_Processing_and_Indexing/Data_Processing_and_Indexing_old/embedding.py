# Step 2: Build embeddings + Chroma vector store
import json
from sentence_transformers import SentenceTransformer
import numpy as np
# from chromadb import Client
from chromadb import PersistentClient
from chromadb.config import Settings

# # Use a valid HF model id (pick one)
# EMB_MODEL_NAME = "thenlper/gte-large"        # strong general-purpose embeddings
# # EMB_MODEL_NAME = "BAAI/bge-large-en-v1.5"  # also strong; needs normalize at encode time

# embedder = SentenceTransformer(EMB_MODEL_NAME)
# print("✅ Loaded:", EMB_MODEL_NAME)


# Load enriched chunks
with open("finance_book_chunks_enriched.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Choose embedding model (offline)
EMB_MODEL_NAME = "thenlper/gte-large"        # strong general-purpose embeddings
# EMB_MODEL_NAME = "gte-large"  # or "BAAI/bge-large-en-v1.5"
embedder = SentenceTransformer(EMB_MODEL_NAME)

# Create Chroma client (local persistence)
client = PersistentClient(path="./chroma_store", settings=Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name="finance_book")


# Compute and upsert embeddings
texts = [d["text"] for d in data]
ids = [d["id"] for d in data]
metadatas = [{"chapter": d["chapter"], "order": d["order"], "tokens": d["tokens"]} for d in data]

embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=embs.tolist(),
    metadatas=metadatas
)

print("✅ Chroma index built and persisted at ./chroma_store")




