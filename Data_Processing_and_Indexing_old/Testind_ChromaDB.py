from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Load persistent ChromaDB
client = PersistentClient(path="./chroma_store", settings=Settings(anonymized_telemetry=False))
collection = client.get_collection(name="finance_book")

# Load the same embedding model we used during indexing
embedder = SentenceTransformer("thenlper/gte-large")

# Test query
query_text = "How can I quickly stop money leaks without making a full budget?"

# Convert query text to embedding
query_emb = embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)

# Run semantic search with precomputed embeddings
results = collection.query(
    query_embeddings=query_emb,
    n_results=3
)

# Print results
print("🔍 Query:", query_text)
print("=" * 80)
for i in range(len(results["ids"][0])):
    print(f"[Rank {i+1}] ID: {results['ids'][0][i]}")
    print(f"Tokens: {results['metadatas'][0][i].get('tokens', 'N/A')}")
    print(f"Chapter: {results['metadatas'][0][i].get('chapter', 'N/A')}")
    print("Text snippet:", results["documents"][0][i][:250], "...\n")
