# Imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import re



# Загрузка книги
with open("finance_book_clean.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Делим на абзацы
paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
print(f"Total paragraphs: {len(paragraphs)}")




# Загружаем pre-trained эмбеддер
model = SentenceTransformer("all-MiniLM-L6-v2")

# Векторизация абзацев
embeddings = model.encode(paragraphs, convert_to_tensor=False)




# Сравниваем каждый абзац с соседним
similarities = [cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                for i in range(len(embeddings)-1)]

# Преобразуем в разницу (semantic drift)
semantic_drift = [1 - s for s in similarities]




plt.figure(figsize=(14, 4))
plt.plot(semantic_drift)
plt.axhline(0.3, color='red', linestyle='--', label="Potential topic break (threshold)")
plt.title("Semantic Drift Between Paragraphs")
plt.xlabel("Paragraph index")
plt.ylabel("1 - Cosine Similarity")
plt.legend()
plt.grid(True)
plt.show()




threshold = 0.3
chunks = []
current_chunk = [paragraphs[0]]

for i, drift in enumerate(semantic_drift):
    if drift > threshold:
        chunks.append(" ".join(current_chunk))
        current_chunk = []
    current_chunk.append(paragraphs[i+1])

# Не забываем последний чанк
if current_chunk:
    chunks.append(" ".join(current_chunk))

print(f"\nTotal semantic chunks: {len(chunks)}")





