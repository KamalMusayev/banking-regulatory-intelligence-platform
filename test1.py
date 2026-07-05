import time
from sentence_transformers import CrossEncoder

# Modeli GPU-da yüklə
model = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    device="cuda"
)

pairs = [["hello", "world"]] * 30

start = time.time()

scores = model.predict(
    pairs,
    batch_size=32,
    show_progress_bar=True,
)

print(f"\nTime: {time.time() - start:.2f} sec")
print(scores[:3])