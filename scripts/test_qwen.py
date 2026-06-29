import time

from sentence_transformers import SentenceTransformer, util

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

print("=" * 80)
print(f"Loading model: {MODEL_NAME}")
print("=" * 80)

start = time.perf_counter()

model = SentenceTransformer(MODEL_NAME)

elapsed = time.perf_counter() - start

print(f"\n Model loaded successfully.")
print(f"Loading time: {elapsed:.2f} seconds")

print("\n" + "=" * 80)
print("Single embedding test")
print("=" * 80)

text = "Azərbaycan Respublikasının Mərkəzi Bankı"

embedding = model.encode(
    text,
    normalize_embeddings=True
)

print(f"Embedding type      : {type(embedding)}")
print(f"Embedding dimension : {len(embedding)}")

print("\n" + "=" * 80)
print("Batch embedding test")
print("=" * 80)

texts = [
    "Azərbaycan bank sistemi",
    "Central Bank of Azerbaijan",
    "Банковская система Азербайджана",
    "Commercial banks"
]

batch_embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

print(f"Batch size          : {len(batch_embeddings)}")
print(f"Embedding dimension : {len(batch_embeddings[0])}")

print("\n" + "=" * 80)
print("Similarity test")
print("=" * 80)

query = model.encode(
    "Bank haqqında qanun",
    normalize_embeddings=True
)

document = model.encode(
    "Azərbaycan Respublikasının Banklar haqqında Qanunu",
    normalize_embeddings=True
)

score = util.cos_sim(query, document)

print(f"Similarity score: {score.item():.4f}")

print("\n" + "=" * 80)
print("Multilingual retrieval test")
print("=" * 80)

docs = [
    "Azərbaycan Respublikasının Mərkəzi Bankı pul siyasətini həyata keçirir.",
    "The Central Bank regulates monetary policy.",
    "Центральный банк регулирует денежно-кредитную политику."
]

doc_embeddings = model.encode(
    docs,
    normalize_embeddings=True
)

queries = [
    "Azərbaycan Mərkəzi Bankı",
    "Central Bank",
    "Центральный банк"
]

for q in queries:

    q_embedding = model.encode(
        q,
        normalize_embeddings=True
    )

    scores = util.cos_sim(q_embedding, doc_embeddings)[0]

    best_index = scores.argmax().item()

    print("\nQuery:", q)
    print("Best match:", docs[best_index])
    print("Score:", round(scores[best_index].item(), 4))

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED SUCCESSFULLY")
print("=" * 80)