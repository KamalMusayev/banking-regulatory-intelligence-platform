from backend.reguaz.retrieval.hybrid_qdrant import HybridQdrantRetriever

retriever = HybridQdrantRetriever(
    model_name="bge_m3",
    qdrant_dir="data/qdrant",
    chunks_dir="data/processed/chunks",
)

results = retriever.retrieve(
    "Banklarda kapital adekvatlığı nədir?",
    top_k=5,
)

print(results)