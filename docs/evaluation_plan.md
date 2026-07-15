# Evaluation Plan — ReguAZ

> Retrieval and generation quality evaluation for the ReguAZ regulatory intelligence platform.

---

## Table of Contents

1. [Why Evaluation Matters](#1-why-evaluation-matters)
2. [Evaluation Architecture](#2-evaluation-architecture)
3. [Gold Dataset](#3-gold-dataset)
4. [Retrieval Evaluation Metrics](#4-retrieval-evaluation-metrics)
5. [Retrieval Evaluation Methodology](#5-retrieval-evaluation-methodology)
6. [Embedding Model Comparison](#6-embedding-model-comparison)
7. [Reranking Evaluation](#7-reranking-evaluation)
8. [Generation Evaluation Metrics](#8-generation-evaluation-metrics)
9. [Generation Evaluation Methodology](#9-generation-evaluation-methodology)
10. [Running Evaluations](#10-running-evaluations)

---

## 1. Why Evaluation Matters

A RAG system that retrieves irrelevant passages will hallucinate or give incorrect answers regardless of how capable the LLM is. In a regulatory compliance context, incorrect answers are not just unhelpful — they are potentially harmful. A compliance officer who receives a wrong answer about a capital adequacy threshold or an AML deadline may act on it.

For this reason, ReguAZ treats retrieval quality as a first-class engineering metric, evaluated rigorously before any answer is generated. The evaluation framework measures how well the retrieval pipeline finds the exact regulatory passages that ground a correct answer, not how good the generated answer sounds.

---

## 2. Evaluation Architecture

The evaluation system is split into two independent sub-pipelines:

```
                    ┌─────────────────────────────────────┐
                    │       Retrieval Evaluation           │
                    │                                      │
  Gold Dataset ─────┤─► ChromaDB (E5, BGE-M3, hybrid)    │──► results/
  (121 questions    │─► Qdrant + BM25 + RRF + rerank      │
   with chunk IDs)  │                                      │
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │      Generation Evaluation           │
                    │                                      │
  Gold Dataset ─────┤─► GenerationPipeline.generate()    │──► answers.csv
  (questions +      │                                      │
   ground truth     │─► ContextEnricher                   │──► enriched.csv
   answers)         │                                      │
                    │─► RagasEvaluator                     │──► ragas_scores.csv
                    │   (Llama 3.3-70B judge via NIM)     │    ragas_metrics.json
                    └─────────────────────────────────────┘
```

---

## 3. Gold Dataset

### Retrieval Gold Dataset

**File**: [gold_dataset_for_embedding_excel.xlsx](file:///c:/Users/user/Documents/banking-regulatory-intelligence-platform/data/evaluation/gold_dataset_for_embedding_excel.xlsx)

**Size**: 121 questions.

**Contents**: Each row contains:
- `question` — a natural language question in Azerbaijani.
- `relevant_chunk_ids` — one or more comma-separated chunk IDs that are the ground-truth relevant chunks for that question.

This dataset was **hand-labeled**: questions were derived from the regulatory corpus, and the relevant chunks were manually identified and verified. It is not synthetically generated.

The dataset covers questions across all eight regulatory categories, varying in specificity from broad definitional questions to precise article-level lookups.

### Generation Gold Dataset

**File**: [gold_dataset_for_llm_generation.xlsx](file:///c:/Users/user/Documents/banking-regulatory-intelligence-platform/data/evaluation/gold_dataset_for_llm_generation.xlsx)

**Contents**: Each row contains:
- `Question ID`
- `Question` — the regulatory question.
- `Ground Truth Answer` — a human-authored reference answer.
- `Category` — the regulatory category of the question.

This dataset is used to evaluate the quality of the generated answers against reference answers using RAGAS metrics.

---

## 4. Retrieval Evaluation Metrics

All retrieval metrics are computed at multiple cutoff values (K = 1, 3, 5, 10) to understand how quickly relevant results appear.

### Recall@K

**What it measures**: The fraction of relevant chunks that appear in the top-K retrieved results.

```
Recall@K = |Relevant ∩ Retrieved@K| / |Relevant|
```

In a regulatory setting, Recall@K measures whether the system surfaces all the chunks that contain the information needed to answer a question. High recall is critical — a missed relevant chunk means the LLM cannot cite that information.

### Precision@K

**What it measures**: The fraction of the top-K retrieved results that are relevant.

```
Precision@K = |Relevant ∩ Retrieved@K| / K
```

Precision@K measures result quality at each cutoff. As K increases, precision naturally decreases even for a good retriever, because more non-relevant results enter the top-K.

### MRR@K (Mean Reciprocal Rank)

**What it measures**: The average reciprocal rank of the first relevant result across all queries.

```
MRR@K = (1/|Q|) × Σ (1 / rank_of_first_relevant_result)
```

MRR@10 is a strong indicator of ranking quality: a system with MRR@10 = 0.765 places the first relevant result at rank 1 or 2 on average. This is critical for the generation stage, because higher-ranked chunks are more likely to be included in the prompt after context budget trimming.

### nDCG@K (Normalized Discounted Cumulative Gain)

**What it measures**: The relevance-weighted ranking quality, discounted by position.

```
DCG@K = Σ (rel_i / log2(i + 1))
nDCG@K = DCG@K / IDCG@K   (where IDCG is ideal DCG)
```

nDCG@K rewards systems that place relevant results at the top of the ranking. It accounts for multiple relevant results with position discounting — a relevant result at rank 1 contributes more than the same result at rank 5.

---

## 5. Retrieval Evaluation Methodology

### Evaluation Scripts

Three scripts cover the full pipeline evolution:

| Script | Pipeline | Backend |
|---|---|---|
| [run_retrieval_evaluation.py](file:///c:/Users/user/Documents/banking-regulatory-intelligence-platform/scripts/run_retrieval_evaluation.py) | Dense-only (E5 or BGE-M3) | ChromaDB |
| [run_hybrid_evaluation.py](file:///c:/Users/user/Documents/banking-regulatory-intelligence-platform/scripts/run_hybrid_evaluation.py) | Dense + BM25 + RRF (E5 or BGE-M3) | ChromaDB |
| [run_hybrid_qdrant_evaluation.py](file:///c:/Users/user/Documents/banking-regulatory-intelligence-platform/scripts/run_hybrid_qdrant_evaluation.py) | Dense + BM25 + RRF + CrossEncoder rerank | Qdrant |

### Evaluation Procedure

For each question in the gold dataset:

1. Generate the query embedding using the target model (with model-specific prefix: `"query: "` for E5, no prefix for BGE-M3).
2. Retrieve the top-K results from the vector store.
3. Compare retrieved chunk IDs against the `relevant_chunk_ids` from the gold dataset.
4. Compute per-question metrics: whether rank 1 is relevant, rank 3, rank 5, rank 10, the rank of the first relevant result, and full nDCG.
5. Aggregate metrics across all 121 questions.

### Output Files

Each evaluation run writes to a subdirectory under `results/`:

| File | Description |
|---|---|
| `metrics.json` | Aggregated metrics for this run |
| `per_question.csv` | Per-question metrics and retrieved IDs |
| `retrieval_results.csv` | Full ranked retrieval results per question |
| `comparison.csv` | Cross-run comparison table |

---

## 6. Embedding Model Comparison

Two embedding models were fully evaluated across both dense-only and hybrid retrieval pipelines:

### Model 1: `intfloat/multilingual-e5-large` (E5)

- **Architecture**: Asymmetric encoder — query and passage must use different prefixes (`"query: "` and `"passage: "`).
- **Evaluation**: Run on ChromaDB, dense-only and hybrid.

### Model 2: `BAAI/bge-m3` (BGE-M3)

- **Architecture**: Symmetric encoder — no prefix required for either queries or passages.
- **Dimension**: 1024.
- **Evaluation**: Run on ChromaDB (dense + hybrid) and Qdrant (hybrid + reranking).
- **Status**: Selected as the production embedding model.

Two additional models are implemented but have not yet been wired into evaluation scripts:

| Model | Status |
|---|---|
| `jinaai/jina-embeddings-v3` | Implemented in `JinaV3EmbeddingService` |
| `Qwen/Qwen3-Embedding-0.6B` | Implemented in `QwenEmbeddingService` |

Evaluating these models against the same gold dataset is future work.

---

## 7. Reranking Evaluation

### Cross-Encoder: `BAAI/bge-reranker-v2-m3`

The cross-encoder reranker is evaluated as part of [run_hybrid_qdrant_evaluation.py](file:///c:/Users/user/Documents/banking-regulatory-intelligence-platform/scripts/run_hybrid_qdrant_evaluation.py). It operates on the top-15 candidates produced by RRF fusion and rescores each `(query, chunk_text)` pair jointly — unlike bi-encoder models that encode query and passage independently.

**Model**: `BAAI/bge-reranker-v2-m3`.
**Input**: Pairs of `(query, chunk_text)`, one pair per candidate.
**Output**: A raw relevance score for each pair (higher = more relevant).
**Max input length**: 1,024 tokens.
**Batch size**: 32.

### Why Reranking Matters

Bi-encoder retrieval (dense or BM25) can return relevant passages that are ranked below non-relevant ones because the models encode query and document independently and cannot model their interaction. Cross-encoders process the query and document together, capturing fine-grained relevance signals that bi-encoders miss.

In the ReguAZ evaluation, reranking produced the largest single improvement in retrieval quality:

| Pipeline | Recall@10 | MRR@10 |
|---|---|---|
| Hybrid BGE-M3 (no rerank) | 0.360 | 0.270 |
| **Hybrid BGE-M3 + CrossEncoder** | **0.897** | **0.765** |

The reranking stage lifts Recall@10 from 0.360 to 0.897 — a 2.5× improvement — and MRR@10 from 0.270 to 0.765, placing the first relevant result at or near rank 1 on average.

---

## 8. Generation Evaluation Metrics

Generation quality is evaluated using **RAGAS** (Retrieval Augmented Generation Assessment), with a large LLM acting as judge.

### Faithfulness

**What it measures**: Whether every factual claim in the generated answer can be directly inferred from the retrieved context.

A high faithfulness score means the LLM is not hallucinating — it is grounding its answer strictly in what the retrieved regulatory passages say.

### Answer Relevancy

**What it measures**: Whether the generated answer is relevant to the question and addresses what was asked.

This metric uses embedding-based similarity between generated questions (reverse-engineered from the answer by the judge LLM) and the original question to detect off-topic or evasive answers.

### Context Recall

**What it measures**: Whether the retrieved context contains sufficient information to produce the ground-truth answer.

Context Recall evaluates whether the retrieval stage is bringing in the necessary information — independent of whether the LLM correctly used it. A low Context Recall score points to a retrieval failure, not a generation failure.

---

## 9. Generation Evaluation Methodology

### Step 1 — Generate Answers (`GenerationEvaluator`)

The script [run_generation_evaluation.py](file:///c:/Users/user/Documents/banking-regulatory-intelligence-platform/scripts/run_generation_evaluation.py) loads the generation gold dataset and runs `GenerationPipeline.generate()` for each question. Results are saved as `results/generation/gemma/answers.csv` with columns: `Question ID`, `Question`, `Ground Truth`, `Category`, `Generated Answer`, plus timing metrics.

### Step 2 — Enrich with Context (`ContextEnricher`)

The script [enrich_generation_results.py](file:///c:/Users/user/Documents/banking-regulatory-intelligence-platform/scripts/enrich_generation_results.py) runs the retriever for each evaluated question and attaches the retrieved chunks to the answers CSV, producing `enriched_answers.csv`. This file contains the `Retrieved Contexts` column required by RAGAS.

### Step 3 — RAGAS Evaluation (`RagasEvaluator`)

The script [run_ragas_evaluation.py](file:///c:/Users/user/Documents/banking-regulatory-intelligence-platform/scripts/run_ragas_evaluation.py) loads the enriched CSV and runs RAGAS metrics using:

- **Judge LLM**: `meta/llama-3.3-70b-instruct` via NVIDIA NIM API (`NVIDIA_API_KEY` required).
- **Embedding model** (for `AnswerRelevancy`): `intfloat/multilingual-e5-large` loaded locally.
- **Run config**: `max_workers=1`, `timeout=240 s`, `max_retries=3`.

Output:
- `results/ragas/ragas_scores.csv` — per-question metric scores.
- `results/ragas/ragas_metrics.json` — aggregated mean scores.

### Judge LLM Selection

The judge LLM (`meta/llama-3.3-70b-instruct`) is a large, capable instruction-following model accessed remotely via NVIDIA NIM, avoiding the need for a second large model on local hardware alongside Gemma. The judge model can be overridden via the `JUDGE_MODEL` environment variable.

---

## 10. Running Evaluations

All commands assume an activated Poetry environment at the project root.

### Retrieval Evaluation

```bash
# Dense-only evaluation (ChromaDB)
python scripts/run_retrieval_evaluation.py --top-k 10 --chroma-dir data/chroma

# Hybrid evaluation (ChromaDB) — both models
python scripts/run_hybrid_evaluation.py --model all --top-k 10 --chroma-dir data/chroma

# Hybrid + reranking evaluation (Qdrant) — production pipeline
python scripts/run_hybrid_qdrant_evaluation.py --top-k 10 --qdrant-dir data/qdrant
```

### Generation Evaluation

```bash
# Step 1: Generate answers
python scripts/run_generation_evaluation.py

# Step 2: Enrich with retrieved contexts
python scripts/enrich_generation_results.py

# Step 3: RAGAS evaluation (requires NVIDIA_API_KEY)
python scripts/run_ragas_evaluation.py
```

### Required Environment

| Evaluation | Required |
|---|---|
| Dense / Hybrid (ChromaDB) | Poetry env, ChromaDB data, embedding model weights |
| Hybrid + Rerank (Qdrant) | Poetry env, Qdrant data, BGE-M3 weights, reranker weights |
| Generation | All above + Gemma 3 4B GGUF model |
| RAGAS | All above + `NVIDIA_API_KEY` in `.env` |
