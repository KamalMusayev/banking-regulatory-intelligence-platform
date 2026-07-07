# ReguAZ

**AI-powered Retrieval-Augmented Generation (RAG) platform for Azerbaijani banking and financial regulations**

ReguAZ ingests, indexes, and semantically searches the corpus of Azerbaijani banking, AML/KYC, prudential, risk-management, and governance regulations, and pairs that retrieval layer with a local LLM generation stage so that regulatory questions can eventually be answered directly from the source documents — in Azerbaijani, and grounded only in retrieved text.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Current Features](#3-current-features)
4. [Pipeline Diagrams](#4-pipeline-diagrams)
5. [Project Structure](#5-project-structure)
6. [Technology Stack](#6-technology-stack)
7. [Installation Guide](#7-installation-guide)
8. [Running the Project](#8-running-the-project)
9. [Development Workflow](#9-development-workflow)
10. [Development Roadmap](#10-development-roadmap)
11. [License](#11-license)

---

## 1. Project Overview

### What ReguAZ is

ReguAZ is a domain-specific RAG system built around 96 real Azerbaijani banking-sector regulatory documents (laws, Central Bank rules, AML/KYC requirements, prudential and risk-management regulations, reporting/audit instructions, and governance standards). The project processes these PDFs into clean, chunked, embedded, and searchable text, and evaluates multiple retrieval strategies against a hand-curated gold question set before layering an LLM answer-generation stage on top.

### Why it exists

Azerbaijani banking regulation is spread across dozens of laws, Central Bank (CBAR) rules, and methodological guidance documents, often only available as long, unstructured PDFs. Manually searching this corpus for a specific requirement (e.g. minimum capital, AML thresholds, reporting deadlines) is slow and error-prone. ReguAZ's goal is to make this corpus queryable in natural language, with answers grounded strictly in the regulatory text — never hallucinated.

### The problem it solves

- Regulatory text is fragmented across many long PDFs with inconsistent formatting.
- Keyword search alone misses semantically related but lexically different phrasing.
- Pure semantic search alone can miss exact legal terms, article numbers, or defined terms.
- Domain experts need traceability back to the exact chunk/article/page a claim came from.

ReguAZ addresses this with a hybrid (semantic + keyword) retrieval pipeline, reranking, and a rigorously evaluated retrieval quality process, before answers are ever generated.

### Long-term vision

The long-term vision is a production-ready **regulatory intelligence assistant**: a user asks a question in Azerbaijani, the system retrieves the most relevant regulatory passages via hybrid search + reranking, and a local LLM generates a concise, citation-grounded, Azerbaijani-language answer — with an explicit refusal to answer when the retrieved context is insufficient. Today the retrieval half of that pipeline is implemented and evaluated; the generation half is implemented and independently verified; the two are not yet wired into a single end-to-end script or API.

---

## 2. Architecture Overview

ReguAZ is organized as a layered pipeline rather than a monolithic application. Each stage reads the output of the previous stage from disk (JSONL/JSON/Markdown), which keeps stages independently re-runnable and testable.

```
Raw PDFs → Extraction → Chunking → Embedding → Vector Storage → Retrieval → (Reranking) → LLM Generation
```

**Module responsibilities** (`backend/reguaz/`):

| Module | Responsibility |
|---|---|
| `config.py` | Centralized paths and default constants (batch sizes, top-k, RRF constant, supported models). |
| `services/ingestion/` | Chunk lookup utilities used during vector-DB ingestion. |
| `services/chunks/` | Read-only chunk lookup utilities used during evaluation/BM25. |
| `services/embeddings/` | One class per embedding model (E5, BGE-M3, Jina v3, Qwen3) behind a common `BaseEmbeddingService` interface, selected via `EmbeddingFactory`. |
| `database/` | Thin persistence managers for ChromaDB and Qdrant — collection lifecycle and batched inserts only, no retrieval logic. |
| `retrieval/` | Dense retrievers (`ChromaRetriever`, `QdrantRetriever`), `BM25Retriever`, RRF fusion (`fusion.py`), `CrossEncoderReranker`, and two orchestrating hybrid retrievers (`HybridRetriever` for Chroma, `HybridQdrantRetriever` for Qdrant + reranking). |
| `llm/` | Provider-agnostic LLM generation: `BaseLLMProvider` interface, `LocalInferenceProvider` (Hugging Face Transformers backend), `LLMProviderFactory`, `PromptBuilder` (Azerbaijani domain system prompt), and `Generator` (orchestration). |
| `utils/logger.py` | Shared logger factory (console + rotating file handlers) used across all modules. |

`scripts/` contains the CLI entry points that drive each stage (extraction, chunking, embedding, ingestion, evaluation, LLM demo/verification) — these are the operational interface to the `backend/reguaz` library code.

---

## 3. Current Features

### Document Processing

✅ Regulatory document collection — 96 PDFs across 8 categories
✅ PDF extraction (`pdfplumber`, page-marker-preserving, footer/page-number stripping)
✅ Text preprocessing (whitespace normalization, blank-line collapsing)
✅ Document chunking (chapter/article-aware regex segmentation + 4000-char sliding window with 500-char overlap)
✅ Metadata generation (per-document JSON: page count, chunk count, language, parser, timestamp)
✅ Stable, human-readable chunk IDs (`{document_id}_{chapter/article}_{index}`)

### Embeddings

✅ Embedding generation pipeline (writer/reader utilities, JSONL persistence)
✅ Multiple embedding model support behind a common interface:
  - `intfloat/multilingual-e5-large` (asymmetric, query/passage prefixes)
  - `BAAI/bge-m3`
  - `jinaai/jina-embeddings-v3` (implemented; not yet wired into ingestion/evaluation)
  - `Qwen/Qwen3-Embedding-0.6B` (implemented; not yet wired into ingestion/evaluation)

### Vector Storage

✅ ChromaDB integration (`ChromaDBManager`, `ChromaRetriever`)
✅ Qdrant integration (`QdrantDBManager`, `QdrantRetriever`), including deterministic UUID5 mapping of human-readable chunk IDs to Qdrant-compatible point IDs

### Retrieval

✅ Dense (semantic) retrieval — both ChromaDB and Qdrant backends
✅ BM25 keyword retrieval (`rank-bm25`, language-agnostic whitespace tokenizer suited to Azerbaijani/Russian/English mixed text)
✅ Hybrid retrieval via Reciprocal Rank Fusion (both a ChromaDB variant and a Qdrant variant)
✅ Cross-Encoder reranking (`BAAI/bge-reranker-v2-m3`) — currently applied in the Qdrant hybrid pipeline only

### Evaluation

✅ Retrieval evaluation framework (Recall@K, Precision@K, MRR@10, nDCG@K)
✅ Gold evaluation dataset (121 hand-labeled questions, Excel format)
✅ Embedding model benchmarking (E5 vs. BGE-M3, plain dense retrieval)
✅ Retrieval pipeline optimization, quantified end-to-end:

| Pipeline | Recall@10 | MRR@10 |
|---|---|---|
| Plain dense — E5 | 0.140 | 0.107 |
| Plain dense — BGE-M3 | 0.368 | 0.280 |
| Hybrid (BM25 + RRF) — E5 | 0.364 | 0.212 |
| Hybrid (BM25 + RRF) — BGE-M3 | 0.360 | 0.270 |
| **Hybrid + Cross-Encoder rerank (Qdrant, BGE-M3)** | **0.897** | **0.765** |

*(See [Development Roadmap](#10-development-roadmap) for what these numbers mean for project direction.)*

### LLM Architecture

✅ `BaseLLMProvider` — abstract, transport-agnostic provider interface
✅ `LocalInferenceProvider` — in-process Hugging Face Transformers backend (default model: `Qwen/Qwen2.5-3B-Instruct`)
✅ `PromptBuilder` — dedicated Azerbaijani-language system prompt enforcing context-only, non-hallucinated, formally worded answers
✅ `Generator` — orchestrates question + context → prompt → provider → answer
✅ `LLMProviderFactory` — single instantiation point; adding new providers (llama.cpp, remote, OpenAI-compatible) requires no changes elsewhere
✅ Dependency Injection — `Generator` receives an already-constructed provider; it is never responsible for building one
✅ Lazy loading — model/tokenizer load only on first `generate()` call, keeping factory construction fast
✅ Provider abstraction / backend isolation — swapping the inference backend touches only `_run_inference()`
✅ Chat-template readiness — `PromptBuilder._build_messages()` already returns role/content dicts, ready for `tokenizer.apply_chat_template()`
✅ Multi-device support — automatic MPS → CUDA → CPU detection and correct `device_map` handling per device

---

## 4. Pipeline Diagrams

### Document Processing Pipeline

```
┌──────────────┐     ┌───────────────────┐     ┌────────────────────┐
│  data/raw/   │     │ scripts/          │     │ data/processed/    │
│  *.pdf       │────▶│ extract_pdfs.py   │────▶│ cleaned_documents/ │
│ (8 category  │     │ (pdfplumber +     │     │ *.md (page-marked, │
│  subfolders) │     │  text cleaning)   │     │  normalized text)  │
└──────────────┘     └───────────────────┘     └─────────┬──────────┘
                                                          │
                                                          ▼
                                              ┌───────────────────────┐
                                              │ scripts/chunker.py    │
                                              │ (chapter/article-     │
                                              │  aware + size-based   │
                                              │  sliding window)      │
                                              └──────────┬────────────┘
                                                          │
                                    ┌─────────────────────┴─────────────────────┐
                                    ▼                                           ▼
                     ┌───────────────────────────┐               ┌───────────────────────────┐
                     │ data/processed/chunks/    │               │ data/processed/metadata/  │
                     │ *.jsonl (chunk_id, text,   │               │ *_metadata.json           │
                     │ chapter, article, pages)   │               │ (pages, chunk counts, …)  │
                     └───────────────────────────┘               └───────────────────────────┘
```

### Retrieval Pipeline

```
                              ┌───────────────────────┐
                              │        Query          │
                              └───────────┬────────────┘
                                          │
                     ┌────────────────────┼────────────────────┐
                     ▼                                          ▼
        ┌─────────────────────────┐                ┌─────────────────────────┐
        │ EmbeddingFactory        │                │ BM25Retriever            │
        │ (model-specific prefix, │                │ (whitespace tokenizer,   │
        │  e.g. "query: " for E5) │                │  BM25Okapi index)        │
        └────────────┬────────────┘                └────────────┬────────────┘
                     ▼                                          │
        ┌─────────────────────────┐                              │
        │ ChromaRetriever   /     │                              │
        │ QdrantRetriever         │                              │
        │ (dense k-NN search)     │                              │
        └────────────┬────────────┘                              │
                     └───────────────┬──────────────────────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │ fusion.py                     │
                     │ Reciprocal Rank Fusion (RRF)   │
                     └───────────────┬────────────────┘
                                     ▼
                  ┌──────────────────────────────────────┐
                  │ CrossEncoderReranker (Qdrant path)   │
                  │ BAAI/bge-reranker-v2-m3               │
                  └───────────────────┬───────────────────┘
                                      ▼
                        Ranked chunk_id list (top-k)
```

### LLM Pipeline

```
   question, context
          │
          ▼
┌───────────────────────┐
│ PromptBuilder          │   builds system + user messages
│ (Azerbaijani system    │   ("answer only from context,
│  prompt, context-only) │    never hallucinate")
└───────────┬────────────┘
            ▼
┌───────────────────────┐
│ LLMProviderFactory     │   selects provider ("local" today)
└───────────┬────────────┘
            ▼
┌───────────────────────┐
│ LocalInferenceProvider │   lazy-loads Qwen2.5-3B-Instruct
│ (HF Transformers,      │   on MPS / CUDA / CPU
│  device auto-detect)   │
└───────────┬────────────┘
            ▼
┌───────────────────────┐
│ Generator              │   orchestrates the above, returns
│ .generate_answer()      │   the final answer string
└───────────┬────────────┘
            ▼
          answer
```

### Complete RAG Pipeline (target — retrieval and generation are implemented but not yet wired together in one script)

```
 User Question
      │
      ▼
 HybridQdrantRetriever.retrieve()  ──▶  top-k chunk_ids  ──▶  fetch chunk text
      │
      ▼
 Assemble context string from retrieved chunks
      │
      ▼
 Generator.generate_answer(question, context)
      │
      ▼
 Grounded Azerbaijani-language answer
```

---

## 5. Project Structure

```
banking-regulatory-intelligence-platform/
│
├── backend/
│   ├── reguaz/
│   │   ├── config.py                     # Central paths & constants
│   │   ├── database/
│   │   │   ├── chroma.py                 # ChromaDB collection manager
│   │   │   └── qdrant.py                 # Qdrant collection manager
│   │   ├── llm/
│   │   │   ├── base.py                   # BaseLLMProvider (ABC)
│   │   │   ├── factory.py                # LLMProviderFactory
│   │   │   ├── generator.py              # Generator (orchestration)
│   │   │   ├── local_provider.py         # LocalInferenceProvider (HF Transformers)
│   │   │   └── prompt_builder.py         # PromptBuilder (AZ system prompt)
│   │   ├── retrieval/
│   │   │   ├── retriever.py              # ChromaRetriever
│   │   │   ├── qdrant_retriever.py        # QdrantRetriever
│   │   │   ├── bm25_retriever.py          # BM25Retriever
│   │   │   ├── fusion.py                 # Reciprocal Rank Fusion
│   │   │   ├── reranker.py               # CrossEncoderReranker
│   │   │   ├── hybrid_retriever.py       # Chroma hybrid orchestrator
│   │   │   └── hybrid_qdrant.py          # Qdrant hybrid + rerank orchestrator
│   │   ├── services/
│   │   │   ├── embeddings/               # BaseEmbeddingService + 4 model impls + factory
│   │   │   ├── ingestion/                # ChunkReader for ingestion scripts
│   │   │   └── chunks/                   # ChunkReader for evaluation/BM25
│   │   ├── utils/logger.py               # Shared logger factory
│   │   └── logs/                         # Runtime log files (module-level)
│   └── tests/                            # Placeholder — no tests implemented yet
│
├── scripts/
│   ├── extract_pdfs.py                   # Stage 1: PDF → cleaned Markdown
│   ├── chunker.py                        # Stage 2: Markdown → chunks + metadata
│   ├── run_embedding_pipeline.py         # Stage 3: chunks → embeddings
│   ├── run_chroma_ingestion.py           # Stage 4a: embeddings → ChromaDB
│   ├── run_qdrant_ingestion.py           # Stage 4b: embeddings → Qdrant
│   ├── run_retrieval_evaluation.py       # Plain dense retrieval evaluation
│   ├── run_hybrid_evaluation.py          # Hybrid (Chroma) retrieval evaluation
│   ├── run_hybrid_qdrant_evaluation.py   # Hybrid + rerank (Qdrant) evaluation
│   ├── export_candidates.py              # Helper: export top-k candidates for gold-set curation
│   ├── run_llm_demo.py                   # Manual LLM pipeline demo
│   └── verify_llm_module.py              # LLM module import/behavior smoke test
│
├── data/
│   ├── raw/                              # 96 source PDFs, 8 category subfolders
│   ├── processed/
│   │   ├── cleaned_documents/            # Extracted Markdown
│   │   ├── chunks/                       # Chunked JSONL (3,806 chunks)
│   │   ├── metadata/                     # Per-document metadata JSON
│   │   └── embeddings/                   # Per-model embedding JSONL (bge_m3, e5)
│   ├── chroma/                           # ChromaDB persisted vector store (gitignored contents)
│   └── evaluation/                       # Gold datasets (retrieval + LLM-generation)
│
├── docs/                                 # architecture.md, api_specification.md, evaluation_plan.md,
│                                          # experiment_results.md, project_scope.md — currently empty
│                                          # placeholders; folder_structure.md sketches a future,
│                                          # not-yet-implemented FastAPI application layer
│
├── logs/                                 # Root-level pipeline logs (extraction, chunking, ingestion, eval)
├── results/                               # Evaluation outputs: metrics.json, per-question CSV, comparisons
├── docker-compose.yml                    # Present but currently empty
├── pyproject.toml / poetry.lock           # Poetry project definition
└── README.md
```

**Directory notes:**
- `data/raw/` contains 96 PDFs across 8 categories: `laws`, `aml_kyc`, `governance_and_compliance`, `guidance_and_methodology`, `payments_and_banking_operations`, `prudential_regulations`, `reporting_and_audit`, `risk_management`.
- `data/chroma/` holds ChromaDB's persisted HNSW index files; its *contents* are gitignored (only structure is versioned), as is `data/qdrant/`.
- `docs/` is mostly placeholder today — `folder_structure.md` documents an aspirational, FastAPI-based application layer (`backend/app/api`, `services/rag`, etc.) that does not exist in the codebase yet. Treat it as a roadmap sketch, not current architecture.

---

## 6. Technology Stack

| Category | Technology |
|---|---|
| Language | Python (3.12 – 3.14) |
| Dependency management | Poetry |
| PDF extraction | pdfplumber |
| Embeddings | sentence-transformers (`intfloat/multilingual-e5-large`, `BAAI/bge-m3`, `jinaai/jina-embeddings-v3`, `Qwen/Qwen3-Embedding-0.6B`) |
| Vector databases | ChromaDB, Qdrant (`qdrant-client`) |
| Keyword search | rank-bm25 |
| Reranking | sentence-transformers `CrossEncoder` (`BAAI/bge-reranker-v2-m3`) |
| LLM inference | Hugging Face Transformers (`Qwen/Qwen2.5-3B-Instruct`), PyTorch, Accelerate |
| Data handling | pandas, openpyxl (gold-dataset Excel files) |
| ML acceleration | einops, Accelerate, PyTorch (CUDA / MPS / CPU) |

---

## 7. Installation Guide

### 7.1 Prerequisites

- **Python 3.12** (project requires `>=3.12,<3.15`)
- **Poetry** for dependency management
- (Optional) a **Hugging Face account/token** — not required for the public models currently used, but recommended if you configure gated or private models via `LocalInferenceProvider`/embedding services

### 7.2 Clone the repository

```bash
git clone https://github.com/KamalMusayev/banking-regulatory-intelligence-platform.git
cd banking-regulatory-intelligence-platform
```

### 7.3 Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry --version
```

### 7.4 Install Python 3.12

```bash
python3.12 --version
```

Install it first if missing (via `pyenv`, your OS package manager, or python.org).

### 7.5 Create the Poetry virtual environment

```bash
poetry env use python3.12
```

### 7.6 Install project dependencies

```bash
poetry install
```

This installs everything declared in `pyproject.toml` / `poetry.lock`, including `torch`, `chromadb`, `qdrant-client`, `sentence-transformers`, and `pdfplumber`.

### 7.7 Activate the virtual environment

```bash
poetry shell
# or, on newer Poetry versions:
poetry env activate
```

### 7.8 Environment variables

The codebase does not currently read any required environment variables (`config.py` uses only filesystem paths). A `.env` file is gitignored for future use — e.g. if you configure a gated Hugging Face model, export a token before running any embedding/LLM script:

```bash
export HF_TOKEN=your_huggingface_token
# or
huggingface-cli login
```

### 7.9 Verify the installation

```bash
python --version        # Expect: Python 3.12.x
poetry show             # Lists installed packages
python scripts/verify_llm_module.py   # Smoke-tests the LLM module imports and abstract-class behavior
```

---

## 8. Running the Project

All commands assume you are in the activated Poetry environment at the project root.

### 8.1 Document processing

```bash
# Stage 1 — Extract text from raw PDFs into cleaned Markdown
python scripts/extract_pdfs.py

# Stage 2 — Chunk cleaned Markdown into JSONL chunks + metadata
python scripts/chunker.py
```

### 8.2 Embedding generation

```bash
python scripts/run_embedding_pipeline.py --model bge_m3
python scripts/run_embedding_pipeline.py --model e5
```

### 8.3 Vector-store ingestion

```bash
# ChromaDB
python scripts/run_chroma_ingestion.py

# Qdrant (production path; drops and recreates the collection each run)
python scripts/run_qdrant_ingestion.py --batch-size 256
```

### 8.4 Retrieval evaluation

```bash
# Plain dense retrieval (E5 vs BGE-M3)
python scripts/run_retrieval_evaluation.py --top-k 10 --chroma-dir data/chroma

# Hybrid retrieval (BM25 + RRF), ChromaDB backend
python scripts/run_hybrid_evaluation.py --model all --top-k 10 --chroma-dir data/chroma

# Hybrid retrieval + Cross-Encoder reranking, Qdrant backend
python scripts/run_hybrid_qdrant_evaluation.py --top-k 10 --qdrant-dir data/qdrant
```

Each evaluation script writes `metrics.json`, `per_question.csv`, `retrieval_results.csv`, and a cross-run `comparison.csv` under `results/`.

### 8.5 LLM verification and demo

```bash
# Verifies imports, abstract-class enforcement, and basic wiring
python scripts/verify_llm_module.py

# Runs a hardcoded question/context through the full Prompt → LLM → Answer pipeline
python scripts/run_llm_demo.py
```

### 8.6 Candidate export (gold-dataset curation helper)

```bash
python scripts/export_candidates.py
```

Retrieves top-10 ChromaDB (E5) candidates per gold-dataset question and writes them to `data/evaluation/chunk_candidates.json`, used when manually curating/expanding the gold evaluation set.

---

## 9. Development Workflow

### Branching

- Branch from `main` using a descriptive prefix, e.g. `feature/hybrid-qdrant-rerank`, `fix/chunker-empty-pages`, `docs/readme-rewrite`.
- Keep branches scoped to a single pipeline stage or concern where possible (extraction, chunking, embeddings, retrieval, LLM).

### Commits

- Write imperative, present-tense commit messages (`Add Cross-Encoder reranking to Qdrant hybrid retriever`, not `Added...`).
- Prefer small, reviewable commits over large multi-stage changes, since each pipeline stage can be tested independently against its on-disk inputs/outputs.

### Pull Requests

- Describe which pipeline stage(s) the PR touches and how it was validated (e.g. "re-ran `run_hybrid_qdrant_evaluation.py`, recall@10 unchanged at 0.897").
- Since there is currently no automated test suite, include the manual verification steps taken (which script(s) were run, against which data).

### Poetry workflow

```bash
poetry show                 # list installed packages
poetry add <package>        # add a new dependency
poetry remove <package>     # remove a dependency
poetry lock                 # refresh the lock file after manual pyproject.toml edits
poetry install               # sync the environment to the lock file
```

### Daily workflow

```bash
cd banking-regulatory-intelligence-platform
poetry shell                # or: poetry env activate
git pull origin main
poetry install
# ... make changes ...
git add .
git commit -m "Your commit message"
git push origin <your-branch>
```

---

## 10. Development Roadmap

# Development Roadmap

| Phase | Status | Completed Work | Next Step |
|--------|--------|----------------|-----------|
| **1. Regulatory Data Collection** | ✅ Completed | Collected and organized Azerbaijani banking regulations and related legal documents across multiple regulatory categories. | — |
| **2. PDF Processing & Text Extraction** | ✅ Completed | Implemented PDF parsing, text extraction, cleaning, and preprocessing pipeline for regulatory documents. | Improve extraction quality for complex PDF layouts if needed. |
| **3. Document Chunking & Metadata Generation** | ✅ Completed | Built chapter/article-aware chunking with overlap, hierarchical chunk IDs, and rich metadata generation. | — |
| **4. Embedding Pipeline** | ✅ Completed | Implemented a modular embedding pipeline with support for multiple embedding models and writer/reader architecture. | Continue benchmarking newly added embedding models. |
| **5. Vector Database Integration** | ✅ Completed | Integrated both ChromaDB and Qdrant with deterministic indexing and retrieval support. | Continue using Qdrant as the primary production backend. |
| **6. Dense Semantic Retrieval** | ✅ Completed | Implemented semantic retrieval for both vector databases using sentence embeddings. | — |
| **7. BM25 Lexical Retrieval** | ✅ Completed | Implemented BM25 keyword retrieval for traditional lexical search. | — |
| **8. Hybrid Retrieval** | ✅ Completed | Combined semantic retrieval and BM25 using Reciprocal Rank Fusion (RRF). | Continue tuning retrieval weights and fusion parameters. |
| **9. CrossEncoder Reranking** | ✅ Completed | Added CrossEncoder reranking to improve final document ranking after hybrid retrieval. | Optimize inference latency and reranker performance. |
| **10. Retrieval Evaluation Framework** | ✅ Completed | Built a complete evaluation framework including Recall@K, Precision@K, MRR, nDCG, latency metrics, and retrieval benchmarking. | Extend evaluation with additional datasets when available. |
| **11. Retrieval Optimization & Benchmarking** | ✅ Completed | Benchmarked multiple embedding models, validated dense vs. hybrid vs. hybrid+rereanking pipelines, and significantly improved retrieval performance. | Continue evaluating future embedding models. |
| **12. LLM Architecture (Phase 1)** | ✅ Completed | Designed and implemented a modular LLM architecture including Provider abstraction, PromptBuilder, Generator, LocalInferenceProvider, Factory pattern, dependency injection, lazy loading, and verification utilities. | — |
| **13. Local LLM Integration** | 🔄 In Progress | Local inference pipeline has been integrated into the architecture and successfully validated. | Finalize local inference stability and select the best instruction model for production. |
| **14. End-to-End RAG Pipeline** | ⏳ Planned | — | Connect the retrieval pipeline directly with the LLM generation pipeline to build a complete RAG workflow. |
| **15. RAG Evaluation** | ⏳ Planned | — | Evaluate generated answers for groundedness, correctness, and faithfulness using retrieval context. |
| **16. Production Deployment** | ⏳ Planned | — | Build REST API, Docker deployment, monitoring, CI/CD, and production infrastructure. |

---

## 11. License

This project is licensed under the MIT License.
