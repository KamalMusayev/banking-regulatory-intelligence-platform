# ReguAZ

> **AI-powered Regulatory Intelligence Platform for Azerbaijani Banking Regulations**

ReguAZ is a Retrieval-Augmented Generation (RAG) platform designed to search, retrieve, and answer questions from Azerbaijani banking regulations using semantic search and Large Language Models (LLMs).

The project aims to become a production-grade AI assistant for regulatory compliance by combining modern retrieval techniques, vector search, hybrid retrieval, reranking, and LLMs.

---

# Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Evaluation](#evaluation)
- [Technology Stack](#technology-stack)
- [Current Status](#current-status)
- [Roadmap](#roadmap)
- [Example Workflow](#example-workflow)
- [Future Goals](#future-goals)
- [License](#license)

---

# Overview

ReguAZ is an AI-powered legal retrieval platform built specifically for Azerbaijani banking regulations.

Instead of relying on keyword search, ReguAZ performs semantic retrieval using embedding models and vector databases, allowing users to locate the most relevant legal passages efficiently.

The project emphasizes:

- Semantic Search
- Retrieval-Augmented Generation (RAG)
- Embedding Evaluation
- Retrieval Evaluation
- Explainable AI with source citations
- Scalable Retrieval Architecture

---

# Features

## Document Processing

- PDF ingestion pipeline
- Regulatory document parsing
- Metadata extraction
- Automatic chunk generation

## Semantic Retrieval

- Dense vector search
- Multiple embedding models
- Top-K retrieval
- Metadata-aware retrieval

## Embedding Models

Currently supported:

- E5
- BGE-M3

## Evaluation

- Gold evaluation dataset
- Retrieval benchmarking
- Model comparison
- Recall@K
- Mean Reciprocal Rank (MRR)

---

# Architecture

```text
                 Regulatory Documents
                         │
                         ▼
               Document Processing
                         │
                         ▼
                   Text Chunking
                         │
                         ▼
                Embedding Generation
                (E5 / BGE-M3)
                         │
                         ▼
                    Vector Database
                     (ChromaDB)
                         │
                         ▼
                     Retriever
                         │
                         ▼
                Top-K Relevant Chunks
                         │
                         ▼
              Large Language Model
                  (Future Phase)
                         │
                         ▼
             Final Answer + Citations
```

---

# Project Structure

```text
backend/
│
├── ingestion/
├── retrieval/
├── services/
│   ├── embeddings/
│   └── ...
├── chunking/
├── evaluation/
└── utils/

data/
│
├── raw/
├── processed/
├── chroma/
└── evaluation/

results/

scripts/

tests/
```

---

# Retrieval Pipeline

Current retrieval workflow:

1. Load regulatory documents
2. Parse document content
3. Generate semantic chunks
4. Create embeddings
5. Store vectors in ChromaDB
6. Convert user query into embedding
7. Retrieve Top-K similar chunks
8. Return retrieved passages

---

# Evaluation

The project includes a dedicated retrieval evaluation framework.

Evaluation metrics include:

- Recall@1
- Recall@3
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)

Supported evaluation models:

- E5
- BGE-M3

Evaluation outputs are automatically exported to:

```text
results/
```

Generated files include:

- e5_results.csv
- bge_m3_results.csv
- comparison.csv

---

# Technology Stack

## Programming Language

- Python

## Machine Learning

- Sentence Transformers
- HuggingFace Transformers

## Vector Database

- ChromaDB

## Data Processing

- Pandas
- NumPy

## Dependency Management

- Poetry

## Utilities

- OpenPyXL
- JSON
- Pathlib

---

# Current Status

## Completed

- Document ingestion pipeline
- Chunking pipeline
- Embedding generation
- ChromaDB integration
- Semantic retrieval
- Retrieval evaluation framework
- Multiple embedding model support
- Model comparison framework

## In Progress

- Gold evaluation dataset refinement
- Retrieval quality improvement
- Chunk optimization
- Metadata consistency validation

---

# Roadmap

## Phase 1 — Evaluation & Retrieval

- Improve Gold Evaluation Dataset
- Validate retrieval quality
- Optimize chunking strategy
- Improve semantic retrieval accuracy

---

## Phase 2 — Hybrid Retrieval

Planned additions:

- Dense Retrieval
- BM25 Sparse Retrieval
- Reciprocal Rank Fusion (RRF)
- Hybrid Search

---

## Phase 3 — Retrieval Optimization

- Cross-Encoder Reranking
- Better metadata filtering
- Query optimization
- Retrieval latency improvements

---

## Phase 4 — Vector Database Migration

Planned migration:

- ChromaDB
    ↓
- Qdrant

Goals:

- Better scalability
- Faster indexing
- Native hybrid retrieval
- Improved filtering

---

## Phase 5 — RAG Pipeline

Planned additions:

- LlamaIndex integration
- Context optimization
- Prompt engineering
- Source-aware retrieval

---

## Phase 6 — LLM Integration

Potential models:

- Llama 3
- Qwen
- DeepSeek
- Mistral

Capabilities:

- Question Answering
- Regulatory Guidance
- Citation Generation
- Context-Aware Responses

---

## Phase 7 — User Interface

Planned features:

- Streamlit dashboard
- Interactive chat interface
- Retrieved chunk visualization
- Source highlighting
- Evaluation dashboard

---

# Example Workflow

```text
              User Question
                    │
                    ▼
           Query Embedding
                    │
                    ▼
            Vector Retrieval
                    │
                    ▼
        Top-K Relevant Chunks
                    │
                    ▼
              RAG Pipeline
                    │
                    ▼
             Large Language Model
                    │
                    ▼
      Final Answer + Source Citations
```

---

# Future Goals

The long-term objective is to build a production-ready AI assistant capable of assisting legal professionals, compliance teams, and financial institutions by providing reliable, explainable, and citation-backed answers from Azerbaijani banking regulations.

Key objectives include:

- Production-grade RAG system
- Hybrid retrieval architecture
- Explainable AI responses
- High retrieval accuracy
- Efficient semantic search
- Source-backed legal answers
- Scalable deployment

---

# License

This project is licensed under the MIT License.




# Project Setup Guide

## 1. Clone the Repository

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd banking-regulatory-intelligence-platform
```

---

## 2. Install Poetry

If Poetry is not installed:

### macOS / Linux

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Verify the installation:

```bash
poetry --version
```

---

## 3. Install Python 3.12

The project uses **Python 3.12**.

Check your installed version:

```bash
python3.12 --version
```

If Python 3.12 is not installed, install it first.

---

## 4. Create the Poetry Virtual Environment

Tell Poetry to use Python 3.12:

```bash
poetry env use python3.12
```

---

## 5. Install Project Dependencies

```bash
poetry install
```

Poetry will:

* create a virtual environment,
* install all dependencies from `poetry.lock`.

---

## 6. Activate the Virtual Environment

```bash
poetry shell
```

Your terminal should now look similar to:

```text
(reguaz-py3.12)
```

---

## 7. Verify the Installation

Check the Python version:

```bash
python --version
```

Expected output:

```text
Python 3.12.x
```

List installed packages:

```bash
poetry show
```

---

## 8. Deactivate the Environment

When finished:

```bash
exit
```

or press:

```text
Ctrl + D
```

---

## Daily Workflow

Whenever you start working on the project:

```bash
cd banking-regulatory-intelligence-platform
poetry shell or poetry env activate => and copy paste the output 
git pull origin main
```

After making changes:

```bash
git add .
git commit -m "Your commit message"
git push origin <your-branch>
```

---

## Useful Poetry Commands

Show installed packages:

```bash
poetry show
```

Add a dependency:

```bash
poetry add package_name
```

Remove a dependency:

```bash
poetry remove package_name
```

Install dependencies after pulling new changes:

```bash
poetry install
```

Update the lock file:

```bash
poetry lock
```
