# ReguAZ

> **AI-powered Regulatory Intelligence Platform for Azerbaijani Banking Regulations**

ReguAZ is a Retrieval-Augmented Generation (RAG) platform designed to retrieve and answer questions from Azerbaijani banking regulations using semantic search, hybrid retrieval, and Large Language Models (LLMs).

The current development phase focuses on building a robust retrieval system before integrating an LLM into the complete RAG pipeline.

---

# Overview

ReguAZ enables semantic and lexical search over Azerbaijani banking regulations through a modular pipeline consisting of document processing, chunking, embedding generation, vector search, hybrid retrieval, reranking, and evaluation.

The project is designed with interchangeable components, making it easy to experiment with different embedding models, retrieval strategies, rerankers, and vector databases.

---

# Current Features

- Regulatory document ingestion
- PDF parsing
- Automatic document chunking
- Chunk metadata generation
- Embedding generation
- Multiple embedding model support
- ChromaDB support
- Qdrant support
- Dense semantic retrieval
- BM25 lexical retrieval
- Hybrid Retrieval (Semantic + BM25)
- Reciprocal Rank Fusion (RRF)
- CrossEncoder reranking
- Retrieval evaluation framework
- Embedding model comparison
- Detailed evaluation reports

---

# Current Retrieval Pipeline

```
Documents
      │
      ▼
 Parsing
      │
      ▼
 Chunking
      │
      ▼
 Embedding Generation
      │
      ▼
 Vector Database
 (Qdrant / ChromaDB)
      │
      ▼
Semantic Retrieval
      │
      ├───────────────┐
      │               │
      ▼               ▼
 Qdrant Search     BM25 Search
      │               │
      └──────┬────────┘
             ▼
 Reciprocal Rank Fusion (RRF)
             ▼
 CrossEncoder Reranker
             ▼
 Final Ranked Results
```

---

# Technology Stack

- Python
- Poetry
- Qdrant
- ChromaDB
- Sentence Transformers
- Hugging Face Transformers
- rank-bm25
- Pandas
- NumPy
- OpenPyXL

---

# Project Structure

```text
backend/
│
├── reguaz/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── retrieval/
│   ├── services/
│   ├── evaluation/
│   └── utils/
│
data/
│
├── raw/
├── processed/
├── embeddings/
├── qdrant/
├── chroma/
└── evaluation/
│
docs/
logs/
results/
scripts/

README.md
docker-compose.yml
pyproject.toml
poetry.lock
```

---

# Project Setup Guide

## 1. Clone the Repository

Clone the repository:

```bash
git clone <repository-url>
```

Navigate into the project:

```bash
cd banking-regulatory-intelligence-platform
```

---

## 2. Install Python

ReguAZ currently targets **Python 3.12**.

Verify your installation:

```bash
python --version
```

or

```bash
python3 --version
```

Expected:

```text
Python 3.12.x
```

If Python is not installed, download it from:

https://www.python.org/downloads/

---

## 3. Install Poetry

Install Poetry.

### Windows (PowerShell)

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### macOS / Linux

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Verify:

```bash
poetry --version
```

---

## 4. Configure Poetry

Create the virtual environment using Python 3.12.

Windows

```bash
poetry env use python
```

macOS/Linux

```bash
poetry env use python3.12
```

---

## 5. Install Dependencies

Install every dependency defined in `pyproject.toml` and `poetry.lock`.

```bash
poetry install
```

This command will:

- Create a virtual environment
- Install all project dependencies
- Install development dependencies

---

## 6. Activate the Virtual Environment

Preferred:

```bash
poetry shell
```

If your Poetry version does not support `poetry shell`:

```bash
poetry env activate
```

Run the command returned by Poetry.

---

## 7. Verify Installation

Check Python:

```bash
python --version
```

Check Poetry environment:

```bash
poetry env info
```

List installed packages:

```bash
poetry show
```

---

## 8. Environment Variables

If the project requires environment variables, create a `.env` file in the project root.

Example:

```text
HF_TOKEN=your_huggingface_token
```

A Hugging Face token is optional but recommended because it:

- avoids download rate limits
- speeds up model downloads
- improves reliability when downloading models

---

## 9. Running Individual Pipelines

### Chunking

```bash
python scripts/run_chunking_pipeline.py
```

### Embedding Generation

```bash
python scripts/run_embedding_pipeline.py
```

### Chroma Retrieval Evaluation

```bash
python scripts/run_retrieval_evaluation.py
```

### Qdrant Retrieval Evaluation

```bash
python scripts/run_qdrant_retrieval_evaluation.py
```

### Hybrid Retrieval Evaluation

```bash
python scripts/run_hybrid_qdrant_evaluation.py
```

---

## 10. Deactivate the Environment

Exit the virtual environment:

```bash
exit
```

or press

```text
Ctrl + D
```

---

# Daily Development Workflow

Every time you start working:

```bash
cd banking-regulatory-intelligence-platform
```

Activate Poetry:

```bash
poetry shell
```

or

```bash
poetry env activate
```

Pull the latest changes:

```bash
git pull origin main
```

Install any new dependencies:

```bash
poetry install
```

Work on your feature.

After finishing:

```bash
git add .
```

```bash
git commit -m "Meaningful commit message"
```

```bash
git push origin <your-branch>
```

Create a Pull Request and merge after review.

---

# Useful Poetry Commands

Install dependencies

```bash
poetry install
```

Add a dependency

```bash
poetry add package_name
```

Remove a dependency

```bash
poetry remove package_name
```

Update dependencies

```bash
poetry update
```

Regenerate lock file

```bash
poetry lock
```

Show installed packages

```bash
poetry show
```

Show virtual environment

```bash
poetry env info
```

List available environments

```bash
poetry env list
```

Remove environment

```bash
poetry env remove python
```

Run a command inside Poetry

```bash
poetry run python script.py
```

---

# Development Roadmap

| Phase | Status | Completed Work | Next Step |
|--------|--------|----------------|-----------|
| **1. Data Collection** | ✅ Completed | Collected Azerbaijani banking regulations and related legal documents. | — |
| **2. Parsing & Chunking** | ✅ Completed | Built the document parsing and chunking pipeline with metadata generation and the current chunk ID format. | — |
| **3. Embedding Pipeline** | ✅ Completed | Generated embeddings using E5 and BGE-M3 models. | Continue benchmarking if additional embedding models are introduced. |
| **4. Vector Databases** | ✅ Completed | Implemented both ChromaDB and Qdrant vector stores. | Continue development using Qdrant as the primary backend. |
| **5. Dense Retrieval** | ✅ Completed | Implemented semantic retrieval for both ChromaDB and Qdrant. | — |
| **6. BM25 Retrieval** | ✅ Completed | Implemented lexical retrieval using BM25. | — |
| **7. Hybrid Retrieval** | ✅ Completed | Combined semantic retrieval and BM25 using Reciprocal Rank Fusion (RRF). | Fine-tune retrieval parameters. |
| **8. CrossEncoder Reranking** | ✅ Completed | Added CrossEncoder reranking using BGE Reranker. | Optimize inference speed and reranking performance. |
| **9. Evaluation Framework** | ✅ Completed | Built evaluation pipelines for semantic and hybrid retrieval with Recall@K, Precision@K, MRR, NDCG and timing metrics. | Continue improving evaluation quality. |
| **10. Gold Evaluation Dataset** | ✅ Completed | The dataset is being rebuilt using the latest chunk IDs generated by the current chunking pipeline. | Finalize the gold dataset and validate all annotations. |
| **11. Embedding Model Benchmarking** | ✅ Completed | Retrieval evaluation for BGE-M3 and E5 is ongoing. | Select the best embedding model based on evaluation metrics. |
| **12. Retrieval Optimization** | 🔄 In Progress | Hybrid retrieval and reranking are implemented and currently being optimized. | Tune BM25, RRF, reranker, and retrieval parameters. |
| **13. LlamaIndex Integration** | ⏳ Planned | — | Integrate LlamaIndex as the orchestration layer for retrieval and response generation. |
| **14. LLM Integration** | ⏳ Planned | — | Integrate an LLM to build the complete RAG pipeline. |
| **15. End-to-End RAG Evaluation** | ⏳ Planned | — | Evaluate the full RAG system including retrieval and answer generation. |
| **16. Production Deployment** | ⏳ Planned | — | Build API endpoints, monitoring, containerization, and deployment infrastructure. |

---

# License

This project is licensed under the MIT License.
