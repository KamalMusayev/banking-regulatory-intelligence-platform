# ReguAZ

> **AI-powered Regulatory Intelligence Platform for Azerbaijani Banking Regulations**

ReguAZ is a Retrieval-Augmented Generation (RAG) project designed to search and retrieve information from Azerbaijani banking regulations using semantic search and vector embeddings.

The project is currently focused on building a reliable retrieval pipeline, evaluating embedding models, and laying the foundation for a production-ready regulatory AI assistant.

---

# Overview

ReguAZ enables semantic search over banking regulations by combining document processing, vector embeddings, and dense retrieval techniques.

Current development focuses on:

- Document processing
- Text chunking
- Embedding generation
- Semantic retrieval
- Retrieval evaluation
- Embedding model comparison

---

# Features

- Regulatory document processing
- Automatic text chunking
- Multiple embedding model support
- ChromaDB vector storage
- Dense semantic retrieval
- Retrieval evaluation framework

---

# Current Status

Implemented:

- Document ingestion pipeline
- Chunking pipeline
- Embedding generation
- ChromaDB integration
- Semantic retrieval
- Retrieval evaluation
- Embedding model comparison

Currently improving:

- Gold evaluation dataset
- Retrieval quality
- Chunking strategy
- Metadata consistency

---

# Project Structure

```text
backend/
data/
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

# Technology Stack

- Python
- Poetry
- ChromaDB
- Sentence Transformers
- Hugging Face Transformers
- Pandas
- NumPy

---

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

- Create a virtual environment
- Install all dependencies from `poetry.lock`

---

## 6. Activate the Virtual Environment

```bash
poetry shell
```

If `poetry shell` is unavailable:

```bash
poetry env activate
```

Copy and run the command returned by Poetry.

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

# Daily Workflow

Whenever you start working on the project:

```bash
cd banking-regulatory-intelligence-platform

poetry shell
```

If needed:

```bash
poetry env activate
```

Then:

```bash
git pull origin main
poetry install
```

After making changes:

```bash
git add .
git commit -m "Your commit message"
git push origin <your-branch>
```

---

# Useful Poetry Commands

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

Install dependencies:

```bash
poetry install
```

Update the lock file:

```bash
poetry lock
```

---

# License

This project is licensed under the MIT License.




| Phase                             | Status         | Completed Work                                                                                       | Next Step                                                                                    |
| --------------------------------- | -------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **1. Data Collection**            | ✅ Completed    | Collected Azerbaijani banking laws, regulations, and related documents.                              | —                                                                                            |
| **2. Parsing & Chunking**         | ✅ Completed    | Parsed PDFs and implemented a document chunking pipeline with a new chunk ID format.                 | Improve chunking strategy based on evaluation results.                                       |
| **3. Embedding Pipeline**         | ✅ Completed    | Generated embeddings using **E5** and **BGE-M3** models.                                             | Select the best-performing embedding model.                                                  |
| **4. Vector Database**            | ✅ Completed    | Created separate **ChromaDB** collections for each embedding model.                                  | Continue development using the selected embedding model.                                     |
| **5. Retrieval Pipeline**         | ✅ Completed    | Implemented semantic retrieval over ChromaDB.                                                        | Extend to Hybrid Retrieval.                                                                  |
| **6. Evaluation Pipeline**        | ✅ Completed    | Built an evaluation framework (Recall@K, MRR, etc.).                                                 | Re-evaluate after fixing the gold dataset.                                                   |
| **7. Gold Evaluation Dataset**    | 🔄 In Progress | Identified that the current dataset contains **outdated chunk IDs** from an older chunking pipeline. | Rebuild the `relevant_chunk_ids` manually using the current chunk IDs.                       |
| **8. Embedding Model Comparison** | ⏳ Pending      | Waiting for the corrected gold dataset.                                                              | Compare **E5** and **BGE-M3**, then select the best model.                                   |
| **9. Chunking Optimization**      | ⏳ Planned      | —                                                                                                    | Improve chunk size, overlap, and document splitting strategy based on retrieval errors.      |
| **10. BM25 Integration**          | ⏳ Planned      | Add BM25 lexical retrieval (`rank-bm25`).                                                            | Build the lexical retrieval component.                                                       |
| **11. Hybrid Retrieval**          | ⏳ Planned      | —                                                                                                    | Combine semantic retrieval with BM25 using a fusion strategy (e.g., Reciprocal Rank Fusion). |
| **12. Reranking**                 | ⏳ Planned      | —                                                                                                    | Add a CrossEncoder reranker to improve the ranking of retrieved chunks.                      |
| **13. LLM Integration**           | ⏳ Planned      | —                                                                                                    | Integrate **LlamaIndex** and an LLM to complete the RAG pipeline.                            |
| **14. Final Evaluation**          | ⏳ Planned      | —                                                                                                    | Evaluate the complete pipeline (Hybrid Retrieval + Reranker + LLM).                          |
