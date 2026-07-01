# ReguAZ

AI-powered regulatory intelligence platform for Azerbaijani banking regulations.

ReguAZ is a Retrieval-Augmented Generation (RAG) project focused on semantic search and retrieval over Azerbaijani banking regulations. The goal is to help users efficiently find relevant regulatory information using modern embedding models and vector search.

---

## Features

- Regulatory document processing
- Automatic text chunking
- Embedding generation
- ChromaDB vector storage
- Semantic retrieval
- Multiple embedding model support
- Retrieval evaluation framework

---

## Current Status

Implemented components:

- Document processing pipeline
- Chunking pipeline
- Embedding generation (E5, BGE-M3)
- ChromaDB integration
- Semantic retrieval
- Retrieval evaluation and model comparison

The project is actively under development, with retrieval quality, evaluation datasets, and the RAG pipeline continuously being improved.

---

## Project Structure

```text
backend/
data/
docs/
logs/
results/
scripts/

README.md
pyproject.toml
poetry.lock
docker-compose.yml
```

---

## Getting Started

### Clone the repository

```bash
git clone <repository-url>
cd banking-regulatory-intelligence-platform
```

### Install dependencies

```bash
poetry install
```

### Activate the environment

```bash
poetry shell
```

---

## Daily Workflow

```bash
git pull origin main

poetry install

poetry shell
```

After making changes:

```bash
git add .
git commit -m "Your commit message"
git push origin <your-branch>
```

---

## Technology Stack

- Python
- Poetry
- ChromaDB
- Sentence Transformers
- Hugging Face Transformers
- Pandas
- NumPy

---

## License

This project is licensed under the MIT License.