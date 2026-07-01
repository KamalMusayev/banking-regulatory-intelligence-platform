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