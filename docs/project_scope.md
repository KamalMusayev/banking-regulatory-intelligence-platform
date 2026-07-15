# Project Scope — ReguAZ

> AI-powered regulatory intelligence platform for Azerbaijani banking compliance.

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Current Scope](#2-current-scope)
3. [Out of Scope](#3-out-of-scope)
4. [Future Scope](#4-future-scope)
5. [Target Users](#5-target-users)
6. [Supported Documents](#6-supported-documents)
7. [Supported Languages](#7-supported-languages)
8. [Current Implementation Status](#8-current-implementation-status)
9. [Deployment Model](#9-deployment-model)

---

## 1. Project Purpose

Azerbaijani banking regulation is fragmented across dozens of laws, Central Bank rules, and methodological guidance documents — most of which exist only as long, regulatory PDFs. Manually locating a specific requirement — a capital adequacy threshold, an AML reporting timeline, a governance obligation — is slow and error-prone.

**ReguAZ** makes this publicly available Central Bank of Azerbaijan (CBA) regulatory corpus queryable in natural language. A compliance professional submits a question in Azerbaijani; the system retrieves the most relevant passages from the indexed regulatory text using a hybrid retrieval pipeline, then generates a concise, citation-grounded answer using a local language model — explicitly refusing to answer when retrieved context is insufficient.

The system is built from the ground up as a production-ready enterprise solution. Every stage of the pipeline — extraction, chunking, embedding, indexing, retrieval, reranking, and local generation — is fully implemented, verified, and integrated.

---

## 2. Current Scope

ReguAZ operates exclusively on **publicly available regulatory documents published by the Central Bank of Azerbaijan (CBA)**. This includes:

- **96 regulatory PDF documents** organized across 8 categories (laws, AML/KYC guidelines, prudential regulations, etc.).
- **Natural language querying in Azerbaijani** against this specific document set.
- **Grounded answer generation** with inline citations mapping directly to source page numbers, chapters, and articles.
- **Interactive document exploration** and page-by-page rendering in the React frontend.
- **Fully local execution** to preserve absolute privacy of organizational compliance queries.

---

## 3. Out of Scope

The following features and document domains are **intentionally excluded** from the current implementation of the platform:

- **Commercial bank internal policies**: The platform does not index or query private internal rules, standard operating procedures (SOPs), or internal audits of commercial banks.
- **Other financial sectors**: Insurance company policies, fintech product guidelines, and private payment provider rules are not supported in the active database.
- **Non-CBA public regulatory documents**: Decrees or regulations from other state agencies (e.g., Ministry of Finance, tax authorities) are excluded unless explicitly republished as CBA banking standards.
- **Public cloud infrastructure or SaaS hosting**: The system is not offered as a public SaaS or cloud service.

---

## 4. Future Scope

The architecture of ReguAZ is built with modular abstraction layers (e.g., `BaseLLMProvider`, `BaseEmbeddingService`, database abstractions) so that similar regulatory intelligence platforms can later be built for other regulated organizations without major architectural changes:

- **Commercial Banks**: Adapting the pipeline to ingest and query internal policy libraries, product guidelines, and risk manuals.
- **Insurance Companies**: Deploying dedicated RAG systems for insurance regulations, underwriting rules, and claim guidelines.
- **Payment Providers & Fintechs**: Configuring instances for card scheme rules, payment system regulations, and fintech-specific compliance frameworks.
- **Other Regulated Organizations**: Applying the ingestion and retrieval workflows to utility, energy, or environmental regulatory intelligence.

Each organization would deploy its own isolated instance of the platform containing its specific document corpus, dedicated vector database collections, and custom gold evaluation datasets to ensure absolute data separation.

---

## 5. Target Users

The primary users ReguAZ is built for:

- **Compliance officers** at Azerbaijani banks who need fast, traceable answers to specific regulatory questions — capital thresholds, AML reporting deadlines, governance requirements — without reading through dozens of PDFs.
- **Risk management teams** who need to cross-reference multiple regulatory documents to validate internal policies against CBA requirements.
- **Internal auditors** who need to verify that bank procedures match published CBA rules and need exact source citations to support audit findings.
- **Legal staff** advising on regulatory interpretation who need rapid access to the exact regulatory text governing a situation.

---

## 6. Supported Documents

All 96 source documents are publicly available PDF publications of the Central Bank of Azerbaijan, organized into eight regulatory categories:

| Category | Slug | Description |
|---|---|---|
| Laws | `laws` | Primary Azerbaijani legislation governing banking |
| AML/KYC | `aml_kyc` | Anti-money laundering and Know Your Customer rules |
| Governance & Compliance | `governance_and_compliance` | Corporate governance and internal control frameworks |
| Guidance & Methodology | `guidance_and_methodology` | Methodological guidelines and instructions |
| Payments & Banking Operations | `payments_and_banking_operations` | Payment system and operational banking rules |
| Prudential Regulations | `prudential_regulations` | Capital adequacy, liquidity, and prudential standards |
| Reporting & Audit | `reporting_and_audit` | Reporting formats, submission requirements, and audit instructions |
| Risk Management | `risk_management` | Risk management frameworks and requirements |

---

## 7. Supported Languages

| Language | Support Level |
|---|---|
| Azerbaijani | **Primary** — LLM answer output, system prompt, and the majority of the regulatory corpus |
| Russian | **Corpus only** — some CBA documents include Russian text; retrieval works; LLM responds in Azerbaijani |
| English | **Corpus only** — queries in English are processed; retrieval works; LLM responds in Azerbaijani |

The LLM is explicitly instructed via the system prompt to output answers in Azerbaijani regardless of the query language.

---

## 8. Current Implementation Status

All core components of the platform are fully implemented:

- **Document Ingestion**: Parsing via `pdfplumber`, page marker injection (`<!-- PAGE: N -->`), and text cleaning.
- **Metadata Generation**: Automatically extracts page counts, categories, and creation timestamps.
- **Intelligent Chunking**: Chapter- and article-aware sliding-window chunking (4,000-char window, 500-char overlap) with stable, human-readable identifiers.
- **Embedding Pipeline**: Common interface with multiple model support (`BAAI/bge-m3`, `intfloat/multilingual-e5-large`, `jinaai/jina-embeddings-v3`, and `Qwen/Qwen3-Embedding-0.6B`). BGE-M3 is configured as the production model.
- **Vector Storage**: Persisted local Qdrant collection for production, with ChromaDB integrated for development and experimentation.
- **Hybrid Retrieval**: Unified pipeline combining dense Qdrant search and sparse BM25 keyword search via Reciprocal Rank Fusion (RRF).
- **Reranking**: Joint-relevance scoring using a Cross-Encoder model (`bge-reranker-v2-m3`).
- **Retrieval Evaluation**: Automated evaluation framework computing Recall@K, Precision@K, MRR, and nDCG against a 121-question gold dataset.
- **Prompt Builder**: Centralized prompt constructor enforcing Azerbaijani-only output, strict context-only answers, and specific inline bracket citations.
- **LLM Provider Abstraction**: Configurable provider architecture supporting local, remote, and mock LLM backends.
- **Local LLM Inference**: Gemma 3 4B running fully locally through a compiled `llama.cpp` backend.
- **End-to-End RAG Pipeline**: Connected FastAPI backend and React frontend providing a synchronous query-to-answer loop.
- **Conversation Memory**: Session-based conversation history maintained throughout an active chat session.
- **Docker Containerization**: All backend services and support databases (Qdrant) are containerized.

---

## 9. Deployment Model

ReguAZ is designed as a B2B platform deployed on-premise to keep all data and queries strictly within the boundaries of the organization:

- **No Cloud Dependencies**: The core platform runs completely offline. Gemma 3 4B is loaded into memory via `llama.cpp` using local CPU/GPU resources (MPS on Apple Silicon, CUDA on NVIDIA, or standard CPU).
- **Dockerized Architecture**: Backend and supporting services are run locally using Docker for containerized deployment, ensuring consistency across environments.
- **Data Governance**: Regulatory documents and user query logs are persisted locally. No external APIs, third-party LLM endpoints, or public cloud infrastructures are used during inference.
