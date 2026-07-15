# API Specification — ReguAZ

> REST API for the ReguAZ regulatory intelligence platform. Version 1.0.0.

All endpoints are served by a local **FastAPI** application running on `http://localhost:8000`.

---

## Table of Contents

1. [General Conventions](#1-general-conventions)
2. [Authentication](#2-authentication)
3. [Error Format](#3-error-format)
4. [Endpoints](#4-endpoints)
   - [GET /](#get-)
   - [GET /health](#get-health)
   - [POST /chat](#post-chat)
   - [GET /documents](#get-documents)
   - [GET /documents/{document_id}](#get-documentsdocument_id)
   - [GET /documents/{document_id}/page/{page_number}](#get-documentsdocument_idpagepage_number)
   - [GET /documents/highlight](#get-documentshighlight)
5. [Schema Reference](#5-schema-reference)
6. [Planned Backend Endpoints](#6-planned-backend-endpoints)

---

## 1. General Conventions

| Aspect | Detail |
|---|---|
| **Base URL** | `http://localhost:8000` |
| **Protocol** | HTTP/1.1 |
| **Content-Type** | `application/json` for all request and response bodies |
| **Encoding** | UTF-8 |
| **CORS** | Allowed origins: `localhost:5173`, `localhost:3000`, `127.0.0.1:5173`, `127.0.0.1:3000` |
| **CORS methods** | `GET`, `POST`, `OPTIONS` |

---

## 2. Authentication

**Current implementation**: No authentication is required at the REST API layer. The current deployment model is completely local/on-premise, keeping data resident within the host organization. Session tracking on `/chat` uses a client-side generated session identifier to manage active conversations in the frontend state.

---

## 3. Error Format

Error responses return the following payload structure:

```json
{
  "success": false,
  "error": {
    "code": "string",
    "message": "string"
  }
}
```

### Error Codes

| HTTP Status | Code | Trigger |
|---|---|---|
| `400` | `bad_request` | Malformed request |
| `404` | `not_found` | Resource not found |
| `422` | `validation_error` | Input validation failure (concatenated messages) |
| `500` | `internal_server_error` | Unexpected internal exception |
| `503` | `service_unavailable` | Component loading failure on startup |

---

## 4. Endpoints

---

### `GET /`

**Summary**: Root status check.

Returns application online status.

#### Response `200 OK`

```json
{
  "name": "ReguAZ API",
  "version": "1.0.0",
  "status": "online"
}
```

---

### `GET /health`

**Summary**: Health check and dependency status.

Returns the initialization status of all core singletons.

#### Response `200 OK`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "pipeline_loaded": true,
  "document_service_loaded": true,
  "llm_loaded": true,
  "chunk_lookup_size": 4521,
  "document_index_size": 96
}
```

---

### `POST /chat`

**Summary**: Submit query to the RAG pipeline.

Executes the query synchronously through the retrieval, reranking, and generation pipeline.

#### Request Body

```json
{
  "question": "Bankın minimum nizamnamə kapitalı nə qədərdir?",
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | `string` | Yes | Natural language query in Azerbaijani. |
| `session_id` | `string` | No | Client conversation session token. |

#### Response `200 OK`

```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "question": "Bankın minimum nizamnamə kapitalı nə qədərdir?",
  "answer": "Bankın minimum nizamnamə kapitalı 50 milyon manat təşkil edir [1]. Xarici bank filialları üçün isə bu tələb 10 milyon manatdır [2].",
  "sources": [
    {
      "citation": 1,
      "chunk_id": "prudential_regulation_001_ch_1_art_5_3",
      "document_id": "prudential_regulation_001",
      "document_name": "Bankların nizamnamə kapitalına dair Qaydalar",
      "category": "prudential_regulations",
      "chapter": "Fəsil 1",
      "article": "Maddə 5",
      "page": 3,
      "chunk_preview": "5.1. Azərbaycan Respublikasında fəaliyyət göstərən bankların minimum nizamnamə kapitalı 50 milyon manat...",
      "rerank_score": 8.421,
      "rrf_score": 0.0317,
      "semantic_rank": 1,
      "bm25_rank": 2
    },
    {
      "citation": 2,
      "chunk_id": "prudential_regulation_001_ch_2_art_8_7",
      "document_id": "prudential_regulation_001",
      "document_name": "Bankların nizamnamə kapitalına dair Qaydalar",
      "category": "prudential_regulations",
      "chapter": "Fəsil 2",
      "article": "Maddə 8",
      "page": 7,
      "chunk_preview": "8.3. Azərbaycan Respublikasında fəaliyyət göstərən xarici bank filiallarının minimum...",
      "rerank_score": 6.843,
      "rrf_score": 0.0298,
      "semantic_rank": 3,
      "bm25_rank": 1
    }
  ],
  "metrics": {
    "retrieval_time": 3.241,
    "generation_time": 12.847,
    "total_time": 16.092
  }
}
```

---

### `GET /documents`

**Summary**: List all indexed regulatory documents.

Returns metadata cards for all active documents in the platform index.

#### Response `200 OK`

```json
[
  {
    "document_id": "prudential_regulation_001",
    "title": "Bankların nizamnamə kapitalına dair Qaydalar",
    "category": "prudential_regulations",
    "total_pages": 24,
    "total_chunks": 47,
    "language": "az",
    "parser": "pdfplumber",
    "publication_date": "2024-01-15T00:00:00",
    "status": "active",
    "related_articles": ["Maddə 1", "Maddə 2", "Maddə 5", "Maddə 8"],
    "document_metadata": {}
  }
]
```

---

### `GET /documents/{document_id}`

**Summary**: Get document metadata catalog card.

#### Response `200 OK`

Returns a single document metadata card corresponding to the given ID.

---

### `GET /documents/{document_id}/page/{page_number}`

**Summary**: Get document page text and sections.

Extracts text matching the specific 1-based page index from the cleaned markdown.

#### Response `200 OK`

```json
{
  "document_id": "prudential_regulation_001",
  "page_number": 3,
  "page_content": "Maddə 5. Bankın minimum nizamnamə kapitalı\n5.1. Azərbaycan Respublikasında ...",
  "article_information": [
    {
      "chapter": "Fəsil 1",
      "article": "Maddə 5",
      "section": null,
      "chunk_id": "prudential_regulation_001_ch_1_art_5_3"
    }
  ],
  "metadata": {
    "source_file": "prudential_regulation_001.pdf"
  }
}
```

---

### `GET /documents/highlight`

**Summary**: Get highlight coordinates for a cited chunk in the viewer.

#### Query Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `document_id` | `string` | Yes | Parent document identifier |
| `chunk_id` | `string` | Yes | The chunk identifier to locate |

#### Response `200 OK`

```json
{
  "document_id": "prudential_regulation_001",
  "page": 3,
  "article": "Maddə 5",
  "chunk_id": "prudential_regulation_001_ch_1_art_5_3",
  "chunk_start": null,
  "chunk_end": null,
  "highlighted_text": "5.1. Azərbaycan Respublikasında fəaliyyət göstərən bankların minimum nizamnamə kapitalı 50 milyon manat...",
  "offset_status": "future_enhancement"
}
```

---

## 5. Schema Reference

Schemas are implemented as standard FastAPI Pydantic models (referenced in `backend/app/schemas/`).

### `ChatRequest`
```json
{
  "question": "string (1-2000 characters, required)",
  "session_id": "string | null (optional)"
}
```

### `ChatResponse`
```json
{
  "session_id": "string",
  "question": "string",
  "answer": "string",
  "sources": "SourceDocument[]",
  "metrics": "MetricsResponse"
}
```

### `SourceDocument`
```json
{
  "citation": "integer",
  "chunk_id": "string",
  "document_id": "string | null",
  "document_name": "string",
  "category": "string",
  "chapter": "string | null",
  "article": "string | null",
  "page": "integer | null",
  "chunk_preview": "string",
  "rerank_score": "float | null",
  "rrf_score": "float | null",
  "semantic_rank": "integer | null",
  "bm25_rank": "integer | null"
}
```

### `MetricsResponse`
```json
{
  "retrieval_time": "float",
  "generation_time": "float",
  "total_time": "float"
}
```

---

## 6. Planned Backend Endpoints

The following features represent planned backend extensions:

- `POST /chat/stream`: Server-Sent Events (SSE) stream endpoint emitting tokens, source structures, and timing payloads reactively.
- `/history` routes (`GET`, `POST`, `DELETE`): Backend endpoints for persisting and purging user session history (currently tracked in frontend state).
