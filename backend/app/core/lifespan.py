"""
app/core/lifespan.py

FastAPI application lifespan context manager.

All heavy singletons (embedding model, Qdrant index, BM25 corpus, cross-encoder
reranker, llama.cpp model, and the GenerationPipeline) are initialised ONCE at
startup and stored on ``app.state.reguaz`` (an ``AppState`` instance).

Route handlers and dependency functions never construct these objects — they
only access the pre-loaded instances through the dependency injection layer
defined in ``app/core/dependencies.py``.

Startup phases
--------------
Phase 1 (current) — scaffold only.
    All slots in AppState are set to None / empty.
    The health endpoint reports ``"status": "starting"``.

Phase 2 (next milestone) — full pipeline initialisation.
    Each ``# TODO`` block below will be replaced with the real constructor call.
    The frozen RAG modules in ``backend/reguaz/`` are imported and wired here
    and NOWHERE ELSE in the API layer.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

# pyrefly: ignore [missing-import]
from fastapi import FastAPI

from backend.app.core.config import get_settings
from backend.reguaz import config as reguaz_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed application state container
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    """
    Typed container for every shared singleton in the application.

    Stored as ``app.state.reguaz`` so that dependency functions can access
    objects through ``request.app.state.reguaz`` without using any module-level
    globals.

    All fields default to ``None`` / empty so the app can start (and the health
    endpoint can answer) even before the heavy models are loaded.
    """

    # Populated in Phase 2 ────────────────────────────────────────────────────

    # dict[chunk_id: str, chunk_metadata: dict]
    # Built by ChunkReader.build_lookup() from backend.reguaz.services.chunks
    chunk_lookup: dict[str, dict[str, Any]] = field(default_factory=dict)

    # HybridQdrantRetriever instance
    # (holds BGE-M3 embedder + Qdrant client + BM25 index + CrossEncoder reranker)
    retriever: Any | None = None

    # BaseLLM instance (currently GemmaService backed by llama.cpp)
    llm: Any | None = None

    # GenerationPipeline instance — wires retriever + chunk_lookup + llm
    generation_pipeline: Any | None = None

    # list[DocumentItem-like dicts] — built from data/processed/metadata/ on startup
    document_index: list[dict[str, Any]] = field(default_factory=list)

    # DocumentService instance wrapping catalog and page extraction logic
    document_service: Any | None = None

    # ── Readiness ─────────────────────────────────────────────────────────────

    # Set to True only after all components have been successfully loaded.
    is_ready: bool = False

    # Recorded at the end of startup; used by the health endpoint.
    startup_time_s: float | None = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage the application lifecycle.

    FastAPI calls this function once at startup (before accepting requests)
    and once at shutdown (after the last request completes).

    The ``yield`` separates startup from shutdown logic.
    """
    settings = get_settings()
    t_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("ReguAZ API v%s — Starting up (%s)", settings.APP_VERSION, settings.APP_ENV)
    logger.info("=" * 60)

    state = AppState()

    # ── STARTUP ───────────────────────────────────────────────────────────────

    # ------------------------------------------------------------------
    # Step 1 — Chunk lookup
    # Reads every .jsonl file under CHUNKS_DIR into a dict[chunk_id → metadata].
    # Used by the GenerationPipeline and by the GET /chunks/{chunk_id} endpoint.
    # Paths are resolved to absolute using PROJECT_ROOT from reguaz.config so
    # startup works regardless of the cwd uvicorn is launched from.
    # ------------------------------------------------------------------
    logger.info("[1/5] Loading Chunk Lookup...")
    try:
        from pathlib import Path
        from backend.reguaz.services.chunks.chunk_reader import ChunkReader

        chunks_abs = reguaz_config.PROJECT_ROOT / settings.CHUNKS_DIR
        chunk_reader = ChunkReader()
        state.chunk_lookup = chunk_reader.build_lookup(chunks_abs)
        logger.info("[1/5] Loaded %d chunks into lookup.", len(state.chunk_lookup))
    except Exception as exc:
        logger.error("[1/5] Failed to load chunk lookup: %s", exc)
        raise

    # ------------------------------------------------------------------
    # Step 2 — Document index and DocumentService
    # Scans METADATA_DIR and loads metadata and cleaned page mappings.
    # ------------------------------------------------------------------
    logger.info("[2/5] Initializing DocumentService...")
    try:
        from backend.app.services.document_service import DocumentService

        metadata_abs = reguaz_config.PROJECT_ROOT / settings.METADATA_DIR
        cleaned_docs_abs = metadata_abs.parent / "cleaned_documents"
        state.document_service = DocumentService(
            metadata_dir=metadata_abs,
            cleaned_docs_dir=cleaned_docs_abs,
            chunk_lookup=state.chunk_lookup,
        )
        # Populate the document index with metadata dicts
        state.document_index = list(state.document_service._doc_metadata_map.values())
        logger.info("[2/5] Loaded %d documents into service.", len(state.document_index))
    except Exception as exc:
        logger.error("[2/5] Failed to initialize DocumentService: %s", exc)
        raise

    # ------------------------------------------------------------------
    # Step 3 — HybridQdrantRetriever
    # Loads BGE-M3 SentenceTransformer, opens the Qdrant collection,
    # builds the BM25Okapi corpus from all chunk texts, and loads the
    # BAAI/bge-reranker-v2-m3 CrossEncoder — all in the constructor.
    #
    # Paths are passed as strings; the retriever resolves them internally.
    # We derive absolute paths from reguaz.config.PROJECT_ROOT so this
    # works regardless of the cwd uvicorn is launched from.
    # ------------------------------------------------------------------
    logger.info("[3/5] Initializing HybridQdrantRetriever (this may take 30–60 s)...")
    try:
        from backend.reguaz.retrieval.hybrid_qdrant import HybridQdrantRetriever

        qdrant_abs = str(reguaz_config.PROJECT_ROOT / settings.QDRANT_DIR)
        chunks_abs = str(reguaz_config.PROJECT_ROOT / settings.CHUNKS_DIR)

        state.retriever = HybridQdrantRetriever(
            model_name=settings.EMBEDDING_MODEL,
            qdrant_dir=qdrant_abs,
            chunks_dir=chunks_abs,
            top_k_semantic=settings.TOP_K_SEMANTIC,
            top_k_bm25=settings.TOP_K_BM25,
            rerank_top_k=settings.RERANK_TOP_K,
            final_top_k=settings.DEFAULT_TOP_K,
            rrf_k=settings.RRF_K,
            reranker_model=settings.RERANKER_MODEL,
        )
        logger.info("[3/5] HybridQdrantRetriever ready.")
    except Exception as exc:
        logger.error("[3/5] Failed to initialize HybridQdrantRetriever: %s", exc)
        raise

    # ------------------------------------------------------------------
    # Step 4 — LLM via LLMFactory
    # LLMFactory.create() pulls model_path and generation parameters
    # from backend.reguaz.config automatically — no path needed here.
    # ------------------------------------------------------------------
    logger.info("[4/5] Loading LLM via LLMFactory (model_type='%s')...", settings.LLM_TYPE)
    try:
        from backend.reguaz.services.generation.llm_factory import LLMFactory

        state.llm = LLMFactory.create(model_type=settings.LLM_TYPE)
        logger.info("[4/5] LLM loaded.")
    except Exception as exc:
        logger.error("[4/5] Failed to load LLM: %s", exc)
        raise

    # ------------------------------------------------------------------
    # Step 5 — GenerationPipeline
    # Wires the retriever, chunk_lookup, and LLM into the single
    # orchestrator that every /chat request calls.
    # ------------------------------------------------------------------
    logger.info("[5/5] Wiring GenerationPipeline...")
    try:
        from backend.reguaz.services.generation.generation_pipeline import GenerationPipeline

        state.generation_pipeline = GenerationPipeline(
            retriever=state.retriever,
            chunk_lookup=state.chunk_lookup,
            llm=state.llm,
            top_k=settings.DEFAULT_TOP_K,
        )
        state.is_ready = True
        logger.info("[5/5] GenerationPipeline ready.")
    except Exception as exc:
        logger.error("[5/5] Failed to wire GenerationPipeline: %s", exc)
        raise

    state.startup_time_s = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("ReguAZ API — Startup complete in %.1f s", state.startup_time_s)
    logger.info("=" * 60)

    # Attach to app.state so dependency functions can read it
    app.state.reguaz = state

    # ── REQUEST SERVING ───────────────────────────────────────────────────────
    yield

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ReguAZ API — Shutting down")

    # Release llama.cpp model memory if the implementation supports it
    if state.llm is not None and hasattr(state.llm, "close"):
        try:
            state.llm.close()
        except Exception:
            pass

    state.is_ready = False
    logger.info("ReguAZ API — Shutdown complete")
    logger.info("=" * 60)
