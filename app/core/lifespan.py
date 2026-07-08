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

from app.core.config import get_settings

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
    # ------------------------------------------------------------------
    logger.info("[1/5] Loading Chunk Lookup...")
    try:
        from pathlib import Path
        from backend.reguaz.services.chunks.chunk_reader import ChunkReader
        chunk_reader = ChunkReader()
        state.chunk_lookup = chunk_reader.build_lookup(settings.CHUNKS_DIR)
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
        from app.services.document_service import DocumentService
        cleaned_docs_dir = Path(settings.METADATA_DIR).parent / "cleaned_documents"
        state.document_service = DocumentService(
            metadata_dir=settings.METADATA_DIR,
            cleaned_docs_dir=cleaned_docs_dir,
            chunk_lookup=state.chunk_lookup,
        )
        # Populate the document index with metadata dicts
        state.document_index = list(state.document_service._doc_metadata_map.values())
        logger.info("[2/5] Loaded %d documents into service.", len(state.document_index))
    except Exception as exc:
        logger.error("[2/5] Failed to initialize DocumentService: %s", exc)
        raise

    # ------------------------------------------------------------------
    # Step 3 — Retriever
    # Initialises HybridQdrantRetriever:
    #   - Loads BGE-M3 SentenceTransformer
    #   - Opens local Qdrant collection
    #   - Builds BM25Okapi index from all chunk texts
    #   - Loads BAAI/bge-reranker-v2-m3 CrossEncoder
    #
    # Phase 2 implementation:
    #   from backend.reguaz.retrieval.hybrid_qdrant import HybridQdrantRetriever
    #   state.retriever = HybridQdrantRetriever(
    #       model_name=settings.EMBEDDING_MODEL,
    #       qdrant_dir=settings.QDRANT_DIR,
    #       chunks_dir=settings.CHUNKS_DIR,
    #       top_k_semantic=settings.TOP_K_SEMANTIC,
    #       top_k_bm25=settings.TOP_K_BM25,
    #       rerank_top_k=settings.RERANK_TOP_K,
    #       final_top_k=settings.DEFAULT_TOP_K,
    #       rrf_k=settings.RRF_K,
    #       reranker_model=settings.RERANKER_MODEL,
    #   )
    # ------------------------------------------------------------------
    logger.info("[3/5] HybridQdrantRetriever — pending (Phase 2)")

    # ------------------------------------------------------------------
    # Step 4 — LLM
    # Loads the Gemma GGUF model into memory via llama.cpp.
    #
    # Phase 2 implementation:
    #   from backend.reguaz.services.generation.llm_factory import LLMFactory
    #   state.llm = LLMFactory.create(model_type=settings.LLM_TYPE)
    # ------------------------------------------------------------------
    logger.info("[4/5] LLM (GemmaService)    — pending (Phase 2)")

    # ------------------------------------------------------------------
    # Step 5 — GenerationPipeline
    # Wires retriever + chunk_lookup + llm into a single orchestrator.
    #
    # Phase 2 implementation:
    #   from backend.reguaz.services.generation.generation_pipeline import GenerationPipeline
    #   state.generation_pipeline = GenerationPipeline(
    #       retriever=state.retriever,
    #       chunk_lookup=state.chunk_lookup,
    #       llm=state.llm,
    #       top_k=settings.DEFAULT_TOP_K,
    #   )
    #   state.is_ready = True
    # ------------------------------------------------------------------
    logger.info("[5/5] GenerationPipeline    — pending (Phase 2)")

    state.startup_time_s = time.perf_counter() - t_start
    logger.info("Startup complete in %.3f s (skeleton mode — no models loaded)", state.startup_time_s)

    # Attach to app.state so dependency functions can read it
    app.state.reguaz = state

    # ── REQUEST SERVING ───────────────────────────────────────────────────────
    yield

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ReguAZ API — Shutting down")

    # Phase 2: release llama.cpp model memory
    # if state.llm is not None and hasattr(state.llm, "close"):
    #     state.llm.close()

    state.is_ready = False
    logger.info("ReguAZ API — Shutdown complete")
    logger.info("=" * 60)
