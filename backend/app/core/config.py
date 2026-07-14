"""
app/core/config.py

Centralised application settings using pydantic-settings.

All values are read from environment variables or a .env file located at the
project root.  Calling get_settings() returns a cached singleton so the .env
file is only parsed once per process.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide configuration.

    Fields are populated (in order of priority) from:
      1. Environment variables (highest priority)
      2. .env file at the project root
      3. Default values defined here (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    APP_NAME: str = "ReguAZ API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "AI-powered Azerbaijani banking regulatory intelligence platform. "
        "Hybrid retrieval (Qdrant + BM25) with cross-encoder reranking and "
        "local LLM generation via llama.cpp."
    )
    DEBUG: bool = False

    # ── API ───────────────────────────────────────────────────────────────────
    API_V1_PREFIX: str = "/api/v1"

    # ── CORS ──────────────────────────────────────────────────────────────────
    # In development the React Vite dev server typically runs on 5173.
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "OPTIONS"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    QDRANT_DIR: str = "qdrant_data"
    CHUNKS_DIR: str = "data/processed/chunks"
    METADATA_DIR: str = "data/processed/metadata"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "bge_m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    TOP_K_SEMANTIC: int = 20
    TOP_K_BM25: int = 20
    RERANK_TOP_K: int = 15
    DEFAULT_TOP_K: int = 5
    RRF_K: int = 60

    # ── LLM ───────────────────────────────────────────────────────────────────
    LLM_TYPE: str = "gemma"

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application settings singleton.

    The instance is constructed once and cached for the lifetime of the process.
    Use ``get_settings.cache_clear()`` in tests to reset the cache between runs.
    """
    return Settings()
