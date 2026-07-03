"""
Global configuration for the ReguAZ project.

All commonly used paths and default values are defined here.
"""

from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

PROCESSED_DIR = DATA_DIR / "processed"

RAW_DATA_DIR = DATA_DIR / "raw"

CHUNKS_DIR = PROCESSED_DIR / "chunks"

EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"

METADATA_DIR = PROCESSED_DIR / "metadata"

QDRANT_DIR = DATA_DIR / "qdrant"

LOGS_DIR = PROJECT_ROOT / "backend" / "reguaz" / "logs"

# Create required directories automatically
LOGS_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Qdrant
# =============================================================================

DEFAULT_COLLECTION_PREFIX = "reguaz"

DEFAULT_DISTANCE_METRIC = "Cosine"

# =============================================================================
# Embeddings
# =============================================================================

DEFAULT_BATCH_SIZE = 100

SUPPORTED_EMBEDDING_MODELS = (
    "bge_m3",
    "e5",
)

# =============================================================================
# Retrieval
# =============================================================================

DEFAULT_TOP_K = 10

BM25_TOP_K = 20

SEMANTIC_TOP_K = 20

RRF_K = 60

# =============================================================================
# Evaluation
# =============================================================================

DEFAULT_EVALUATION_TOP_K = 10

# =============================================================================
# Logging
# =============================================================================

DEFAULT_LOG_LEVEL = "INFO"