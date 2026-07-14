import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# =============================================================================
# Project Paths
# =============================================================================

# This resolves to the directory containing this file: backend/reguaz/config
_CURRENT_DIR = Path(__file__).resolve().parent

# The backend root is 2 levels up: backend/reguaz/config -> backend/reguaz -> backend
BACKEND_ROOT = _CURRENT_DIR.parents[1]

# Default local paths
_DEFAULT_DATA_DIR = BACKEND_ROOT.parent / "data"
if not _DEFAULT_DATA_DIR.exists():
    _DEFAULT_DATA_DIR = BACKEND_ROOT / "data"

# Default Qdrant path is inside backend folder now: backend/qdrant_data/
_DEFAULT_QDRANT_PATH = BACKEND_ROOT / "qdrant_data"
if not _DEFAULT_QDRANT_PATH.exists():
    # Fallback to repo data/qdrant if not created yet
    _DEFAULT_QDRANT_PATH = _DEFAULT_DATA_DIR / "qdrant"

_DEFAULT_LLM_MODEL_PATH = BACKEND_ROOT / "reguaz" / "models" / "gemma-4-E4B-it-Q4_K_M.gguf"
_DEFAULT_GOLD_DATASET_PATH = _DEFAULT_DATA_DIR / "evaluation" / "gold_dataset_for_llm_generation.xlsx"
_DEFAULT_RESULTS_PATH = BACKEND_ROOT / "results"
_DEFAULT_LOGS_PATH = BACKEND_ROOT / "logs"

# Use BACKEND_ROOT as PROJECT_ROOT to ensure standalone backend runs completely within the backend directory
PROJECT_ROOT = BACKEND_ROOT

# Environment overridden paths
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", str(_DEFAULT_QDRANT_PATH)))
LLM_MODEL_PATH = Path(os.getenv("LLM_MODEL_PATH", str(_DEFAULT_LLM_MODEL_PATH)))
GOLD_DATASET_PATH = Path(os.getenv("GOLD_DATASET_PATH", str(_DEFAULT_GOLD_DATASET_PATH)))
RESULTS_PATH = Path(os.getenv("RESULTS_PATH", str(_DEFAULT_RESULTS_PATH)))
LOGS_PATH = Path(os.getenv("LOGS_PATH", str(_DEFAULT_LOGS_PATH)))

# Sub-directories
DATA_DIR = _DEFAULT_DATA_DIR
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
METADATA_DIR = PROCESSED_DIR / "metadata"

# Create required directories automatically
LOGS_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
QDRANT_PATH.mkdir(parents=True, exist_ok=True)

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
BM25_TOP_K = 15
SEMANTIC_TOP_K = 15
RRF_K = 60

# =============================================================================
# Evaluation
# =============================================================================
DEFAULT_EVALUATION_TOP_K = 5

# =============================================================================
# Logging
# =============================================================================
DEFAULT_LOG_LEVEL = "INFO"

# =============================================================================
# LLM Generation
# =============================================================================
DEFAULT_LLM_TYPE = "gemma"

# Common LLM Parameters
LLM_CONTEXT_WINDOW = 8192
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 512
LLM_TOP_P = 0.95
LLM_TOP_K = 40
LLM_REPEAT_PENALTY = 1.1
LLM_SEED = 42
PROMPT_RESERVED_TOKENS = 300

# Model Paths
GEMMA_MODEL_PATH = BACKEND_ROOT / "reguaz" / "models" / "gemma-4-E4B-it-Q4_K_M.gguf"

