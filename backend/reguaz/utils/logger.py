import logging
from pathlib import Path


def setup_logger(model_name: str) -> logging.Logger:
    """
    Creates a logger that writes both to console and a model-specific log file.

    Example:
        logs/embedding_bge_m3.log
        logs/embedding_e5.log
        logs/embedding_jina_v3.log
    """

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"embedding_{model_name}.log"

    logger = logging.getLogger(f"embedding_{model_name}")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # Console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File
    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
        mode="a"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger