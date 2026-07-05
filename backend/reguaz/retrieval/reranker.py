"""
backend/reguaz/retrieval/reranker.py

Cross-Encoder reranker wrapper.
Performs semantic reranking of candidate documents using a Cross-Encoder model.
"""

from __future__ import annotations

import logging
import time

# pyrefly: ignore [missing-import]
import torch

# pyrefly: ignore [missing-import]
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks a list of candidate documents relative to a query using a Cross-Encoder.

    This class loads a sentence-transformers Cross-Encoder model (defaulting to
    ``"BAAI/bge-reranker-v2-m3"``) and computes scores for (query, document) pairs.

    Parameters
    ----------
    model_name : str
        Name/path of the Cross-Encoder model to load (e.g. ``"BAAI/bge-reranker-v2-m3"``).
    device : str | None
        Target device for inference (e.g. ``"cuda"``, ``"cpu"``). If None, defaults
        to auto-detection.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
    ) -> None:
        logger.info(
            "CrossEncoderReranker: loading Cross-Encoder model '%s' on device %s ...",
            model_name,
            device or "auto",
        )

        try:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(
                "CrossEncoderReranker: loading Cross-Encoder model '%s' on device '%s' ...",
                model_name,
                device,
            )

            self._model = CrossEncoder(
                model_name,
                device=device,
                max_length=1024,
            )

            logger.info(
                "CrossEncoderReranker: actual model device = %s",
                self._model.model.device,
            )

            logger.info(
                "CrossEncoderReranker: model '%s' loaded successfully.",
                model_name,
            )

        except Exception as exc:
            logger.error(
                "CrossEncoderReranker: failed to load model '%s': %s",
                model_name,
                exc,
            )
            raise

    def rerank(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """
        Compute similarity scores for a query and a list of candidate documents.

        Parameters
        ----------
        query : str
            The query text.
        documents : list[str]
            A list of document texts to rerank.

        Returns
        -------
        list[float]
            List of similarity scores (higher = more relevant) corresponding to
            the input documents in the same order.
        """
        if not documents:
            return []

        logger.info(
            "CrossEncoderReranker: reranking %d documents.",
            len(documents),
        )

        # Prepare (query, document) pairs for the cross-encoder.
        pairs = [[query, doc] for doc in documents]

        start = time.perf_counter()

        # Compute raw scores.
        raw_scores = self._model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=True,
        )

        elapsed = time.perf_counter() - start

        logger.info(
            "CrossEncoderReranker: predict() finished in %.3f s.",
            elapsed,
        )

        # Ensure we return Python float type.
        if isinstance(raw_scores, list):
            return [float(score) for score in raw_scores]

        return [float(score) for score in raw_scores.tolist()]