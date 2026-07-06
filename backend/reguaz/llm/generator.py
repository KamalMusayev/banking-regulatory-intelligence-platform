"""
backend/reguaz/llm/generator.py

Orchestration layer for the LLM pipeline.

``Generator`` is responsible **only** for wiring the pipeline stages
together.  It does not contain prompt-construction logic or inference
logic.  Its sole responsibilities are:

1. Receive a question and retrieved context.
2. Delegate prompt construction to ``PromptBuilder``.
3. Obtain a provider via ``LLMProviderFactory``.
4. Call ``provider.generate(prompt)``.
5. Return the final answer.

Generator does not know which model is used, which inference engine is
used, whether inference is local or remote, or how deployment is
performed.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from backend.reguaz.llm.base import BaseLLMProvider
from backend.reguaz.llm.factory import LLMProviderFactory
from backend.reguaz.llm.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class Generator:
    """
    Orchestrates the Prompt → LLM → Answer pipeline.

    This class is the single entry point for obtaining an LLM-generated
    answer from a question and its retrieved context.  All implementation
    details are delegated to the ``PromptBuilder`` and the
    ``BaseLLMProvider`` returned by the factory.

    Parameters
    ----------
    provider_type : str
        Provider type identifier forwarded to
        ``LLMProviderFactory.get_provider()``.  Default: ``"local"``.
    **provider_kwargs : Any
        Additional keyword arguments forwarded to the provider
        constructor (e.g. ``model_name``, ``device``,
        ``max_new_tokens``).
    """

    def __init__(
        self,
        provider_type: str = "local",
        **provider_kwargs: Any,
    ) -> None:
        logger.info(
            "Generator: initialising (provider_type='%s').",
            provider_type,
        )

        self._prompt_builder = PromptBuilder()
        self._provider: BaseLLMProvider = LLMProviderFactory.get_provider(
            provider_type=provider_type,
            **provider_kwargs,
        )

        logger.info(
            "Generator: ready (provider_type='%s', model='%s').",
            provider_type,
            self._provider.model_name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """
        Return the model name of the underlying provider.

        Returns
        -------
        str
            Model identifier.
        """
        return self._provider.model_name

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate an answer for a question given retrieved context.

        Pipeline:
            question + context → PromptBuilder → prompt → Provider → answer

        Parameters
        ----------
        question : str
            The end-user's question in natural language.
        context : str
            The retrieved document context assembled by the retrieval
            pipeline.

        Returns
        -------
        str
            The LLM-generated answer text.
        """
        logger.info(
            "Generator.generate_answer: processing question "
            "(question length=%d chars, context length=%d chars).",
            len(question),
            len(context),
        )
        t0 = time.perf_counter()

        # Step 1 — Build the fully formatted prompt.
        prompt = self._prompt_builder.build_prompt(
            question=question,
            context=context,
        )

        # Step 2 — Delegate to the provider.
        answer = self._provider.generate(prompt)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Generator.generate_answer: completed in %.3f s "
            "(answer length=%d chars).",
            elapsed,
            len(answer),
        )

        return answer
