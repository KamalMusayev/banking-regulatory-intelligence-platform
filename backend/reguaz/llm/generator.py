"""
backend/reguaz/llm/generator.py

Orchestration layer for the LLM pipeline.

``Generator`` is responsible **only** for wiring the pipeline stages
together.  It does not contain prompt-construction logic or inference
logic.  Its sole responsibilities are:

1. Receive a question and retrieved context.
2. Delegate prompt construction to ``PromptBuilder``.
3. Call ``provider.generate(prompt)``.
4. Return the final answer.

Generator does not know which model is used, which inference engine is
used, whether inference is local or remote, or how deployment is
performed.  Provider creation is the caller's responsibility — Generator
receives an already-constructed ``BaseLLMProvider`` through its
constructor (dependency injection).
"""

from __future__ import annotations

import logging
import time

from backend.reguaz.llm.base import BaseLLMProvider
from backend.reguaz.llm.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class Generator:
    """
    Orchestrates the Prompt → LLM → Answer pipeline.

    This class is the single entry point for obtaining an LLM-generated
    answer from a question and its retrieved context.  All implementation
    details are delegated to the injected ``PromptBuilder`` and
    ``BaseLLMProvider``.

    The provider is constructed externally (via ``LLMProviderFactory`` or
    any other means) and injected at construction time.  Generator is
    therefore completely independent from the factory and from any
    inference technology.

    Parameters
    ----------
    provider : BaseLLMProvider
        A fully constructed LLM provider.  The provider is responsible
        for all inference; Generator never interacts with it beyond
        calling ``provider.generate(prompt)``.
    prompt_builder : PromptBuilder | None
        Prompt-construction component.  If ``None``, a default
        ``PromptBuilder`` instance is created automatically.

    Examples
    --------
    Typical usage::

        provider = SomeProvider(...)
        generator = Generator(provider)
        answer = generator.generate_answer(question, context)
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self._provider = provider
        self._prompt_builder = prompt_builder if prompt_builder is not None else PromptBuilder()

        logger.info(
            "Generator: initialised (model='%s').",
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

        # Step 2 — Delegate inference to the provider.
        answer = self._provider.generate(prompt)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Generator.generate_answer: completed in %.3f s "
            "(answer length=%d chars).",
            elapsed,
            len(answer),
        )

        return answer
