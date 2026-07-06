"""
backend/reguaz/llm/base.py

Abstract base class for all LLM providers.

Every provider — whether performing inference locally, over HTTP, via RPC,
through llama.cpp, Ollama, vLLM, or any future backend — must implement
this interface.  The interface is deliberately transport-agnostic: it makes
no assumption about *how* or *where* inference is executed.

The provider receives a fully formatted prompt (constructed by
``PromptBuilder``) and returns the generated answer text.  It has zero
knowledge of questions, retrieved context, or prompt structure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM inference providers.

    Subclasses must implement:

    * ``generate`` — perform inference on a fully formatted prompt.
    * ``model_name`` — expose the underlying model identifier for logging
      and diagnostics.

    The interface is transport-agnostic.  Concrete implementations may
    perform inference in-process, call an HTTP server, communicate via
    RPC, or use any other mechanism.  The rest of the application never
    needs to know.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from a fully formatted prompt.

        Parameters
        ----------
        prompt : str
            A complete, ready-to-use prompt string.  The provider must
            never modify the prompt structure — it is the sole product
            of ``PromptBuilder``.

        Returns
        -------
        str
            The generated answer text.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return the identifier of the underlying model.

        Returns
        -------
        str
            Model name or path (e.g. ``"Qwen/Qwen2.5-3B-Instruct"``).
        """
