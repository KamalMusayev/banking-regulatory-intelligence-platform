"""
backend/reguaz/llm/factory.py

Factory for selecting and instantiating LLM providers.

``LLMProviderFactory`` is the **only** entry point for obtaining a
provider instance.  The ``Generator`` never references concrete provider
classes directly — it asks the factory for a ``BaseLLMProvider`` and
uses it through the abstract interface.

Today the factory supports a single provider type (``"local"``).
Future provider types (``"llamacpp"``, ``"remote"``,
``"openai_compatible"``, ``"custom"``) can be added by registering a
new branch without modifying any other component.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.reguaz.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class LLMProviderFactory:
    """
    Factory class to instantiate the appropriate LLM provider.

    The factory interface is intentionally stable: ``get_provider()``
    accepts a provider type string and forwards all remaining keyword
    arguments to the concrete provider constructor.  Adding a new
    provider never changes the factory's public signature.
    """

    @staticmethod
    def get_provider(
        provider_type: str = "local",
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """
        Instantiate and return the LLM provider for the given type.

        Parameters
        ----------
        provider_type : str
            Identifier selecting the provider implementation.
            Currently supported:

            * ``"local"`` — ``LocalInferenceProvider`` (default).

            Future examples: ``"llamacpp"``, ``"remote"``,
            ``"openai_compatible"``, ``"custom"``.
        **kwargs : Any
            Additional keyword arguments forwarded to the provider
            constructor (e.g. ``model_name``, ``device``,
            ``max_new_tokens``).

        Returns
        -------
        BaseLLMProvider
            A fully constructed provider instance.

        Raises
        ------
        ValueError
            If ``provider_type`` is not recognised.
        """
        provider_type_normalised = provider_type.lower().strip()

        logger.info(
            "LLMProviderFactory: creating provider (type='%s').",
            provider_type_normalised,
        )

        if provider_type_normalised == "local":
            # Import here to avoid circular imports and to keep the
            # factory importable even when the backend dependency for
            # a *different* provider is not installed.
            from backend.reguaz.llm.local_provider import (
                LocalInferenceProvider,
            )

            provider = LocalInferenceProvider(**kwargs)

        else:
            raise ValueError(
                f"LLMProviderFactory: unsupported provider type "
                f"'{provider_type}'.  Currently supported: 'local'."
            )

        logger.info(
            "LLMProviderFactory: provider created (type='%s', model='%s').",
            provider_type_normalised,
            provider.model_name,
        )

        return provider
