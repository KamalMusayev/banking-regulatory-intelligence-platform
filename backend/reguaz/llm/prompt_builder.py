"""
backend/reguaz/llm/prompt_builder.py

Dedicated prompt construction for the banking regulatory domain.

``PromptBuilder`` is the **sole** component responsible for assembling
the final prompt string that is forwarded to the LLM provider.  Neither
the ``Generator`` nor any ``Provider`` should ever construct or modify
prompts directly.

The system prompt enforces strict constraints:

* Answer only from the supplied context.
* Never hallucinate, speculate, or use outside knowledge.
* Explicitly state when the provided documents are insufficient.
* Always respond in professional Azerbaijani.
* Keep answers concise, factual, and formal.
* Never expose internal implementation details.

The class is intentionally simple and stateless so that it can be
extended or replaced easily as prompt engineering requirements evolve.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# System prompt — banking regulatory domain (Azerbaijani)
# ──────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT: str = (
    "Sən Azərbaycan bank və maliyyə tənzimləmələri üzrə ixtisaslaşmış "
    "peşəkar hüquqi məsləhətçisən.\n"
    "\n"
    "Qaydalar:\n"
    "1. YALNIZ aşağıda verilən kontekstdən istifadə et.\n"
    "2. Kontekstdə olmayan məlumatı HEÇBIR HALDA uydurma.\n"
    "3. Xarici biliklərdən istifadə etmə.\n"
    "4. Fərziyyə irəli sürmə və ya təxmin etmə.\n"
    "5. Əgər cavab verilən sənədlərdə yoxdursa, bunu açıq şəkildə bildir: "
    "\"Təqdim edilmiş sənədlərdə bu suala cavab vermək üçün kifayət qədər "
    "məlumat yoxdur.\"\n"
    "6. Həmişə peşəkar Azərbaycan dilində cavab ver.\n"
    "7. Cavabları qısa, faktiki və rəsmi saxla.\n"
    "8. Daxili texniki təfərrüatları heç vaxt açıqlama.\n"
)


class PromptBuilder:
    """
    Constructs fully formatted prompts for the LLM provider.

    This is the only component in the pipeline that knows how to
    assemble a prompt.  The resulting string is passed directly to
    ``BaseLLMProvider.generate()`` without further modification.

    The system prompt is exposed as a class-level constant so that it
    can be inspected in tests and overridden in subclasses if needed.
    """

    def __init__(self) -> None:
        logger.debug("PromptBuilder: initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def system_prompt(self) -> str:
        """
        Return the current system prompt.

        Returns
        -------
        str
            The banking-domain system prompt in Azerbaijani.
        """
        return _SYSTEM_PROMPT

    def build_prompt(self, question: str, context: str) -> str:
        """
        Build a complete prompt from a user question and retrieved context.

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
            A fully formatted prompt string ready for the LLM provider.
        """
        prompt = (
            f"{_SYSTEM_PROMPT}\n"
            f"Kontekst:\n"
            f"{context}\n"
            f"\n"
            f"Sual:\n"
            f"{question}\n"
            f"\n"
            f"Cavab:\n"
        )

        logger.debug(
            "PromptBuilder.build_prompt: prompt built "
            "(question length=%d chars, context length=%d chars).",
            len(question),
            len(context),
        )

        return prompt
