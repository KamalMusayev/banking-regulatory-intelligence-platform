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

Chat-template readiness
-----------------------
The internal method ``_build_messages()`` structures the prompt as a
list of role/content message dicts — the standard input format for
``tokenizer.apply_chat_template()``.  Today ``build_prompt()`` joins
these messages into a plain string; in a future migration to chat
templates, only ``build_prompt()`` needs to change while
``_build_messages()`` remains untouched.
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

    The system prompt is exposed via the ``system_prompt`` property so
    that it can be inspected in tests and overridden in subclasses.

    Future migration path
    ---------------------
    When the project adopts chat-template-based models, ``build_prompt``
    can be updated to call ``tokenizer.apply_chat_template(_build_messages(...))``
    instead of joining to a string.  ``_build_messages`` itself — and
    therefore the prompt semantics — will remain unchanged.
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

        Internally delegates to ``_build_messages()`` to construct the
        canonical message list, then formats it as a plain string.

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
        messages = self._build_messages(question=question, context=context)

        # Render messages to a plain string.
        # Future: replace this block with tokenizer.apply_chat_template(messages).
        user_content = messages[1]["content"]
        prompt = (
            f"{_SYSTEM_PROMPT}\n"
            f"{user_content}"
            f"Cavab:\n"
        )

        logger.debug(
            "PromptBuilder.build_prompt: prompt built "
            "(question length=%d chars, context length=%d chars).",
            len(question),
            len(context),
        )

        return prompt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        question: str,
        context: str,
    ) -> list[dict[str, str]]:
        """
        Build the canonical message list for this prompt.

        Returns a list of role/content dicts in the format expected by
        ``tokenizer.apply_chat_template()``.  Today this list is
        rendered to a plain string by ``build_prompt()``; in a future
        migration it can be passed directly to the chat-template API.

        Parameters
        ----------
        question : str
            The end-user's question in natural language.
        context : str
            The retrieved document context.

        Returns
        -------
        list[dict[str, str]]
            A two-element list::

                [
                    {"role": "system", "content": <system_prompt>},
                    {"role": "user",   "content": <context_and_question>},
                ]
        """
        user_content = (
            f"Kontekst:\n"
            f"{context}\n"
            f"\n"
            f"Sual:\n"
            f"{question}\n"
            f"\n"
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
