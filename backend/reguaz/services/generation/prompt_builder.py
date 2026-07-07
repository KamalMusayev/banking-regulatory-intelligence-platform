"""
backend/reguaz/services/generation/prompt_builder.py
"""

from typing import Any

from backend.reguaz.utils.logger import get_logger

logger = get_logger(__name__, "prompt_builder.log")

class PromptBuilder:
    """
    Model-agnostic prompt builder for RAG generation.
    """
    
    SYSTEM_INSTRUCTION = (
        "You are an expert legal and regulatory assistant for the ReguAZ platform. "
        "Answer the user's question based strictly on the provided context documents. "
        "Do not hallucinate or use outside knowledge. If the answer is not contained "
        "in the context, state clearly that you do not have enough information to answer."
    )

    @classmethod
    def build_prompt(
        cls, 
        question: str, 
        sources: list[dict[str, Any]], 
        token_counter: Any = None
    ) -> str:
        """
        Builds the final prompt string from the question and retrieved sources.

        Parameters
        ----------
        question : str
            The user's query.
        sources : list[dict[str, Any]]
            A list of chunk metadata dictionaries.
        token_counter : Any, optional
            A callable that takes a string and returns a token count.

        Returns
        -------
        str
            The formatted prompt string containing system instructions, context, and the question.
        """
        logger.debug("PromptBuilder: formatting prompt with %d sources.", len(sources))
        
        context_blocks = []
        chunk_char_counts = []
        for idx, source in enumerate(sources, start=1):
            block = cls._format_source(idx, source)
            context_blocks.append(block)
            chunk_char_counts.append(len(block))
            
        context_str = "\n\n---\n\n".join(context_blocks)
        
        prompt = (
            f"{cls.SYSTEM_INSTRUCTION}\n\n"
            f"Context:\n\n{context_str}\n\n"
            f"---\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        
        # Diagnostic logging
        total_chars = len(prompt)
        prompt_tokens = token_counter(prompt) if token_counter else 0
        
        logger.info(
            "Prompt Diagnostics:\n"
            "Total prompt character count: %d\n"
            "Approximate prompt token count: %d\n"
            "Number of chunks inserted: %d\n"
            "Character count contributed by each chunk: %s",
            total_chars,
            prompt_tokens,
            len(sources),
            chunk_char_counts
        )
        
        # Save prompt to debug file
        try:
            from pathlib import Path
            debug_path = Path("debug/debug_prompt.txt")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(prompt, encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to write debug_prompt.txt: %s", e)
            
        return prompt

    @staticmethod
    def _format_source(index: int, source: dict[str, Any]) -> str:
        """Formats a single source metadata dictionary into a readable context block."""
        lines = [f"--- Document {index} ---"]
        
        # Format metadata fields consistently, skipping missing ones
        metadata_fields = [
            ("Document", "title"),
            ("Category", "category"),
            ("Chapter", "chapter"),
            ("Article", "article"),
            ("Section", "section"),
            ("Subsection", "subsection"),
        ]
        
        for label, key in metadata_fields:
            if value := source.get(key):
                lines.append(f"{label}: {value}")
                
        # Handle page/page range
        page_start = source.get("page_start")
        page_end = source.get("page_end")
        if page_start is not None and page_end is not None:
            if page_start == page_end:
                lines.append(f"Page: {page_start}")
            else:
                lines.append(f"Pages: {page_start}-{page_end}")
        elif page_start is not None:
            lines.append(f"Page: {page_start}")
            
        lines.append("")
        lines.append("Content:")
        
        text = source.get("text") or source.get("content") or ""
        lines.append(text.strip())
        
        return "\n".join(lines)
