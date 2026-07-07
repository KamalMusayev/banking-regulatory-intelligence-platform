"""Gemma LLM implementation using llama.cpp."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from llama_cpp import Llama

from backend.reguaz import config
from backend.reguaz.services.generation.base_llm import BaseLLM
from backend.reguaz.utils.logger import get_logger

logger = get_logger(__name__, "llm_generation.log")


class GemmaService(BaseLLM):
    """
    LLM generation service using Gemma models via llama.cpp.

    This service is designed for Apple Silicon natively by using
    hardware acceleration (Metal) via llama-cpp-python.
    """

    def __init__(
        self,
        model_path: str | Path,
        context_window: int,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        seed: int,
    ) -> None:
        """
        Initialize the Gemma service and load the model.

        Parameters
        ----------
        model_path : str | Path
            Path to the GGUF model file.
        context_window : int
            Maximum context length for generation.
        temperature : float
            Sampling temperature (0.0 to 1.0 is typical).
        max_tokens : int
            Maximum number of tokens to generate.
        top_p : float
            Nucleus sampling probability.
        top_k : int
            Top-K sampling value.
        repeat_penalty : float
            Penalty factor for repeated tokens.
        seed : int
            Random seed for reproducibility.

        Raises
        ------
        ValueError
            If validation of any argument fails.
        RuntimeError
            If model loading fails.
        """
        self.model_path = Path(model_path)
        self.context_window = context_window
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.seed = seed

        self._validate_config()
        self._llm: Llama | None = None
        self._load_model()

    def _validate_config(self) -> None:
        """
        Validate all constructor parameters before model loading.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        if not self.model_path.exists() or not self.model_path.is_file():
            raise ValueError(f"Model path does not exist or is not a file: {self.model_path}")
        if self.context_window < 1:
            raise ValueError("context_window must be at least 1.")
        if self.temperature < 0.0:
            raise ValueError("temperature cannot be negative.")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1.")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0.")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if self.repeat_penalty < 1.0:
            raise ValueError("repeat_penalty should be >= 1.0.")

    def _load_model(self) -> None:
        """
        Load the llama.cpp model into memory.

        Raises
        ------
        RuntimeError
            If the model fails to load.
        """
        config_str = (
            f"context_window={self.context_window}, temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, top_p={self.top_p}, top_k={self.top_k}, "
            f"repeat_penalty={self.repeat_penalty}, seed={self.seed}"
        )
        logger.info(f"Loading Gemma model from {self.model_path} with config: {config_str}")
        start_time = time.perf_counter()

        try:
            # Apple Silicon configuration
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_window,
                n_gpu_layers=-1,
                seed=self.seed,
                verbose=False,
            )
            elapsed = time.perf_counter() - start_time
            logger.info(f"Gemma model loaded successfully in {elapsed:.4f} seconds.")
        except Exception as e:
            logger.error("Failed to load Gemma model", exc_info=True)
            raise RuntimeError(f"Gemma model loading failed: {e}") from e

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using llama.cpp tokenizer.
        """
        if self._llm is None or not text:
            return 0
        return len(self._llm.tokenize(text.encode("utf-8")))

    def get_prompt_budget(self) -> int:
        """
        Returns the token budget exclusively reserved for the context portion of the prompt.
        """
        return self.context_window - self.max_tokens - config.PROMPT_RESERVED_TOKENS

    def generate(self, prompt: str) -> str:
        """
        Generate text based on the provided prompt.

        Parameters
        ----------
        prompt : str
            The user prompt to send to the model.

        Returns
        -------
        str
            The text generated by the model.

        Raises
        ------
        ValueError
            If the prompt is empty.
        RuntimeError
            If generation fails or returns an invalid response format.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if self._llm is None:
            raise RuntimeError("Model is not loaded. Cannot generate.")

        logger.info("Generation started")
        start_time = time.perf_counter()

        prompt_tokens = self.count_tokens(prompt)
        total_requested = prompt_tokens + self.max_tokens
        
        logger.info(f"Context window: {self.context_window}")
        logger.info(f"Prompt tokens: {prompt_tokens}")
        logger.info(f"Generation tokens: {self.max_tokens}")
        logger.info(f"Requested total: {total_requested}")
        
        if total_requested > self.context_window:
            excess = total_requested - self.context_window
            logger.warning(f"Prompt exceeds context window by {excess} tokens!")

        try:
            response = self._llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repeat_penalty=self.repeat_penalty,
                echo=False,
            )
            
            answer = self._extract_text(response)
            elapsed = time.perf_counter() - start_time
            logger.info(f"Generation completed in {elapsed:.4f} seconds.")
            return answer
        
        except ValueError as ve:
            logger.error(f"Validation error during generation: {ve}")
            raise RuntimeError(f"Invalid model response: {ve}") from ve
        except Exception as e:
            logger.error("Generation failed", exc_info=True)
            raise RuntimeError(f"Text generation failed: {e}") from e

    def _extract_text(self, response: Any) -> str:
        """
        Extract the generated text safely from the llama.cpp response dictionary.

        Parameters
        ----------
        response : Any
            The raw response returned by the llama.cpp call.

        Returns
        -------
        str
            The extracted generated text.

        Raises
        ------
        ValueError
            If the response format is unexpected or missing required keys.
        """
        if not isinstance(response, dict):
            raise ValueError(f"Expected dict response, got {type(response)}")

        choices = response.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise ValueError("Response missing 'choices' or 'choices' is empty.")

        first_choice = choices[0]
        if not isinstance(first_choice, dict) or "text" not in first_choice:
            raise ValueError("First choice is missing the 'text' key.")

        text = first_choice["text"]
        if not isinstance(text, str):
            raise ValueError(f"Expected string for 'text', got {type(text)}")

        return text.strip()

    def close(self) -> None:
        """
        Release model resources safely.
        """
        if self._llm is not None:
            logger.info("Closing Gemma model and releasing resources.")
            try:
                if hasattr(self._llm, "close") and callable(self._llm.close):
                    self._llm.close()
            except Exception as e:
                logger.warning(f"Error while closing model: {e}")
            finally:
                self._llm = None
