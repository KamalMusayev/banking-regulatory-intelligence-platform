"""
backend/reguaz/llm/local_provider.py

Local inference provider.

``LocalInferenceProvider`` performs LLM inference within the current
process.  The underlying inference backend is an **internal implementation
detail** — today it uses Hugging Face Transformers as a temporary backend
to validate the complete Retrieval → Prompt → LLM → Answer pipeline.

The provider abstraction (``BaseLLMProvider``) ensures that replacing this
backend later (e.g. with llama.cpp, vLLM, or any other engine) requires
**zero changes** to Generator, PromptBuilder, Factory interface, or
Retrieval.

Internal structure
------------------
``generate(prompt)``
  → ``_load_model()``       (lazy, first-call only)
  → ``_prepare_inputs()``   (tokenisation, returns inputs + input_length)
  → ``_run_inference()``    (backend-specific — isolated)
  → ``_postprocess()``      (decode, clean, extract answer)
  → return answer

Device handling
---------------
``device_map`` (an Accelerate feature) is used only for CUDA, where it
enables multi-GPU sharding via ``device_map="auto"``.  For MPS and CPU
the model is loaded in the default (CPU) state and then moved with
``.to(device)`` — the correct cross-platform approach.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from backend.reguaz.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Default generation parameters
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"
_DEFAULT_MAX_NEW_TOKENS: int = 512
_DEFAULT_TEMPERATURE: float = 0.3
_DEFAULT_TOP_P: float = 0.9
_DEFAULT_REPETITION_PENALTY: float = 1.1


class LocalInferenceProvider(BaseLLMProvider):
    """
    Provider representing local inference.

    Performs text generation using a model loaded into the current
    process.  The model and all generation parameters are fully
    configurable.

    Parameters
    ----------
    model_name : str
        Model identifier or local path.  Default:
        ``"Qwen/Qwen2.5-3B-Instruct"``.
    device : str | None
        Target device for inference (``"cuda"``, ``"mps"``, ``"cpu"``).
        If ``None``, auto-detected following the project convention
        (MPS → CUDA → CPU).
    max_new_tokens : int
        Maximum number of tokens to generate.  Default: 512.
    temperature : float
        Sampling temperature.  Default: 0.3.
    top_p : float
        Nucleus sampling probability.  Default: 0.9.
    repetition_penalty : float
        Repetition penalty factor.  Default: 1.1.

    Notes
    -----
    The model and tokenizer are loaded **lazily** on the first call to
    ``generate()`` to keep factory instantiation fast.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        device: str | None = None,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        top_p: float = _DEFAULT_TOP_P,
        repetition_penalty: float = _DEFAULT_REPETITION_PENALTY,
    ) -> None:
        self._model_name_str = model_name
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty

        # Lazy-loaded backend resources.
        self._tokenizer: Any = None
        self._model: Any = None
        self._is_loaded: bool = False

        logger.info(
            "LocalInferenceProvider: created (model='%s', device=%s, "
            "max_new_tokens=%d, temperature=%.2f, top_p=%.2f, "
            "repetition_penalty=%.2f).  Model will be loaded on first use.",
            self._model_name_str,
            self._device or "auto",
            self._max_new_tokens,
            self._temperature,
            self._top_p,
            self._repetition_penalty,
        )

    # ------------------------------------------------------------------
    # BaseLLMProvider interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """
        Return the configured model identifier.

        Returns
        -------
        str
            Model name or path.
        """
        return self._model_name_str

    def generate(self, prompt: str) -> str:
        """
        Generate a response from a fully formatted prompt.

        The method orchestrates the internal pipeline:
        load → prepare → infer → postprocess.

        Parameters
        ----------
        prompt : str
            A complete, ready-to-use prompt string produced by
            ``PromptBuilder``.

        Returns
        -------
        str
            The generated answer text.
        """
        if not self._is_loaded:
            self._load_model()

        logger.info(
            "LocalInferenceProvider.generate: starting inference "
            "(prompt length=%d chars).",
            len(prompt),
        )
        t0 = time.perf_counter()

        inputs, input_length = self._prepare_inputs(prompt)
        raw_output = self._run_inference(inputs)
        answer = self._postprocess(raw_output, input_length)

        elapsed = time.perf_counter() - t0
        logger.info(
            "LocalInferenceProvider.generate: inference completed in %.3f s "
            "(answer length=%d chars).",
            elapsed,
            len(answer),
        )

        return answer

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Lazily load the model and tokenizer into memory.

        This method is called once on the first ``generate()`` invocation.
        The backend-specific import is intentionally placed here to keep
        the module importable without the backend dependency installed
        (useful for tests that mock the provider).
        """
        logger.info(
            "LocalInferenceProvider: loading model '%s' …",
            self._model_name_str,
        )
        t0 = time.perf_counter()

        # Backend-specific imports — isolated here.
        # pyrefly: ignore [missing-import]
        import torch
        # pyrefly: ignore [missing-import]
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Device auto-detection (mirrors reranker.py convention).
        if self._device is None:
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"

        logger.info(
            "LocalInferenceProvider: selected device '%s'.",
            self._device,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_str,
            trust_remote_code=True,
        )

        # ``device_map`` is an Accelerate feature designed for CUDA
        # multi-GPU sharding.  It does not support MPS or plain CPU
        # in the same way.  For CUDA we use ``device_map="auto"`` to
        # enable automatic sharding; for MPS and CPU we load into the
        # default (CPU) state and move the model with ``.to(device)``.
        if self._device == "cuda":
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name_str,
                trust_remote_code=True,
                device_map="auto",
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name_str,
                trust_remote_code=True,
            )
            self._model = self._model.to(self._device)

        self._model.eval()

        self._is_loaded = True

        elapsed = time.perf_counter() - t0
        logger.info(
            "LocalInferenceProvider: model '%s' loaded on device '%s' "
            "in %.3f s.",
            self._model_name_str,
            self._device,
            elapsed,
        )

    def _prepare_inputs(self, prompt: str) -> tuple[Any, int]:
        """
        Tokenise and format the prompt for the inference backend.

        Returns both the input tensors and the input token count.  The
        token count is forwarded to ``_postprocess()`` to avoid
        re-tokenising the prompt a second time.

        Parameters
        ----------
        prompt : str
            The fully formatted prompt string.

        Returns
        -------
        tuple[Any, int]
            A ``(inputs, input_length)`` pair where ``inputs`` is the
            dict of input tensors ready for inference and
            ``input_length`` is the number of prompt tokens.
        """
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )

        input_length: int = inputs["input_ids"].shape[-1]

        # Move tensors to the model's device.
        inputs = {
            key: value.to(self._model.device)
            for key, value in inputs.items()
        }

        logger.debug(
            "LocalInferenceProvider._prepare_inputs: "
            "tokenised to %d tokens.",
            input_length,
        )

        return inputs, input_length

    def _run_inference(self, inputs: Any) -> Any:
        """
        Execute the inference backend.

        This is the **only** method that interacts with the underlying
        inference engine.  Swapping to a different backend (llama.cpp,
        vLLM, etc.) requires changing only this method.

        Parameters
        ----------
        inputs : Any
            Backend-specific input tensors from ``_prepare_inputs()``.

        Returns
        -------
        Any
            Raw model output (token IDs).
        """
        # pyrefly: ignore [missing-import]
        import torch

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                repetition_penalty=self._repetition_penalty,
                do_sample=self._temperature > 0.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        logger.debug(
            "LocalInferenceProvider._run_inference: "
            "generated %d output tokens.",
            output_ids.shape[-1],
        )

        return output_ids

    def _postprocess(self, raw_output: Any, input_length: int) -> str:
        """
        Decode and clean the raw model output.

        Strips the input prompt tokens from the output so that only the
        newly generated text is returned.  The ``input_length`` is
        supplied by ``_prepare_inputs()`` to avoid re-tokenising the
        prompt a second time.

        Parameters
        ----------
        raw_output : Any
            Raw token IDs from ``_run_inference()``.
        input_length : int
            Number of tokens in the input prompt, as returned by
            ``_prepare_inputs()``.

        Returns
        -------
        str
            The cleaned, generated answer text.
        """
        # Decode only the newly generated tokens.
        generated_ids = raw_output[0, input_length:]
        answer = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        answer = answer.strip()

        logger.debug(
            "LocalInferenceProvider._postprocess: "
            "decoded %d new tokens → %d chars.",
            len(generated_ids),
            len(answer),
        )

        return answer
