"""Abstract base and concrete LLM backends for Phase B.

Backends conform to :class:`~bioasq.common.protocols.BaseModelBackend`.

Refactored from ``phaseB/loaders/base.py`` + ``phaseB/loaders/local.py``
+ ``phaseB/loaders/cloud.py`` (merged both phaseB and phaseB-alex versions).
"""

from __future__ import annotations

import gc
import os
import time
from collections.abc import Sequence

from bioasq.common.protocols import BaseModelBackend


# ---------------------------------------------------------------------------
# Local vLLM backend
# ---------------------------------------------------------------------------


class VLLMBackend(BaseModelBackend):
    """Local model backend using vLLM.

    Parameters
    ----------
    model_path:
        Path to model weights on disk.
    max_new_tokens:
        Maximum tokens to generate per prompt.
    temperature:
        Sampling temperature.
    tensor_parallel_size:
        Number of GPUs to shard across.
    gpu_memory_utilization:
        Fraction of GPU VRAM vLLM may use (0.0–1.0).
    max_model_len:
        Maximum context length in tokens.
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 1000,
        temperature: float = 0.5,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
    ) -> None:
        from vllm import SamplingParams

        self.model_path: str = model_path
        self.sampling_params: SamplingParams = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        self.tensor_parallel_size: int = tensor_parallel_size
        self.gpu_memory_utilization: float = gpu_memory_utilization
        self.max_model_len: int = max_model_len
        self._llm: object | None = None

    def load(self) -> None:
        """Load model weights onto GPU via vLLM."""
        from vllm import LLM

        print(f"Loading model from {self.model_path}...")
        self._llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
        )
        print("Model loaded.")

    def generate(self, prompt: str) -> str:
        """Generate a single completion."""
        return self.generate_batch([prompt])[0]

    def generate_chat(
        self, messages: Sequence[dict[str, str]]
    ) -> str:
        """Generate using the model's chat template."""
        if self._llm is None:
            msg: str = "Model is not loaded. Call load() first."
            raise RuntimeError(msg)
        outputs = self._llm.chat(  # type: ignore[union-attr]
            messages=[list(messages)],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text

    def _truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt to fit within context window."""
        if self._llm is None:
            return prompt
        tokenizer = self._llm.get_tokenizer()  # type: ignore[union-attr]
        max_new_tokens: int = self.sampling_params.max_tokens or 16
        max_input: int = self.max_model_len - max_new_tokens
        ids: list[int] = tokenizer.encode(prompt)
        if len(ids) <= max_input:
            return prompt
        ids = ids[:max_input]
        return tokenizer.decode(ids)

    def generate_batch(self, prompts: Sequence[str]) -> list[str]:
        """Run inference on a batch of prompts."""
        if self._llm is None:
            msg: str = "Model is not loaded. Call load() first."
            raise RuntimeError(msg)
        truncated: list[str] = [self._truncate_prompt(p) for p in prompts]
        outputs = self._llm.generate(truncated, self.sampling_params)  # type: ignore[union-attr]
        return [o.outputs[0].text for o in outputs]

    def unload(self) -> None:
        """Release GPU memory."""
        import torch

        del self._llm
        self._llm = None
        gc.collect()
        torch.cuda.empty_cache()
        print("Model unloaded.")


# ---------------------------------------------------------------------------
# Cloud (OpenRouter) backend
# ---------------------------------------------------------------------------


class OpenRouterBackend(BaseModelBackend):
    """Cloud LLM backend via OpenRouter API.

    Merges features from both phaseB and phaseB-alex versions:
    - ``generate_chat`` support (from phaseB)
    - ``request_delay`` throttling (from phaseB-alex)

    Parameters
    ----------
    model:
        OpenRouter model name (e.g. ``"google/gemini-2.5-flash"``).
    max_tokens:
        Maximum tokens per response.
    temperature:
        Sampling temperature.
    request_delay:
        Seconds to wait between requests (rate limiting).
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 1000,
        temperature: float = 0.5,
        request_delay: float = 0.0,
    ) -> None:
        self.model: str = model
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature
        self.request_delay: float = request_delay
        self._client: object | None = None

    def load(self) -> None:
        """Establish connection to OpenRouter API."""
        from openai import OpenAI

        api_key: str | None = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            msg: str = "OPENROUTER_API_KEY environment variable not set."
            raise RuntimeError(msg)

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        print(f"OpenRouter client ready (model={self.model}).")

    def generate(self, prompt: str) -> str:
        """Generate a single completion via OpenRouter."""
        return self.generate_chat([{"role": "user", "content": prompt}])

    def generate_chat(
        self, messages: Sequence[dict[str, str]]
    ) -> str:
        """Generate using a chat-style message list."""
        if self._client is None:
            msg: str = "Client not initialised. Call load() first."
            raise RuntimeError(msg)

        if self.request_delay > 0:
            time.sleep(self.request_delay)

        response = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self.model,
            messages=list(messages),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def generate_batch(self, prompts: Sequence[str]) -> list[str]:
        """Generate completions sequentially (no native batching)."""
        return [self.generate(p) for p in prompts]

    def unload(self) -> None:
        """Release client resources."""
        self._client = None
        print("OpenRouter client released.")
