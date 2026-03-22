"""
loaders/cloud.py

Cloud model backend for BioASQ Phase B inference.

Provides one backend:
    - OpenRouterBackend — any model via OpenRouter (openrouter.ai)

OpenRouter gives access to Claude, Gemma, Mistral, and many others through a
single OpenAI-compatible API. Switching models is just a matter of changing
the model name string in the Slurm script — no code changes needed.

Switching from local to cloud in run.py is a single argument change:
    --backend local      → uses VLLMBackend
    --backend openrouter → uses OpenRouterBackend

API key is read from the environment — never hardcoded:
    OPENROUTER_API_KEY
"""

import os
import time
from loaders.base import BaseModelBackend


# ---------------------------------------------------------------------------
# OpenRouter — primary cloud backend
# ---------------------------------------------------------------------------

class OpenRouterBackend(BaseModelBackend):
    """
    Cloud backend via OpenRouter (openrouter.ai).

    OpenRouter is OpenAI-compatible and routes requests to many models
    including Claude, Gemma, Mistral, and others. This lets us test
    different cloud models without changing any pipeline code — just
    swap the model name.

    Reads OPENROUTER_API_KEY from the environment.

    Model name format: "provider/model-name", e.g.:
        "anthropic/claude-sonnet-4-6"
        "google/gemma-3-27b-it"
        "mistralai/mistral-7b-instruct"

    Example:
        backend = OpenRouterBackend(model="anthropic/claude-sonnet-4-6")
        backend.load()
        answer = backend.generate("Question: ... Context: ...")
        backend.unload()
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-6",
        max_tokens: int = 1000,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        request_delay: float = 0.0,
    ):
        self.model         = model
        self.max_tokens    = max_tokens
        self.temperature   = temperature
        self.max_retries   = max_retries
        self.retry_delay   = retry_delay
        self.request_delay = request_delay
        self._client       = None

    def load(self) -> None:
        """Initialise the OpenAI client pointed at OpenRouter."""
        import openai

        api_key: str | None = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY environment variable is not set."
            )

        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=self.OPENROUTER_BASE_URL,
        )
        print(f"OpenRouter backend ready — model: {self.model}")

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to OpenRouter and return the text response.

        Retries automatically on transient API errors (rate limits, timeouts).
        """
        if self._client is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content or ""

            except Exception as e:
                if attempt == self.max_retries:
                    print(f"OpenRouter API failed after {self.max_retries} attempts: {e}")
                    return ""
                print(f"OpenRouter API error (attempt {attempt}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay)

        return ""

    def generate_batch(self, prompts: list[str]) -> list[str]:
        results = []
        for i, prompt in enumerate(prompts):
            results.append(self.generate(prompt))
            if self.request_delay > 0 and i < len(prompts) - 1:
                time.sleep(self.request_delay)
        return results

    def unload(self) -> None:
        """No-op for cloud backends — nothing to release."""
        self._client = None
        print("OpenRouter backend unloaded.")

