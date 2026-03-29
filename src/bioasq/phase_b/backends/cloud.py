import os
import time
from collections.abc import Sequence

from bioasq.phase_b.backends.base import BaseModelBackend


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

    def generate_chat(self, messages: Sequence[dict[str, str]]) -> str:
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
