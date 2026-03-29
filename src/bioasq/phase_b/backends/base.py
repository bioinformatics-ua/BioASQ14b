from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from bioasq.common.aliases import Score


@runtime_checkable
class Scorable(Protocol):
    """Anything that can produce a relevance score for a query-document pair."""

    def score(self, query: str, document: str) -> Score: ...


@runtime_checkable
class Loadable(Protocol):
    """Anything that can be loaded into / released from memory."""

    def load(self) -> None: ...
    def unload(self) -> None: ...


@runtime_checkable
class Generatable(Protocol):
    """Anything that can generate text from a prompt."""

    def generate(self, prompt: str) -> str: ...
    def generate_batch(self, prompts: Sequence[str]) -> list[str]: ...


@runtime_checkable
class SamplePreprocessor(Protocol):
    """Tokeniser-based preprocessor for (query, document) samples.

    Implementations: :class:`BasicSamplePreprocessing`,
    :class:`NemotronSamplePreprocessing`, etc.
    """

    def __call__(self, sample: dict[str, str | int]) -> dict[str, str | int]: ...


class BaseModelBackend(ABC):
    """Abstract base for LLM inference backends (local vLLM, OpenRouter, …).

    Uses the template-method pattern: subclasses implement
    :meth:`generate` and optionally override :meth:`generate_batch`
    for optimised batching.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights / establish connection."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a single completion."""

    def generate_chat(self, messages: Sequence[dict[str, str]]) -> str:
        """Generate using a chat-style message list.

        Default implementation concatenates messages into a single
        prompt.  Override for native chat support.
        """
        combined: str = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )
        return self.generate(combined)

    def generate_batch(self, prompts: Sequence[str]) -> list[str]:
        """Generate completions for multiple prompts.

        Default is sequential; override for native batching.
        """
        return [self.generate(p) for p in prompts]

    @abstractmethod
    def unload(self) -> None:
        """Release resources (GPU memory, connections, …)."""
