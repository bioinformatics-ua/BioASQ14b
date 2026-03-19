from abc import ABC, abstractmethod


class BaseModelBackend(ABC):
    """
    Abstract base class for all model backends.

    Every backend (local vLLM, cloud API, etc.) must implement this interface.
    This ensures that swapping backends in the inference script is just a matter
    of changing which backend is instantiated — the rest of the code stays identical.

    Usage:
        class MyBackend(BaseModelBackend):
            def load(self): ...
            def generate(self, prompt): ...
            def unload(self): ...
    """

    @abstractmethod
    def load(self) -> None:
        """
        Load the model into memory.

        For local models (vLLM) this means loading weights onto GPU.
        For cloud models (Claude, OpenAI) this is a no-op or sets up the API client.
        Called once before inference starts.
        """
        ...

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response for a single prompt and return it as a plain string.

        This is the only method the inference script calls per question.
        All batching, retries, and parsing happen inside the backend implementation.

        Args:
            prompt: The fully formatted prompt string (question + context).

        Returns:
            The model's response as a plain string.
        """
        ...

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """
        Generate responses for a list of prompts.

        Default implementation calls generate() in a loop.
        Override in backends that support native batching (e.g. vLLM).
        """
        return [self.generate(p) for p in prompts]

    @abstractmethod
    def unload(self) -> None:
        """
        Release model resources (GPU memory, API connections, etc.).

        For local models this frees GPU memory between runs.
        For cloud models this is typically a no-op.
        Called once after all inference is done.
        """
        ...
