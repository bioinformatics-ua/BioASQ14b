"""Local tokenizer wrapper for Context-1 document truncation and token accounting."""

from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase


class Context1Tokenizer:
    """Lazy local tokenizer for the Context-1 model family."""

    def __init__(self, model_name: str = "chromadb/context-1") -> None:
        self.model_name = model_name
        self._tokenizer: PreTrainedTokenizerBase | None = None

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            self._tokenizer = cast(
                "PreTrainedTokenizerBase",
                AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                ),
            )
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Count model tokens without adding chat special tokens."""

        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate plain text to at most ``max_tokens`` model tokens."""

        if max_tokens <= 0 or not text.strip():
            return ""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        return self.tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True).strip()
