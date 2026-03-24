"""Sample preprocessing for reranker training and inference.

Implements tokeniser-based preprocessors that convert raw
``(query, document)`` samples into tokenised inputs.  All classes
conform to the :class:`~bioasq.common.protocols.SamplePreprocessor`
protocol.

Refactored from ``refactored-trainer/sample_preprocessing.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from transformers import BatchEncoding, PreTrainedTokenizerBase

    from bioasq.common.aliases import Sample

# ---------------------------------------------------------------------------
# Nemotron-specific prompt template
# ---------------------------------------------------------------------------


def _nemotron_prompt_template(query: str, passage: str) -> str:
    """Format query and passage per NVIDIA Nemotron reranker spec."""
    return f"question:{query} \n \n passage:{passage}"


# ---------------------------------------------------------------------------
# Preprocessor implementations
# ---------------------------------------------------------------------------


class NemotronSamplePreprocessing:
    """Preprocessor for ``nvidia/llama-nemotron-rerank-1b-v2``.

    Uses the model's expected format:
    ``question:{query} \\n \\n passage:{passage}``
    as a single string, tokenised in one pass.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model_max_length: int = -1,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.model_max_length: int = (
            model_max_length if model_max_length != -1 else cast("int", tokenizer.model_max_length)
        )

    def __call__(self, sample: Sample) -> Sample:
        assert "query_text" in sample
        assert "doc_text" in sample

        text: str = _nemotron_prompt_template(str(sample["query_text"]), str(sample["doc_text"]))
        inputs: BatchEncoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.model_max_length,
        )

        result: Sample = cast("Sample", dict(inputs))
        result["id"] = str(sample["id"])
        result["doc_id"] = str(sample.get("doc_id", ""))
        if "label" in sample:
            result["labels"] = int(sample["label"])
        return result


class BasicSamplePreprocessing:
    """Standard cross-encoder preprocessor.

    Tokenises ``(query, document)`` as a pair using the tokeniser's
    default separator strategy (e.g. ``[SEP]`` for BERT).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model_max_length: int = -1,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.model_max_length: int = (
            model_max_length if model_max_length != -1 else cast("int", tokenizer.model_max_length)
        )

    def __call__(self, sample: Sample) -> Sample:
        assert "query_text" in sample
        assert "doc_text" in sample

        inputs: BatchEncoding = self.tokenizer(
            str(sample["query_text"]),
            str(sample["doc_text"]),
            truncation="only_second" if "label" in sample else True,
            max_length=self.model_max_length,
        )

        result: Sample = cast("Sample", dict(inputs))
        result["id"] = str(sample["id"])
        result["doc_id"] = str(sample.get("doc_id", ""))
        if "label" in sample:
            result["labels"] = int(sample["label"])
        return result
