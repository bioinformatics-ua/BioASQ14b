from typing import cast

from transformers import BatchEncoding, PreTrainedTokenizerBase

from bioasq.phase_a.reranker.aliases import Sample


def _needs_zero_token_type_ids(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Detect Gemma3-style tokenizers that require multimodal token_type_ids during training.

    Gemma3/MedGemma text tokenizers do not emit ``token_type_ids`` by default, but the
    model requires them during training to distinguish image tokens. For text-only reranking,
    every token should remain text, so we synthesize an all-zero mask.
    """

    model_input_names = set(getattr(tokenizer, "model_input_names", []))
    tokenizer_class = tokenizer.__class__.__name__.lower()
    name_or_path = str(getattr(tokenizer, "name_or_path", "")).lower()
    return (
        "token_type_ids" not in model_input_names
        and hasattr(tokenizer, "image_token_id")
        and ("gemma" in tokenizer_class or "gemma" in name_or_path)
    )


def _ensure_token_type_ids(
    tokenizer: PreTrainedTokenizerBase,
    inputs: BatchEncoding,
) -> BatchEncoding:
    """Normalize Gemma3-style text-only token_type_ids to all zeros."""

    if not _needs_zero_token_type_ids(tokenizer):
        return inputs

    input_ids = inputs.get("input_ids")
    if input_ids is None:
        return inputs

    inputs["token_type_ids"] = [0] * len(input_ids)
    return inputs


def _nemotron_prompt_template(query: str, passage: str) -> str:
    """Format query and passage per NVIDIA Nemotron reranker spec: question:... \\n \\n passage:..."""
    return f"question:{query} \n \n passage:{passage}"


class NemotronSamplePreprocessing:
    """Preprocessor for nvidia/llama-nemotron-rerank-1b-v2.

    Uses the model's expected format: question:{query} \\n \\n passage:{passage}
    as a single string, tokenized in one pass.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, model_max_length: int = -1) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.model_max_length: int = (
            model_max_length if model_max_length != -1 else cast(int, tokenizer.model_max_length)
        )

    def __call__(self, sample: Sample) -> Sample:
        assert "query_text" in sample and "doc_text" in sample

        text = _nemotron_prompt_template(str(sample["query_text"]), str(sample["doc_text"]))
        inputs: BatchEncoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.model_max_length,
        )
        inputs = _ensure_token_type_ids(self.tokenizer, inputs)

        result: Sample = cast(Sample, dict(inputs))
        result["id"] = str(sample["id"])
        result["doc_id"] = str(sample.get("doc_id", ""))
        if "label" in sample:
            result["labels"] = int(sample["label"])
        return result


class BasicSamplePreprocessing:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, model_max_length: int = -1) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.model_max_length: int = (
            model_max_length if model_max_length != -1 else cast(int, tokenizer.model_max_length)
        )

    def __call__(self, sample: Sample) -> Sample:
        assert "query_text" in sample and "doc_text" in sample

        inputs: BatchEncoding = self.tokenizer(
            str(sample["query_text"]),
            str(sample["doc_text"]),
            truncation="only_second" if "label" in sample else True,
            max_length=self.model_max_length,
        )
        inputs = _ensure_token_type_ids(self.tokenizer, inputs)

        result: Sample = cast(Sample, dict(inputs))
        result["id"] = str(sample["id"])
        result["doc_id"] = str(sample.get("doc_id", ""))
        if "label" in sample:
            result["labels"] = int(sample["label"])
        return result
