from typing import cast
from transformers import PreTrainedTokenizerBase, BatchEncoding
from aliases import Sample


def _nemotron_prompt_template(query: str, passage: str) -> str:
    """Format query and passage per NVIDIA Nemotron reranker spec: question:... \\n \\n passage:..."""
    return f"question:{query} \n \n passage:{passage}"


class NemotronSamplePreprocessing:
    """Preprocessor for nvidia/llama-nemotron-rerank-1b-v2.

    Uses the model's expected format: question:{query} \\n \\n passage:{passage}
    as a single string, tokenized in one pass.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, model_max_length: int = -1
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.model_max_length: int = (
            model_max_length
            if model_max_length != -1
            else cast(int, tokenizer.model_max_length)
        )

    def __call__(self, sample: Sample) -> Sample:
        assert "query_text" in sample and "doc_text" in sample

        text = _nemotron_prompt_template(
            str(sample["query_text"]), str(sample["doc_text"])
        )
        inputs: BatchEncoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.model_max_length,
        )

        result: Sample = cast(Sample, dict(inputs))
        result["id"] = str(sample["id"])
        result["doc_id"] = str(sample.get("doc_id", ""))
        if "label" in sample:
            result["labels"] = int(sample["label"])
        return result


class BasicSamplePreprocessing:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, model_max_length: int = -1
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.model_max_length: int = (
            model_max_length
            if model_max_length != -1
            else cast(int, tokenizer.model_max_length)
        )

    def __call__(self, sample: Sample) -> Sample:
        assert "query_text" in sample and "doc_text" in sample

        inputs: BatchEncoding = self.tokenizer(
            str(sample["query_text"]),
            str(sample["doc_text"]),
            truncation="only_second" if "label" in sample else True,
            max_length=self.model_max_length,
        )

        result: Sample = cast(Sample, dict(inputs))
        result["id"] = str(sample["id"])
        result["doc_id"] = str(sample.get("doc_id", ""))
        if "label" in sample:
            result["labels"] = int(sample["label"])
        return result
