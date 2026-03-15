from typing import cast
from transformers import PreTrainedTokenizerBase, BatchEncoding
from aliases import Sample


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
