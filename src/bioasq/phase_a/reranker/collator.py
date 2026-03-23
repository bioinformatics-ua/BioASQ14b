"""Data collation for reranker training and inference.

Collators batch together tokenised samples, pad model inputs, and
preserve metadata keys for downstream evaluation.

Hierarchy::

    RankingCollator
    ├── RankingCollatorForCausalLM   (decoder-only models)
    └── RankingCollatorForSeq2Seq    (encoder-decoder models)

    PairwiseCollator
    MultiNegativePairwiseCollator

    SentenceCollator
    ├── PairwiseSentenceCollator
    └── RankingSentenceCollator

Refactored from ``refactored-trainer/collator.py``.
"""

from __future__ import annotations

from typing import cast

from transformers import BatchEncoding, PreTrainedTokenizerBase

type SentenceBatchSample = dict[str, list[list[int]]]

# Tokenised sample before collation: keys are input_ids, attention_mask, etc.
# Values are lists of ints (token IDs) or metadata (str, int).
type TokenisedSample = dict[str, list[int] | int | str]

_MODEL_INPUT_KEYS: set[str] = {"input_ids", "attention_mask", "token_type_ids"}


def _filter_model_inputs(
    samples: list[TokenisedSample],
) -> list[dict[str, list[int]]]:
    """Keep only tensorizable keys for ``tokenizer.pad``."""
    return [
        {k: v for k, v in s.items() if k in _MODEL_INPUT_KEYS}  # type: ignore[misc]
        for s in samples
    ]


# ---------------------------------------------------------------------------
# Core collators
# ---------------------------------------------------------------------------


class RankingCollator:
    """Collator for single-encoding-per-sample ranking tasks.

    Each sample is one (query, document) pair already tokenized.  Pads
    model inputs and passes through metadata (labels, doc_id, …).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model_inputs: set[str] | None = None,
        padding: bool | str = True,
        max_length: int | None = None,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.model_inputs: set[str] | None = model_inputs
        self.padding: bool | str = padding
        self.max_length: int | None = max_length

    def __call__(
        self, batch: list[TokenisedSample]
    ) -> dict[str, BatchEncoding | list[int] | list[str]]:
        grouped: dict[str, list[list[int] | int | str]] = {
            key: [i[key] for i in batch] for key in batch[0]
        }

        if self.model_inputs is None:
            standard_inputs: set[str] = {"input_ids", "attention_mask", "token_type_ids"}
            model_inputs: set[str] = set(grouped.keys()) & standard_inputs
        else:
            model_inputs = self.model_inputs

        reminder_keys: set[str] = set(grouped.keys()) - model_inputs
        return {
            "inputs": self.tokenizer.pad(
                {k: grouped[k] for k in model_inputs},
                padding=self.padding,
                max_length=self.max_length,
                return_tensors="pt",
            )
        } | {k: grouped[k] for k in reminder_keys}


class PairwiseCollator:
    """Collator for pairwise training with (positive, negative) document pairs."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer

    def __call__(
        self, batch: list[dict[str, TokenisedSample]]
    ) -> dict[str, BatchEncoding]:
        pos_list: list[dict[str, list[int]]] = _filter_model_inputs(
            [s["pos_inputs"] for s in batch]  # type: ignore[arg-type]
        )
        neg_list: list[dict[str, list[int]]] = _filter_model_inputs(
            [s["neg_inputs"] for s in batch]  # type: ignore[arg-type]
        )
        return {
            "pos_inputs": self.tokenizer.pad(
                pos_list, padding=True, return_tensors="pt"
            ),
            "neg_inputs": self.tokenizer.pad(
                neg_list, padding=True, return_tensors="pt"
            ),
        }


class MultiNegativePairwiseCollator:
    """Collator for 1 positive + N negatives per sample.

    Pads positives to one batch ``[batch_size, seq_len]``, and negatives
    to a LIST of batches: ``neg_inputs[i]`` = batch of the i-th negative
    from each sample.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer

    def __call__(
        self, batch: list[dict[str, TokenisedSample | list[TokenisedSample]]]
    ) -> dict[str, BatchEncoding | list[BatchEncoding]]:
        pos_list: list[dict[str, list[int]]] = _filter_model_inputs(
            [sample["pos_inputs"] for sample in batch]  # type: ignore[arg-type]
        )
        pos_inputs: BatchEncoding = self.tokenizer.pad(
            pos_list, padding=True, return_tensors="pt"
        )

        neg_inputs_raw: list[list[TokenisedSample]] = [
            sample["neg_inputs"] for sample in batch  # type: ignore[misc]
        ]
        num_negs: int = len(neg_inputs_raw[0])
        neg_batches: list[BatchEncoding] = []
        for j in range(num_negs):
            neg_j: list[dict[str, list[int]]] = _filter_model_inputs(
                [neg_inputs_raw[i][j] for i in range(len(batch))]
            )
            neg_batches.append(
                self.tokenizer.pad(neg_j, padding=True, return_tensors="pt")
            )

        return {"pos_inputs": pos_inputs, "neg_inputs": neg_batches}


# ---------------------------------------------------------------------------
# Architecture-specific variants
# ---------------------------------------------------------------------------


class RankingCollatorForCausalLM(RankingCollator):
    """Ranking collator for decoder-only (causal LM) models.

    Uses only ``input_ids`` and ``attention_mask``; no ``token_type_ids``.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model_inputs: set[str] | None = None,
    ) -> None:
        if model_inputs is None:
            model_inputs = {"input_ids", "attention_mask"}
        super().__init__(tokenizer, model_inputs=model_inputs)


class RankingCollatorForSeq2Seq(RankingCollator):
    """Ranking collator for encoder-decoder models (T5, BART).

    Includes ``decoder_input_ids`` in model inputs.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model_inputs: set[str] | None = None,
    ) -> None:
        if model_inputs is None:
            model_inputs = {"input_ids", "attention_mask", "decoder_input_ids"}
        super().__init__(tokenizer, model_inputs=model_inputs)


# ---------------------------------------------------------------------------
# Sentence-level collators (for multi-chunk documents)
# ---------------------------------------------------------------------------


class SentenceCollator:
    """Collator for samples with multiple sentences per document.

    Flattens all sentences across the batch into one padded tensor and
    tracks ``sentences_count`` per original sample.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: bool | str = True,
        max_length: int | None = None,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.padding: bool | str = padding
        self.max_length: int | None = max_length

    def __call__(self, batch: list[SentenceBatchSample]) -> BatchEncoding:
        first_sample: SentenceBatchSample = batch[0]
        standard_keys: set[str] = {"input_ids", "attention_mask", "token_type_ids"}
        present_keys: list[str] = [k for k in standard_keys if k in first_sample]

        expanded_batch: dict[str, list[list[int]]] = {k: [] for k in present_keys}
        sentences_count: list[int] = []

        for sample in batch:
            for k in expanded_batch:
                expanded_batch[k].extend(sample[k])
            sentences_count.append(len(sample["input_ids"]))

        expanded_batch_padded: BatchEncoding = self.tokenizer.pad(
            expanded_batch,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )
        expanded_batch_padded["sentences_count"] = sentences_count
        return expanded_batch_padded


class PairwiseSentenceCollator(SentenceCollator):
    """Pairwise training with multi-sentence documents."""

    def __call__(  # type: ignore[override]
        self, batch: list[dict[str, SentenceBatchSample]]
    ) -> dict[str, BatchEncoding]:
        pos_batch: list[SentenceBatchSample] = []
        neg_batch: list[SentenceBatchSample] = []
        for sample in batch:
            pos_batch.append(sample["pos_inputs"])
            neg_batch.append(sample["neg_inputs"])

        return {
            "pos_inputs": super().__call__(pos_batch),
            "neg_inputs": super().__call__(neg_batch),
        }


class RankingSentenceCollator(SentenceCollator):
    """Sentence-level collator with metadata passthrough."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model_inputs_keys: set[str] | None = None,
        padding: bool | str = True,
        max_length: int | None = None,
    ) -> None:
        if model_inputs_keys is None:
            model_inputs_keys = {"input_ids", "attention_mask", "token_type_ids"}
        super().__init__(tokenizer, padding=padding, max_length=max_length)
        self.model_inputs_keys: set[str] = model_inputs_keys

    def __call__(  # type: ignore[override]
        self, batch: list[dict[str, SentenceBatchSample | int | str]]
    ) -> dict[str, BatchEncoding | list[int] | list[str]]:
        first_sample: dict[str, SentenceBatchSample | int | str] = batch[0]
        actual_model_keys: set[str] = {
            k for k in self.model_inputs_keys if k in first_sample
        }

        model_inputs: list[SentenceBatchSample] = []
        reminder_inputs: dict[str, list[int | str]] = {
            k: [] for k in first_sample if k not in actual_model_keys
        }

        for sample in batch:
            model_inputs.append(
                cast(SentenceBatchSample, {k: sample[k] for k in actual_model_keys})
            )
            for k in reminder_inputs:
                reminder_inputs[k].append(sample[k])  # type: ignore[arg-type]

        return {"inputs": super().__call__(model_inputs)} | reminder_inputs
