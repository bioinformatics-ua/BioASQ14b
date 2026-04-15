from typing import Any, cast

from transformers import BatchEncoding, PreTrainedTokenizerBase

SentenceBatchSample = dict[str, list[list[int]]]


class RankingCollator:
    """Collator for single-encoding-per-sample ranking tasks.

    Each sample is one (query, document) pair already tokenized. Pads model
    inputs (input_ids, attention_mask, token_type_ids) and passes through any
    metadata (labels, doc_id, query_id, etc.) unchanged.

    Use for evaluation (score many docs per query) or pointwise training
    (each sample has a relevance label).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model_inputs: set[str] | None = None,
        padding: bool | str = True,
        max_length: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model_inputs = model_inputs
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, BatchEncoding | list[Any]]:
        grouped: dict[str, list[Any]] = {key: [i[key] for i in batch] for key in batch[0]}

        # Auto-detect model inputs if not specified
        if self.model_inputs is None:
            standard_inputs = {"input_ids", "attention_mask", "token_type_ids"}
            present_inputs = set(grouped.keys()) & standard_inputs
            # Must have at least input_ids and attention_mask
            model_inputs = present_inputs
        else:
            model_inputs = self.model_inputs

        reminder_keys = set(grouped.keys()) - model_inputs
        return {
            "inputs": self.tokenizer.pad(
                {k: grouped[k] for k in model_inputs},
                padding=self.padding,
                max_length=self.max_length,
                return_tensors="pt",
            )
        } | {k: grouped[k] for k in reminder_keys}


_MODEL_INPUT_KEYS = {"input_ids", "attention_mask", "token_type_ids"}


def _filter_model_inputs(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only tensorizable keys (input_ids, attention_mask, token_type_ids) for tokenizer.pad."""
    return [{k: v for k, v in s.items() if k in _MODEL_INPUT_KEYS} for s in samples]


class PairwiseCollator:
    """Collator for pairwise training with (positive, negative) document pairs.

    Each sample has ``pos_inputs`` and ``neg_inputs`` (already tokenized).
    Pads each separately and returns two batched tensors for contrastive
    or margin loss training.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, BatchEncoding]:
        pos_list = _filter_model_inputs([s["pos_inputs"] for s in batch])
        neg_list = _filter_model_inputs([s["neg_inputs"] for s in batch])
        return {
            "pos_inputs": self.tokenizer.pad(pos_list, padding=True, return_tensors="pt"),
            "neg_inputs": self.tokenizer.pad(neg_list, padding=True, return_tensors="pt"),
        }


class MultiNegativePairwiseCollator:
    """Collator for BioASQMultiNegativePairwiseIterator: 1 positive + N negatives per sample.

    Each sample has ``pos_inputs`` (single dict) and ``neg_inputs`` (list of N dicts).
    Pads positives to one batch [batch_size, seq_len], and negatives to a LIST of
    batches: neg_inputs[i] = batch of the i-th negative from each sample [batch_size, seq_len].

    Usage:
        scores_pos = model(**pos_inputs)
        scores_negs = [model(**neg_i) for neg_i in neg_inputs]
        # loss over all pairs: sum(margin_loss(scores_pos, neg_i) for neg_i in scores_negs)
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> dict[str, BatchEncoding | list[BatchEncoding]]:
        # pos_inputs: list of dicts, one per sample (filter to model keys only)
        pos_list = _filter_model_inputs([sample["pos_inputs"] for sample in batch])
        pos_inputs = self.tokenizer.pad(pos_list, padding=True, return_tensors="pt")

        # neg_inputs: list of lists. Transpose so neg_inputs[j] = j-th neg from each sample
        neg_inputs_raw: list[list[dict[str, Any]]] = [sample["neg_inputs"] for sample in batch]
        num_negs = len(neg_inputs_raw[0])
        neg_batches: list[BatchEncoding] = []
        for j in range(num_negs):
            neg_j = _filter_model_inputs([neg_inputs_raw[i][j] for i in range(len(batch))])
            neg_batches.append(self.tokenizer.pad(neg_j, padding=True, return_tensors="pt"))

        return {"pos_inputs": pos_inputs, "neg_inputs": neg_batches}


class RankingCollatorForCasualLM(RankingCollator):
    """Ranking collator for decoder-only (causal LM) models, e.g. GPT-style.

    Uses only ``input_ids`` and ``attention_mask``; no token_type_ids.
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
    """Ranking collator for encoder-decoder models, e.g. T5, BART.

    Includes ``decoder_input_ids`` in model inputs alongside encoder fields.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model_inputs: set[str] | None = None,
    ) -> None:
        if model_inputs is None:
            model_inputs = {"input_ids", "attention_mask", "decoder_input_ids"}
        super().__init__(tokenizer, model_inputs=model_inputs)


class SentenceCollator:
    """Collator for samples with multiple sentences per document.

    Each sample has lists of encodings (e.g. doc split into chunks). Flattens
    all sentences across the batch into one padded tensor and tracks
    ``sentences_count`` per original sample for reconstruction.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: bool | str = True,
        max_length: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch: list[SentenceBatchSample]) -> BatchEncoding:
        # Dynamically detect keys from the first sample (token_type_ids may not exist)
        first_sample = batch[0]
        standard_keys = {"input_ids", "attention_mask", "token_type_ids"}
        present_keys = [k for k in standard_keys if k in first_sample]

        expanded_batch: dict[str, list[list[int]]] = {k: [] for k in present_keys}

        sentences_count: list[int] = []

        for sample in batch:
            for k in expanded_batch.keys():
                expanded_batch[k].extend(sample[k])
            sentences_count.append(len(sample["input_ids"]))

        expanded_batch_padded = self.tokenizer.pad(
            expanded_batch,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # batch = {key: [i[key] for i in batch] for key in batch[0]}
        expanded_batch_padded["sentences_count"] = sentences_count

        return expanded_batch_padded


class PairwiseSentenceCollator(SentenceCollator):
    """Pairwise training with multi-sentence documents.

    Applies SentenceCollator separately to positive and negative inputs.
    Use when each pos/neg document is split into multiple sentences/chunks.
    """

    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride, reportImplicitOverride]
        self, batch: list[dict[str, SentenceBatchSample]]
    ) -> dict[str, BatchEncoding]:  # ty:ignore[invalid-method-override]
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
    """Sentence-level collator that preserves metadata like RankingCollator.

    Combines multi-sentence flattening with passthrough of non-model fields
    (labels, doc_id, query_id, etc.). Use for ranking with long documents
    split into multiple chunks.
    """

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

    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride, reportImplicitOverride]
        self, batch: list[dict[str, SentenceBatchSample]]
    ) -> dict[str, BatchEncoding | list[dict[str, list[list[int]]]]]:  # ty:ignore[invalid-method-override]
        # Filter model_inputs_keys to only include keys present in the batch
        # (token_type_ids may not be present for some tokenizers)
        first_sample = batch[0]
        actual_model_keys = {k for k in self.model_inputs_keys if k in first_sample}

        model_inputs: list[SentenceBatchSample] = []
        reminder_inputs: dict[str, list[dict[str, list[list[int]]]]] = {
            k: [] for k in first_sample.keys() if k not in actual_model_keys
        }

        for sample in batch:
            model_inputs.append(
                cast(SentenceBatchSample, {k: sample[k] for k in actual_model_keys})
            )

            for k in reminder_inputs.keys():
                reminder_inputs[k].append(sample[k])

        return {"inputs": super().__call__(model_inputs)} | reminder_inputs
