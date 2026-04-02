"""Dataset iterators and factory functions for reranker training.

Provides:
- :class:`BioASQPointwiseIterator` — pointwise (query, doc, label) samples
- :class:`BioASQPairwiseIterator` — pairwise (pos, neg) samples
- :class:`BioASQMultiNegativePairwiseIterator` — 1 pos + N neg samples
- :class:`BioASQDataset` — IterableDataset wrapper with DDP support
- :class:`BioASQInferenceDataset` — Map-style dataset for inference/eval
- :class:`BioASQPairwiseEvalDataset` — Pairwise evaluation dataset

Factory functions:
- :func:`create_bioASQ_datasets` — build all datasets from data paths
- :func:`create_inference_dataset_from_bioasq_json` — build from JSON/JSONL

Refactored from ``refactored-trainer/data.py``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, cast

import msgspec
from torch.utils.data import Dataset, IterableDataset

from bioasq.common.aliases import (
    Collection,
    ProcessedSample,
    QrelsDict,
    Sample,
    SliceDataset,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from bioasq.phase_a.reranker.preprocessing import BasicSamplePreprocessing
    from bioasq.phase_a.reranker.sampler import BasicSampler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_relevance_order_from_dataset(dataset: SliceDataset) -> list[int]:
    """Extract sorted relevance levels from the first question."""
    if len(dataset) == 0:
        return []
    _sample = dataset[next(iter(dataset.keys()))]
    return sorted([k for k in _sample if isinstance(k, int)], reverse=True)


def _get_negative_positive_index(dataset: SliceDataset) -> tuple[int, int]:
    """Return (negative_index, positive_index) from dataset."""
    _sample = dataset[next(iter(dataset.keys()))]
    relevance_order: list[int] = [k for k in _sample if isinstance(k, int)]
    return min(relevance_order), max(relevance_order)


def _load_rank_data(
    bm25_rank_path: Path,
    at: int = 1000,
    qrels: QrelsDict | None = None,
) -> SliceDataset:
    """Load ranked data from JSONL file."""
    dataset: SliceDataset = {}
    with bm25_rank_path.open() as f:
        for line in f:
            q_data: dict[str, str | list[dict[str, str]]] = msgspec.json.decode(line)
            q_id: str = str(q_data["id"])
            if qrels and q_id not in qrels:
                continue
            docs: list[dict[str, str]] = q_data["documents"][:at]  # type: ignore[index]
            dataset[q_id] = {
                "documents": docs,
                "question": str(q_data["question"]),
            }
    return dataset


# ---------------------------------------------------------------------------
# Iterators
# ---------------------------------------------------------------------------


class BioASQPointwiseIterator:
    """Iterator yielding pointwise (query, doc, label) training samples.

    Each ``__next__`` call randomly selects a question, picks a positive
    or negative document, tokenises the pair, and returns a
    :data:`~bioasq.common.aliases.ProcessedSample`.
    """

    def __init__(
        self,
        sample_preprocessing: BasicSamplePreprocessing,
        sampler_class_type: type[BasicSampler],
        num_neg_samples: int = 1,
        sampler_kwargs: dict[str, object] | None = None,
    ) -> None:
        self.sample_preprocessing: BasicSamplePreprocessing = sample_preprocessing
        self.sampler_class_type: type[BasicSampler] = sampler_class_type
        self.num_neg_samples: int = num_neg_samples
        self.sampler_kwargs: dict[str, object] = sampler_kwargs or {}

        self._sampler: BasicSampler | None = None
        self._sample_index: int = 0
        self._epoch: int = 0
        self._total_samples: int = 0

    def attach_dataset(
        self,
        slice_dataset: SliceDataset,
        collection: Collection | None = None,
    ) -> None:
        """Bind a dataset to this iterator (called by BioASQDataset)."""
        self._sampler = self.sampler_class_type(
            slice_dataset, collection, **self.sampler_kwargs
        )
        self._total_samples = sum(
            sum(
                len(cast("list[dict[str, str]]", v))
                for k, v in q.items()
                if isinstance(k, int)
            )
            for q in slice_dataset.values()
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._sample_index = 0

    @property
    def total_samples(self) -> int:
        return self._total_samples

    def _apply_pointwise_sampler_preprocessing(
        self,
    ) -> ProcessedSample | None:
        """Sample a pointwise (query, doc, label) and tokenise."""
        assert self._sampler is not None
        q_id, q_text = self._sampler.choose_question(self._sample_index, self._epoch)

        is_positive: bool = random.random() > 0.5
        if is_positive:
            doc_text: str | None = self._sampler.choose_positive_doc(
                self._sample_index, self._epoch, q_id
            )
            label: int = 1
        else:
            doc_text = self._sampler.choose_negative_doc(
                self._sample_index, self._epoch, q_id
            )
            label = 0

        if doc_text is None:
            return None

        sample: Sample = {
            "id": q_id,
            "query_text": q_text,
            "doc_text": doc_text,
            "label": label,
        }
        return cast("ProcessedSample", self.sample_preprocessing(sample))

    def __next__(self) -> ProcessedSample:
        result: ProcessedSample | None = None
        while result is None:
            result = self._apply_pointwise_sampler_preprocessing()
            self._sample_index += 1
        return result


class BioASQPairwiseIterator(BioASQPointwiseIterator):
    """Iterator yielding pairwise (pos_inputs, neg_inputs) samples."""

    def __next__(self) -> ProcessedSample:
        result: ProcessedSample | None = None
        while result is None:
            assert self._sampler is not None
            q_id, q_text = self._sampler.choose_question(
                self._sample_index, self._epoch
            )
            pos_text, neg_text = self._sampler.choose_positive_and_negative_doc(
                self._sample_index, self._epoch, q_id
            )

            if pos_text is None or neg_text is None:
                self._sample_index += 1
                continue

            pos_sample: Sample = {
                "id": q_id,
                "query_text": q_text,
                "doc_text": pos_text,
                "label": 1,
            }
            neg_sample: Sample = {
                "id": q_id,
                "query_text": q_text,
                "doc_text": neg_text,
                "label": 0,
            }

            pos_processed: Sample = self.sample_preprocessing(pos_sample)
            neg_processed: Sample = self.sample_preprocessing(neg_sample)

            result = {
                "pos_inputs": pos_processed,
                "neg_inputs": neg_processed,
            }
            self._sample_index += 1
        return result


class BioASQMultiNegativePairwiseIterator:
    """Iterator yielding 1 pos + N neg samples per step.

    Parameters
    ----------
    num_neg_samples:
        Number of negative documents to pair with each positive.
    """

    def __init__(
        self,
        sample_preprocessing: BasicSamplePreprocessing,
        sampler_class: type[BasicSampler],
        num_neg_samples: int = 4,
        sampler_kwargs: dict[str, object] | None = None,
    ) -> None:
        self.sample_preprocessing: BasicSamplePreprocessing = sample_preprocessing
        self.sampler_class: type[BasicSampler] = sampler_class
        self.num_neg_samples: int = num_neg_samples
        self.sampler_kwargs: dict[str, object] = sampler_kwargs or {}

        self._sampler: BasicSampler | None = None
        self._sample_index: int = 0
        self._epoch: int = 0
        self._total_samples: int = 0

    def attach_dataset(
        self,
        slice_dataset: SliceDataset,
        collection: Collection | None = None,
    ) -> None:
        self._sampler = self.sampler_class(
            slice_dataset, collection, **self.sampler_kwargs
        )
        self._total_samples = sum(
            sum(
                len(cast("list[dict[str, str]]", v))
                for k, v in q.items()
                if isinstance(k, int)
            )
            for q in slice_dataset.values()
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._sample_index = 0

    @property
    def total_samples(self) -> int:
        return self._total_samples

    def __next__(self) -> ProcessedSample:
        assert self._sampler is not None
        result: ProcessedSample | None = None

        while result is None:
            q_id, q_text = self._sampler.choose_question(
                self._sample_index, self._epoch
            )
            pos_text: str | None = self._sampler.choose_positive_doc(
                self._sample_index, self._epoch, q_id
            )
            if pos_text is None:
                self._sample_index += 1
                continue

            neg_texts: list[str] = []
            for _ in range(self.num_neg_samples):
                neg_text: str | None = self._sampler.choose_negative_doc(
                    self._sample_index, self._epoch, q_id
                )
                if neg_text is not None:
                    neg_texts.append(neg_text)

            if len(neg_texts) < self.num_neg_samples:
                self._sample_index += 1
                continue

            pos_sample: Sample = {
                "id": q_id,
                "query_text": q_text,
                "doc_text": pos_text,
                "label": 1,
            }
            pos_processed: Sample = self.sample_preprocessing(pos_sample)

            neg_processed_list: list[Sample] = []
            for nt in neg_texts:
                neg_sample: Sample = {
                    "id": q_id,
                    "query_text": q_text,
                    "doc_text": nt,
                    "label": 0,
                }
                neg_processed_list.append(self.sample_preprocessing(neg_sample))

            result = {
                "pos_inputs": pos_processed,
                "neg_inputs": neg_processed_list,
            }
            self._sample_index += 1
        return result


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

_IteratorType = (
    BioASQPointwiseIterator
    | BioASQPairwiseIterator
    | BioASQMultiNegativePairwiseIterator
)


class BioASQDataset(IterableDataset[ProcessedSample]):
    """Wraps an iterator as an :class:`~torch.utils.data.IterableDataset`.

    Handles epoch tracking and DDP-aware data splitting via
    ``torch.utils.data.get_worker_info()``.
    """

    def __init__(
        self,
        iterator: _IteratorType,
        slice_dataset: SliceDataset,
        collection: Collection | None = None,
    ) -> None:
        super().__init__()
        self.iterator: _IteratorType = iterator
        self.slice_dataset: SliceDataset = slice_dataset
        self.collection: Collection | None = collection
        self._epoch: int = 0

        iterator.attach_dataset(slice_dataset, collection)

    def __len__(self) -> int:
        return self.iterator.total_samples

    def __iter__(self) -> Iterator[ProcessedSample]:
        self.iterator.set_epoch(self._epoch)
        for _ in range(len(self)):
            yield next(self.iterator)
        self._epoch += 1


class BioASQInferenceDataset(Dataset[ProcessedSample]):
    """Map-style dataset for inference and evaluation.

    Pre-processes all samples at init time for random access by index.
    """

    def __init__(
        self,
        samples: list[Sample],
        sample_preprocessing: BasicSamplePreprocessing,
        qrels: QrelsDict | None = None,
    ) -> None:
        self._samples: list[ProcessedSample] = []
        self._qrels: QrelsDict = qrels or {}

        for s in samples:
            processed: Sample = sample_preprocessing(s)
            if qrels is not None and s.get("id") and s.get("doc_id"):
                q_id: str = str(s["id"])
                d_id: str = str(s["doc_id"])
                if q_id in qrels and d_id in qrels[q_id]:
                    processed["labels"] = qrels[q_id][d_id]
                else:
                    processed["labels"] = 0
            self._samples.append(cast("ProcessedSample", processed))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> ProcessedSample: # type: ignore[override]
        return self._samples[idx]

    def get_qrels(self) -> QrelsDict:
        return self._qrels

    def __repr__(self) -> str:
        return f"BioASQInferenceDataset(n={len(self._samples)})"


class BioASQPairwiseEvalDataset(Dataset[ProcessedSample]):
    """Pairwise evaluation dataset.

    Builds (pos, neg) pairs from qrels and preprocesses them.
    """

    def __init__(
        self,
        slice_dataset: SliceDataset,
        sample_preprocessing: BasicSamplePreprocessing,
        is_multi_negative: bool = False,
        num_neg_samples: int = 4,
    ) -> None:
        self._samples: list[ProcessedSample] = []
        self._build_samples(
            slice_dataset, sample_preprocessing, is_multi_negative, num_neg_samples
        )

    def _build_samples(
        self,
        slice_dataset: SliceDataset,
        sample_preprocessing: BasicSamplePreprocessing,
        is_multi_negative: bool,
        num_neg_samples: int,
    ) -> None:
        neg_idx, pos_idx = _get_negative_positive_index(slice_dataset)

        for q_id, q_data in slice_dataset.items():
            q_text: str = str(q_data["question"])
            pos_docs: list[dict[str, str]] = cast(
                "list[dict[str, str]]", q_data.get(pos_idx, [])
            )
            neg_docs: list[dict[str, str]] = cast(
                "list[dict[str, str]]", q_data.get(neg_idx, [])
            )

            if not pos_docs or not neg_docs:
                continue

            for pos_doc in pos_docs:
                pos_sample: Sample = {
                    "id": q_id,
                    "query_text": q_text,
                    "doc_text": pos_doc.get("text", ""),
                    "label": 1,
                }
                pos_processed: Sample = sample_preprocessing(pos_sample)

                if is_multi_negative:
                    selected_negs: list[dict[str, str]] = random.sample(
                        neg_docs, min(num_neg_samples, len(neg_docs))
                    )
                    neg_processed_list: list[Sample] = []
                    for neg_doc in selected_negs:
                        neg_sample: Sample = {
                            "id": q_id,
                            "query_text": q_text,
                            "doc_text": neg_doc.get("text", ""),
                            "label": 0,
                        }
                        neg_processed_list.append(sample_preprocessing(neg_sample))
                    self._samples.append(
                        {
                            "pos_inputs": pos_processed,
                            "neg_inputs": neg_processed_list,
                        }
                    )
                else:
                    neg_doc: dict[str, str] = random.choice(neg_docs)
                    neg_s: Sample = {
                        "id": q_id,
                        "query_text": q_text,
                        "doc_text": neg_doc.get("text", ""),
                        "label": 0,
                    }
                    neg_processed: Sample = sample_preprocessing(neg_s)
                    self._samples.append(
                        {
                            "pos_inputs": pos_processed,
                            "neg_inputs": neg_processed,
                        }
                    )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> ProcessedSample: # type: ignore[override]
        return self._samples[idx]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _normalize_doc(doc: dict[str, str]) -> dict[str, str]:
    """Normalise document dict to ``{"id": ..., "text": ...}`` format.

    Handles alternative key names (``pmid`` → ``id``, ``full_text`` → ``text``).
    """
    return {
        "id": str(doc.get("id") or doc.get("pmid", "")),
        "text": str(doc.get("text") or doc.get("full_text", "")),
    }


def _get_docs_key(question: dict[str, str | list[dict[str, str]]]) -> str | None:
    """Identify the key containing document candidates."""
    for key in ("documents", "neg_docs", "bm25"):
        if question.get(key):
            return key
    return None


def create_bioasq_datasets(
    positive_data_path: Path,
    all_data_path: Path,
    iterator: _IteratorType,
    test_sample_preprocessing: BasicSamplePreprocessing,
    val_files: list[Path] | None = None,
    relevance_mapping: dict[str, int] | None = None,
    collection: Collection | None = None,
) -> tuple[
    BioASQDataset,
    BioASQInferenceDataset,
    BioASQPairwiseEvalDataset | None,
    BioASQPairwiseEvalDataset | None,
    BioASQPairwiseEvalDataset | None,
]:
    """Build all datasets from data paths.

    Parameters
    ----------
    positive_data_path:
        JSONL with id, body, documents (positives).
    all_data_path:
        JSONL with id, neg_docs (BM25 negatives).
    iterator:
        The training iterator to bind.
    test_sample_preprocessing:
        Preprocessor for test/eval data.
    val_files:
        Optional golden JSON files for validation splitting.
    relevance_mapping:
        Maps doc key names to relevance levels.
    collection:
        Optional external doc collection.

    Returns
    -------
    ``(train_dataset, test_dataset, eval_pointwise, eval_pairwise, eval_multi_neg)``
    """
    relevance_mapping = relevance_mapping or {"documents": 1}

    # Load positive data
    positive_data: dict[str, dict[str, str | list[dict[str, str]]]] = {}
    with positive_data_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            q: dict[str, str | list[dict[str, str]]] = msgspec.json.decode(line)
            positive_data[str(q["id"])] = q

    # Load all data (with negatives)
    all_data: dict[str, dict[str, str | list[dict[str, str]]]] = {}
    with all_data_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            q = msgspec.json.decode(line)
            all_data[str(q["id"])] = q

    # Build slice dataset
    slice_dataset: SliceDataset = {}
    qrels_test: QrelsDict = {}

    # Determine val_qids for test split
    val_qids: set[str] = set()
    if val_files:
        for vf in val_files:
            with vf.open() as f:
                data: dict[str, list[dict[str, str]]] = msgspec.json.decode(f.read())
            for qq in data.get("questions", []):
                val_qids.add(str(qq["id"]))

    for q_id, q_data in all_data.items():
        pos_q: dict[str, str | list[dict[str, str]]] | None = positive_data.get(q_id)
        if pos_q is None:
            continue

        entry: dict[str | int, list[dict[str, str]] | str] = {
            "question": str(pos_q.get("body", pos_q.get("question", ""))),
        }

        # Add positives by relevance level
        for doc_key, relevance in relevance_mapping.items():
            docs: list[dict[str, str]] = pos_q.get(doc_key, [])  # type: ignore[assignment]
            if relevance not in entry:
                entry[relevance] = []
            docs_list: list[dict[str, str]] = cast(
                "list[dict[str, str]]", entry[relevance]
            )
            docs_list.extend(docs)

        # Add negatives (level 0) – normalise keys to {id, text}
        raw_neg_docs: list[dict[str, str]] = q_data.get("neg_docs", [])  # type: ignore[assignment]
        neg_docs: list[dict[str, str]] = [_normalize_doc(d) for d in raw_neg_docs]
        if 0 not in entry:
            entry[0] = []
        neg_list: list[dict[str, str]] = cast("list[dict[str, str]]", entry[0])
        neg_list.extend(neg_docs)

        if val_qids and q_id in val_qids:
            # This question is for testing
            pos_list_for_test: list[dict[str, str]] = cast(
                "list[dict[str, str]]", entry.get(1, [])
            )
            for doc in pos_list_for_test:
                if q_id not in qrels_test:
                    qrels_test[q_id] = {}
                qrels_test[q_id][str(doc.get("id", ""))] = 1
            neg_list_for_test: list[dict[str, str]] = cast(
                "list[dict[str, str]]", entry.get(0, [])
            )
            for doc in neg_list_for_test:
                if q_id not in qrels_test:
                    qrels_test[q_id] = {}
                qrels_test[q_id][str(doc.get("id", ""))] = 0
        else:
            slice_dataset[q_id] = entry

    # Build train dataset
    train_dataset: BioASQDataset = BioASQDataset(
        iterator=iterator,
        slice_dataset=slice_dataset,
        collection=collection,
    )

    # Build test dataset
    test_samples: list[Sample] = []
    for q_id in qrels_test:
        q_data_raw = all_data.get(q_id, {})
        pos_q_raw = positive_data.get(q_id, {})
        q_text: str = str(pos_q_raw.get("body", pos_q_raw.get("question", "")))

        pos_docs_list: list[dict[str, str]] = pos_q_raw.get("documents", [])  # type: ignore[assignment]
        neg_docs_list: list[dict[str, str]] = q_data_raw.get("neg_docs", [])  # type: ignore[assignment]
        all_docs: list[dict[str, str]] = pos_docs_list + neg_docs_list
        for doc in all_docs:
            test_samples.append(
                {
                    "id": q_id,
                    "doc_id": str(doc.get("id", "")),
                    "query_text": q_text,
                    "doc_text": doc.get("text", ""),
                }
            )

    test_dataset: BioASQInferenceDataset = BioASQInferenceDataset(
        samples=test_samples,
        sample_preprocessing=test_sample_preprocessing,
        qrels=qrels_test,
    )

    # Build eval datasets
    eval_slice: SliceDataset | None = None
    if val_qids:
        eval_slice = {}
        for q_id in val_qids:
            if q_id in all_data:
                pos_q_raw = positive_data.get(q_id, {})
                eval_entry: dict[str | int, list[dict[str, str]] | str] = {
                    "question": str(
                        pos_q_raw.get("body", pos_q_raw.get("question", ""))
                    ),
                }
                for doc_key, relevance in relevance_mapping.items():
                    if relevance not in eval_entry:
                        eval_entry[relevance] = []
                    rel_list: list[dict[str, str]] = cast(
                        "list[dict[str, str]]", eval_entry[relevance]
                    )
                    rel_list.extend(pos_q_raw.get(doc_key, []))  # type: ignore[arg-type]
                if 0 not in eval_entry:
                    eval_entry[0] = []
                neg_eval_list: list[dict[str, str]] = cast(
                    "list[dict[str, str]]", eval_entry[0]
                )
                neg_eval_list.extend(all_data[q_id].get("neg_docs", []))  # type: ignore[arg-type]
                eval_slice[q_id] = eval_entry

    eval_pointwise: BioASQPairwiseEvalDataset | None = None
    eval_pairwise: BioASQPairwiseEvalDataset | None = None
    eval_multi_neg: BioASQPairwiseEvalDataset | None = None

    if eval_slice:
        eval_pairwise = BioASQPairwiseEvalDataset(eval_slice, test_sample_preprocessing)
        eval_multi_neg = BioASQPairwiseEvalDataset(
            eval_slice, test_sample_preprocessing, is_multi_negative=True
        )

    return train_dataset, test_dataset, eval_pointwise, eval_pairwise, eval_multi_neg


def create_inference_dataset_from_bioasq_json(
    questions_path: str | Path,
    sample_preprocessing: BasicSamplePreprocessing,
    max_docs: int = 100,
) -> BioASQInferenceDataset:
    """Create an inference dataset from a BioASQ JSON or JSONL file.

    Parameters
    ----------
    questions_path:
        Path to JSON (``{"questions": [...]}`` format) or JSONL.
    sample_preprocessing:
        Tokeniser-based preprocessor.
    max_docs:
        Maximum document candidates per question.
    """
    questions_path = Path(questions_path)
    questions: list[dict[str, str | list[dict[str, str]]]] = []

    with questions_path.open() as f:
        content: str = f.read().strip()
        try:
            data: dict[str, list[dict[str, str | list[dict[str, str]]]]] = json.loads(
                content
            )
            questions = data.get("questions", [])
        except json.JSONDecodeError:
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    questions.append(json.loads(line))

    samples: list[Sample] = []
    for q in questions:
        q_id: str = str(q["id"])
        q_text: str = str(q.get("body", q.get("question", "")))

        docs_key: str | None = _get_docs_key(q)
        if docs_key is None:
            continue

        docs: list[dict[str, str]] = q[docs_key][:max_docs]  # type: ignore[index]
        for doc in docs:
            samples.append(
                {
                    "id": q_id,
                    "doc_id": str(doc.get("id", "")),
                    "query_text": q_text,
                    "doc_text": doc.get("text", ""),
                }
            )

    return BioASQInferenceDataset(
        samples=samples, sample_preprocessing=sample_preprocessing
    )
