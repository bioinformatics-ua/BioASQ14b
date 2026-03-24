"""Data sampling strategies for reranker training.

Samplers control how training examples are drawn from the
:data:`~bioasq.common.aliases.SliceDataset` — which questions to pick,
and which positive / negative documents to pair for contrastive learning.

Hierarchy::

    BasicSampler
    ├── EmptySampler          (no-op, testing)
    ├── BasicV2Sampler        (fallback to other questions' negatives)
    ├── ExponentialWeightSampler  (weighted by relevance level)
    ├── HigherConfidenceNegativesSampler  (skip top BM25 negatives)
    └── ShifterSampler        (curriculum learning: easy → hard negatives)

Refactored from ``refactored-trainer/sampler.py``.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from bioasq.common.aliases import Collection, SliceDataset


def _get_relevance_order_from_dataset(dataset: SliceDataset) -> list[int]:
    """Extract sorted relevance levels from the first question in the dataset."""
    if len(dataset) == 0:
        return []
    _sample: dict[str | int, list[dict[str, str]] | str] = dataset[
        next(iter(dataset.keys()))
    ]
    relevance_order: list[int] = sorted(
        [k for k in _sample if isinstance(k, int)], reverse=True
    )
    return relevance_order


# ---------------------------------------------------------------------------
# Base sampler
# ---------------------------------------------------------------------------


class BasicSampler:
    """Default sampler: uniformly random question, random pos/neg docs."""

    def __init__(
        self,
        slice_dataset: SliceDataset,
        collection: Collection | None = None,
    ) -> None:
        self.slice_dataset: SliceDataset = slice_dataset
        self.collection: Collection | None = collection
        self.q_ids: list[str] = list(self.slice_dataset.keys())

        self.relevance_order: list[int] = _get_relevance_order_from_dataset(
            self.slice_dataset
        )

        self.negative_index: int = min(self.relevance_order)
        self.positive_index: int = max(self.relevance_order)

    def _lookup_doc(self, document: dict[str, str]) -> str:
        """Resolve document text, either from collection or inline."""
        if self.collection:
            pmid: str = document["id"]
            return self.collection[pmid]
        return document["text"]

    def choose_question(self, _sample_index: int, _epoch: int) -> tuple[str, str]:
        """Select a random question (id, text)."""
        q_id: str = random.choice(self.q_ids)
        q_text: str = str(self.slice_dataset[q_id]["question"])
        return q_id, q_text

    def choose_positive_doc(
        self, _sample_index: int, _epoch: int, q_id: str
    ) -> str | None:
        """Select a random positive document for the given question."""
        valid_sample_groups: list[int] = [
            ro
            for ro in self.relevance_order
            if len(self.slice_dataset[q_id][ro]) > 0 and ro != self.negative_index
        ]
        pos_index: int = random.choice(valid_sample_groups)

        docs: list[dict[str, str]] | str = self.slice_dataset[q_id][pos_index]
        doc: dict[str, str] = cast("dict[str, str]", random.choice(docs))
        return self._lookup_doc(doc)

    def choose_negative_doc(
        self, _sample_index: int, _epoch: int, q_id: str
    ) -> str | None:
        """Select a random negative document for the given question."""
        neg_docs: list[dict[str, str]] = cast(
            "list[dict[str, str]]", self.slice_dataset[q_id][self.negative_index]
        )
        if len(neg_docs) == 0:
            return None
        doc: dict[str, str] = random.choice(neg_docs)
        return self._lookup_doc(doc)

    def choose_positive_and_negative_doc(
        self, _sample_index: int, _epoch: int, q_id: str
    ) -> tuple[str | None, str | None]:
        """Select a random positive and negative document pair."""
        valid_sample_groups: list[list[dict[str, str]]] = [
            cast("list[dict[str, str]]", self.slice_dataset[q_id][ro])
            for ro in self.relevance_order
            if len(self.slice_dataset[q_id][ro]) > 0
        ]

        if len(valid_sample_groups) < 2:
            print(
                f"Warning question ({q_id}) only contains "
                f"{len(valid_sample_groups)} valid groups for sampling.",
                flush=True,
            )
            return None, None

        pos_doc_index: int = random.randrange(len(valid_sample_groups[:-1]))
        pos_doc: dict[str, str] = random.choice(valid_sample_groups[pos_doc_index])
        pos_doc_text: str = self._lookup_doc(pos_doc)

        neg_doc_list: list[dict[str, str]] = random.choice(
            valid_sample_groups[pos_doc_index + 1 :]
        )
        neg_doc: dict[str, str] = random.choice(neg_doc_list)
        neg_doc_text: str = self._lookup_doc(neg_doc)

        return pos_doc_text, neg_doc_text


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


class EmptySampler(BasicSampler):
    """No-op sampler for testing / placeholder usage."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def choose_question(self, _sample_index: int, _epoch: int) -> tuple[str, str]:
        return "", ""

    def choose_positive_doc(
        self, _sample_index: int, _epoch: int, _q_id: str
    ) -> str | None:
        return None


class BasicV2Sampler(BasicSampler):
    """Sampler that falls back to negatives from other questions if empty."""

    def get_negatives_from_another_question(self, _q_id: str) -> dict[str, str]:
        """Select a random negative from a different question that has them."""
        while True:
            new_q_id: str = random.choice(self.q_ids)
            valid_sample_groups: list[list[dict[str, str]] | str] = [
                self.slice_dataset[new_q_id][ro]
                for ro in self.relevance_order
                if len(self.slice_dataset[new_q_id][ro]) > 0
            ]
            if len(valid_sample_groups) >= 2:
                break

        neg_docs: list[dict[str, str]] = cast(
            "list[dict[str, str]]",
            self.slice_dataset[new_q_id][self.negative_index],
        )
        return random.choice(neg_docs)

    def choose_negative_doc(
        self, _sample_index: int, _epoch: int, q_id: str
    ) -> str | None:
        neg_docs: list[dict[str, str]] = cast(
            "list[dict[str, str]]", self.slice_dataset[q_id][self.negative_index]
        )
        if len(neg_docs) == 0:
            return self._lookup_doc(self.get_negatives_from_another_question(q_id))
        doc: dict[str, str] = random.choice(neg_docs)
        return self._lookup_doc(doc)

    def choose_positive_and_negative_doc(
        self, _sample_index: int, _epoch: int, q_id: str
    ) -> tuple[str | None, str | None]:
        valid_sample_groups: list[list[dict[str, str]] | str] = [
            self.slice_dataset[q_id][ro]
            for ro in self.relevance_order
            if len(self.slice_dataset[q_id][ro]) > 0
        ]

        if len(valid_sample_groups) < 2:
            pos_doc_text: str = self._lookup_doc(
                cast("dict[str, str]", random.choice(valid_sample_groups[0]))
            )
            neg_doc_text: str = self._lookup_doc(
                self.get_negatives_from_another_question(q_id)
            )
        else:
            pos_doc_index: int = random.randrange(len(valid_sample_groups[:-1]))
            pos_doc_text = self._lookup_doc(
                cast(
                    "dict[str, str]", random.choice(valid_sample_groups[pos_doc_index])
                )
            )
            neg_doc_list: list[dict[str, str]] | str = random.choice(
                valid_sample_groups[pos_doc_index + 1 :]
            )
            neg_doc_text = self._lookup_doc(
                cast("dict[str, str]", random.choice(neg_doc_list))
            )

        return pos_doc_text, neg_doc_text


class ExponentialWeightSampler(BasicSampler):
    """Sampler with exponential weighting by relevance level."""

    def choose_positive_and_negative_doc(
        self, _sample_index: int, _epoch: int, q_id: str
    ) -> tuple[str | None, str | None]:
        valid_sample_groups_with_ro: list[tuple[int, list[dict[str, str]] | str]] = [
            (ro, self.slice_dataset[q_id][ro])
            for ro in self.relevance_order
            if len(self.slice_dataset[q_id][ro]) > 0
        ]

        valid_ro, valid_groups = zip(*valid_sample_groups_with_ro, strict=False)
        valid_ro = list(valid_ro)
        valid_groups = list(valid_groups)

        if len(valid_groups) < 2:
            print(
                f"Warning question ({q_id}) only contains "
                f"{len(valid_groups)} valid groups for sampling.",
                flush=True,
            )
            return None, None

        weights: list[int] = [2**x for x in valid_ro]
        pos_doc_index: int = random.choices(
            range(len(valid_groups[:-1])), weights=weights[:-1], k=1
        )[0]
        pos_doc_list: list[dict[str, str]] = cast(
            "list[dict[str, str]]", valid_groups[pos_doc_index]
        )
        pos_doc: dict[str, str] = random.choice(pos_doc_list)
        pos_doc_text: str = self._lookup_doc(pos_doc)

        inverse_weights: list[int] = [5 - x for x in valid_ro[pos_doc_index + 1 :]]
        neg_doc_list: list[dict[str, str]] = cast(
            "list[dict[str, str]]",
            random.choices(
                valid_groups[pos_doc_index + 1 :], weights=inverse_weights, k=1
            )[0],
        )
        neg_doc: dict[str, str] = random.choice(neg_doc_list)
        neg_doc_text: str = self._lookup_doc(neg_doc)

        return pos_doc_text, neg_doc_text


class HigherConfidenceNegativesSampler(BasicSampler):
    """Sampler that skips the top BM25 negatives (first 10)."""

    def choose_negative_doc(
        self, _sample_index: int, _epoch: int, q_id: str
    ) -> str | None:
        neg_docs: list[dict[str, str]] = cast(
            "list[dict[str, str]]", self.slice_dataset[q_id]["neg_docs"]
        )
        if len(neg_docs) <= 10:
            return None
        if self.collection:
            neg_pmid: str = random.choice(neg_docs[10:])["id"]
            return self.collection[neg_pmid]
        return random.choice(neg_docs[10:])["text"]


class ShifterSampler(BasicSampler):
    """Curriculum learning sampler: easy → hard negatives over epochs.

    Implements a shrinking-window strategy where:

    - Early epochs: sample from all negatives ``[0:N]``
    - Final epochs: sample only from hardest negatives ``[0:K]``

    Parameters
    ----------
    max_epoch:
        Total number of training epochs (determines window shrink rate).
    """

    def __init__(
        self,
        slice_dataset: SliceDataset,
        collection: Collection | None,
        *args: object,
        **kwargs: object,
    ) -> None:
        max_epoch: int = int(kwargs.pop("max_epoch", 10))
        super().__init__(slice_dataset, collection, *args, **kwargs)
        self.max_epoch: int = max_epoch

    def choose_negative_doc(
        self, _sample_index: int, epoch: int, q_id: str
    ) -> str | None:
        """Sample negative with shrinking window toward hardest negatives."""
        neg_docs: list[dict[str, str]] = cast(
            "list[dict[str, str]]", self.slice_dataset[q_id][self.negative_index]
        )
        if len(neg_docs) == 0:
            return None

        interval: int = len(neg_docs) // (self.max_epoch + 1)
        end_pos: int = len(neg_docs) - (interval * epoch)

        min_pool_size: int = min(10, len(neg_docs))
        end_pos = max(end_pos, min_pool_size)

        doc: dict[str, str] = random.choice(neg_docs[:end_pos])
        return self._lookup_doc(doc)

    def choose_positive_doc(
        self, sample_index: int, epoch: int, q_id: str
    ) -> str | None:
        return super().choose_positive_doc(sample_index, epoch, q_id)
