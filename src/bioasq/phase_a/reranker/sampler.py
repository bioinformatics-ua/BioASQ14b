import random
from typing import cast

from bioasq.phase_a.reranker.aliases import Collection, SliceDataset
from bioasq.phase_a.reranker.utils import get_relevance_order_from_dataset


class BasicSampler:
    def __init__(
        self,
        slice_dataset: SliceDataset,
        collection: Collection | None = None,
        *args,
        **kwargs,
    ):
        self.slice_dataset: SliceDataset = slice_dataset
        self.collection: Collection | None = collection
        self.q_ids: list[str] = list(self.slice_dataset.keys())

        self.relevance_order: list[int] = get_relevance_order_from_dataset(self.slice_dataset)

        self.negative_index: int = min(self.relevance_order)
        self.positive_index: int = max(self.relevance_order)

    # dict str -> str | int
    def _lookup_doc(self, document: dict[str, str]) -> str:
        if self.collection:
            pmid = document["id"]
            return self.collection[pmid]
        else:
            return document["text"]

    def choose_question(self, sample_index: int, epoch: int) -> tuple[str, str]:
        q_id = random.choice(self.q_ids)
        q_text = str(self.slice_dataset[q_id]["question"])
        return q_id, q_text

    def choose_positive_doc(self, sample_index: int, epoch: int, q_id: str) -> str | None:
        valid_sample_groups = [
            ro
            for ro in self.relevance_order
            if len(self.slice_dataset[q_id][ro]) > 0 and ro != self.negative_index
        ]
        pos_index = random.choice(valid_sample_groups)

        docs = self.slice_dataset[q_id][pos_index]
        doc: dict[str, str] = cast(dict[str, str], random.choice(docs))
        return self._lookup_doc(doc)

    def choose_negative_doc(self, sample_index: int, epoch: int, q_id: str) -> str | None:
        neg_docs: list[dict[str, str]] = cast(
            list[dict[str, str]], self.slice_dataset[q_id][self.negative_index]
        )  # type: ignore

        if len(neg_docs) == 0:
            return None

        doc: dict[str, str] = random.choice(neg_docs)
        return self._lookup_doc(doc)

    def choose_positive_and_negative_doc(
        self, sample_index: int, epoch: int, q_id: str
    ) -> tuple[str | None, str | None]:
        valid_sample_groups: list[list[dict[str, str]]] = [
            cast(list[dict[str, str]], self.slice_dataset[q_id][ro])
            for ro in self.relevance_order
            if len(self.slice_dataset[q_id][ro]) > 0
        ]

        if len(valid_sample_groups) < 2:
            valid_sample_groups_ro = [
                ro for ro in self.relevance_order if len(self.slice_dataset[q_id][ro]) > 0
            ]
            # its impossible to sample from this questions
            print(
                f"Warning question ({q_id}) only contain ({len(valid_sample_groups_ro)}) valid_sample_groups ({valid_sample_groups_ro}) groups for sampling.",
                flush=True,
            )
            return None, None

        pos_doc_index = random.randrange(len(valid_sample_groups[:-1]))
        pos_doc: dict[str, str] = random.choice(valid_sample_groups[pos_doc_index])
        pos_doc_text = self._lookup_doc(pos_doc)

        neg_doc_list = random.choice(valid_sample_groups[pos_doc_index + 1 :])
        neg_doc: dict[str, str] = random.choice(neg_doc_list)
        neg_doc_text = self._lookup_doc(neg_doc)

        return pos_doc_text, neg_doc_text


class EmptySampler(BasicSampler):
    def __init__(self, *args, **kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType, reportMissingSuperCall]
        pass

    def choose_question(self, sample_index: int, epoch: int) -> tuple[str, str]:  # pyright: ignore[reportImplicitOverride]
        return "", ""

    def choose_positive_doc(self, sample_index: int, epoch: int, q_id: str) -> str | None:  # pyright: ignore[reportImplicitOverride]
        return None


class BasicV2Sampler(BasicSampler):
    def get_negatives_from_another_question(self, q_id: str) -> dict[str, str]:
        # select a random question with negs
        while True:
            new_q_id = random.choice(self.q_ids)

            valid_sample_groups = [
                self.slice_dataset[new_q_id][ro]
                for ro in self.relevance_order
                if len(self.slice_dataset[new_q_id][ro]) > 0
            ]

            if len(valid_sample_groups) >= 2:
                break

        neg_docs: list[dict[str, str]] = cast(
            list[dict[str, str]], self.slice_dataset[new_q_id][self.negative_index]
        )
        return random.choice(neg_docs)

    def choose_negative_doc(self, sample_index: int, epoch: int, q_id: str) -> str | None:
        neg_docs: list[dict[str, str]] = cast(
            list[dict[str, str]], self.slice_dataset[q_id][self.negative_index]
        )

        if len(neg_docs) == 0:
            return self._lookup_doc(self.get_negatives_from_another_question(q_id))

        doc: dict[str, str] = random.choice(neg_docs)
        return self._lookup_doc(doc)

    def choose_positive_and_negative_doc(
        self, sample_index: int, epoch: int, q_id: str
    ) -> tuple[str | None, str | None]:
        valid_sample_groups = [
            self.slice_dataset[q_id][ro]
            for ro in self.relevance_order
            if len(self.slice_dataset[q_id][ro]) > 0
        ]

        if len(valid_sample_groups) < 2:
            pos_doc_text = self._lookup_doc(
                cast(dict[str, str], random.choice(valid_sample_groups[0]))
            )
            neg_doc_text = self._lookup_doc(self.get_negatives_from_another_question(q_id))

        else:
            pos_doc_index = random.randrange(len(valid_sample_groups[:-1]))
            pos_doc_text = self._lookup_doc(
                cast(dict[str, str], random.choice(valid_sample_groups[pos_doc_index]))
            )

            neg_doc_list = random.choice(valid_sample_groups[pos_doc_index + 1 :])
            neg_doc_text = self._lookup_doc(cast(dict[str, str], random.choice(neg_doc_list)))

        return pos_doc_text, neg_doc_text


class ExponentialWeightSampler(BasicSampler):
    def choose_positive_and_negative_doc(
        self, sample_index: int, epoch: int, q_id: str
    ) -> tuple[str | None, str | None]:
        valid_sample_groups = [
            (ro, self.slice_dataset[q_id][ro])
            for ro in self.relevance_order
            if len(self.slice_dataset[q_id][ro]) > 0
        ]

        valid_sample_groups_ro, valid_sample_groups = list(zip(*valid_sample_groups))

        if len(valid_sample_groups) < 2:
            valid_sample_groups_ro = [
                ro for ro in self.relevance_order if len(self.slice_dataset[q_id][ro]) > 0
            ]
            # its impossible to sample from this questions
            print(
                f"Warning question ({q_id}) only contain ({len(valid_sample_groups_ro)}) valid_sample_groups ({valid_sample_groups_ro}) groups for sampling.",
                flush=True,
            )
            return None, None

        weights = [2**x for x in valid_sample_groups_ro]
        pos_doc_index = random.choices(
            range(len(valid_sample_groups[:-1])), weights=weights[:-1], k=1
        )[0]
        pos_doc_list: list[dict[str, str]] = cast(
            list[dict[str, str]], valid_sample_groups[pos_doc_index]
        )
        pos_doc: dict[str, str] = random.choice(pos_doc_list)
        pos_doc_text = self._lookup_doc(pos_doc)

        inverse_weights = [5 - x for x in valid_sample_groups_ro[pos_doc_index + 1 :]]
        # neg_doc_list = random.choice(valid_sample_groups[pos_doc_index+1:])
        neg_doc_list: list[dict[str, str]] = cast(
            list[dict[str, str]],
            random.choices(valid_sample_groups[pos_doc_index + 1 :], weights=inverse_weights, k=1)[
                0
            ],
        )
        neg_doc: dict[str, str] = random.choice(neg_doc_list)
        neg_doc_text = self._lookup_doc(neg_doc)

        return pos_doc_text, neg_doc_text


class HigherConfidenceNegativesSampler(BasicSampler):
    def choose_negative_doc(self, sample_index: int, epoch: int, q_id: str) -> str | None:
        neg_docs: list[dict[str, str]] = cast(
            list[dict[str, str]], self.slice_dataset[q_id]["neg_docs"]
        )

        if len(neg_docs) <= 10:
            # print("no_negative")
            return None

        if self.collection:
            neg_pmid = random.choice(neg_docs[10:])["id"]
            # print(f"{self.slice_dataset[q_id]['neg_docs'].index(neg_pmid)} - {len(self.slice_dataset[q_id]['neg_docs'])}")

            return self.collection[neg_pmid]
        else:
            # print(f" - {len(self.slice_dataset[q_id]['neg_docs'])}")

            return random.choice(neg_docs[10:])["text"]


class ShifterSampler(BasicSampler):
    """
    Curriculum learning sampler that progressively shifts from easier to harder negatives.

    This sampler implements a "shifting window" strategy where:
    - Early epochs: Sample from all negatives (positions 0 to N) - easier
    - Later epochs: Sample only from harder negatives (skipping first K*epoch positions)
    - Final epochs: Sample only from hardest negatives (near position 999)

    Assumes BM25 returns negatives roughly sorted by relevance (even if all are "negative",
    the first ones are closer semantic matches to the query).

    Args:
        slice_dataset: The dataset dictionary {q_id: {0: [...], 1: [...], "question": ...}}
        collection: Optional external document collection (passed to parent)
        max_epoch: Total number of training epochs (required)
        *args, **kwargs: Additional arguments passed to parent BasicSampler

    Example:
        With 999 negatives and max_epoch=10:
        - Epoch 0: sample from positions [0:999] (all)
        - Epoch 5: sample from positions [495:999] (skip first 495)
        - Epoch 9: sample from positions [891:999] (hardest 108 only)

    Note:
        The epoch parameter must be passed by the iterator for this to work.
        BioASQDataset.__iter__() automatically increments and passes the epoch.
    """

    def __init__(
        self,
        slice_dataset: SliceDataset,
        collection: Collection | None,
        *args,
        **kwargs,
    ):
        super().__init__(slice_dataset, collection, *args, **kwargs)
        self.max_epoch: int = kwargs.pop("max_epoch", 10)  # Default to 10 epochs

    # this is from start to the end of the list
    def choose_negative_doc_v1(self, sample_index: int, epoch: int, q_id: str) -> str | None:
        """
        Sample negative with shifting window based on epoch.

        Args:
            sample_index: Current sample index in iteration
            epoch: Current training epoch (0-indexed)
            q_id: Query ID to sample negative for

        Returns:
            Document text of selected negative, or None if epoch exceeds range
        """
        # Get negatives from the standard key (0)
        neg_docs: list[dict[str, str]] = cast(
            list[dict[str, str]], self.slice_dataset[q_id][self.negative_index]
        )  # type: ignore

        if len(neg_docs) == 0:
            return None

        # Calculate interval size based on max_epoch
        # Divide negative list into (max_epoch + 1) chunks
        interval = len(neg_docs) // (self.max_epoch + 1)

        # Starting position shifts right each epoch
        start_pos = interval * epoch

        # Check if we've exhausted the negatives for this epoch
        if len(neg_docs) < start_pos:
            return None

        # Sample from the remaining (harder) negatives
        doc: dict[str, str] = random.choice(neg_docs[start_pos:])
        return self._lookup_doc(doc)

    # this is from the end to the start of the list
    # so starts learn the very easy
    def choose_negative_doc(self, sample_index: int, epoch: int, q_id: str) -> str | None:
        """
        Sample negative with shrinking window toward the hardest negatives.
        - Early epochs: Sample from all negatives [0:N] (mix of hard and easy)
        - Final epochs: Sample only from the top hardest negatives [0:K]
        """
        # Get negatives from the standard key (0)
        neg_docs: list[dict[str, str]] = cast(
            list[dict[str, str]], self.slice_dataset[q_id][self.negative_index]
        )  # type: ignore

        if len(neg_docs) == 0:
            return None

        # Calculate how many negatives to drop from the EASY tail end each epoch
        interval = len(neg_docs) // (self.max_epoch + 1)

        # End position shrinks leftward (towards index 0) each epoch
        end_pos = len(neg_docs) - (interval * epoch)

        # Safety net: Ensure we always have a small pool of the hardest docs to sample from
        # (e.g., never shrink below the top 5 or 10, depending on what's available)
        min_pool_size = min(10, len(neg_docs))
        end_pos = max(end_pos, min_pool_size)

        # Sample from the hardest remaining negatives [0:end_pos]
        doc: dict[str, str] = random.choice(neg_docs[:end_pos])
        return self._lookup_doc(doc)

    def choose_positive_doc(self, sample_index: int, epoch: int, q_id: str) -> str | None:
        """
        Sample positive document using parent's logic.

        Overrides parent to maintain consistent signature with epoch parameter,
        but delegates to parent's positive sampling logic.
        """
        # Use parent's implementation (random from positive relevance levels)
        return super().choose_positive_doc(sample_index, epoch, q_id)
