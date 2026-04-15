import random
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Self, cast, override

import orjson
import torch
from transformers import AutoTokenizer, TokenizersBackend

from bioasq.phase_a.reranker.aliases import (
    Collection,
    ProcessedSample,
    QrelsDict,
    Sample,
    SliceDataset,
)
from bioasq.phase_a.reranker.sample_preprocessing import BasicSamplePreprocessing
from bioasq.phase_a.reranker.sampler import BasicSampler, EmptySampler
from bioasq.phase_a.reranker.utils import (
    get_relevance_order_from_dataset,
    split_chunks,
)


class BioASQPointwiseIterator:
    # Class-level type annotations for attributes set in __call__
    slice_dataset: SliceDataset
    collection: Collection | None
    sampler_class_type: type[BasicSampler]
    index: int
    epoch: int
    relevance_order: list[int]
    negative_index: int
    positive_index: int

    def __init__(
        self,
        sample_preprocessing: BasicSamplePreprocessing,
        sampler_class_type: type[BasicSampler],
        num_neg_samples: int = 1,
        sampler_kwargs: dict[str, object] | None = None,
        **kwargs: object,
    ):
        self.sample_preprocessing: BasicSamplePreprocessing = sample_preprocessing
        self.num_neg_samples: int = num_neg_samples
        self.sampler_class: type[BasicSampler] = sampler_class_type
        self.sampler_kwargs: dict[str, object] = sampler_kwargs or {}
        # Initialize attributes to satisfy type checker
        self.sampler: BasicSampler = EmptySampler(slice_dataset={}, collection=None)
        self.slice_dataset = {}
        self.collection = None
        self.index = 0
        self.epoch = 0
        self.relevance_order = []
        self.negative_index = 0
        self.positive_index = 0
        assert num_neg_samples > 0

    def __call__(
        self,
        dataset: SliceDataset,
        epoch: int = 0,
        collection: Collection | None = None,
    ) -> Self:
        self.slice_dataset = dataset
        self.collection = collection
        self.sampler = self.sampler_class(
            slice_dataset=dataset,
            collection=collection,
            **self.sampler_kwargs,
        )
        self.index = 0
        self.epoch = epoch
        self.relevance_order = get_relevance_order_from_dataset(dataset)
        self.negative_index, self.positive_index = (
            min(self.relevance_order),
            max(self.relevance_order),
        )
        return self

    def __iter__(self) -> Self:
        return self

    def __len__(self) -> int:
        return sum([len(x[self.positive_index]) for x in self.slice_dataset.values()]) * (
            1 + self.num_neg_samples
        )

    def __next__(self) -> ProcessedSample:
        # spot criteria
        if self.index >= len(self):
            raise StopIteration

        label = not self.index % (self.num_neg_samples + 1)

        while True:
            q_id, q_text = self.sampler.choose_question(self.index, self.epoch)

            doc_text = (
                self.sampler.choose_positive_doc(self.index, self.epoch, q_id)
                if label
                else self.sampler.choose_negative_doc(self.index, self.epoch, q_id)
            )

            if doc_text is None:
                return self.__next__()  # ?
                # break # skip this sample bc no doc

            sample: Sample = {
                "id": q_id,
                "query_text": q_text,
                "doc_text": doc_text,
                "label": int(label),
            }

            # Process and return valid sample
            processed_sample: Sample = self.sample_preprocessing(sample)
            break

        self.index += 1

        return (
            processed_sample  # TODO THIS NEEDS TO COMPLAIN THE TYPE, POINTWISERS WILL NEED TO ADAPT
        )


class BioASQPairwiseIterator(BioASQPointwiseIterator):
    @override
    def __len__(self) -> int:
        return sum([len(x[self.positive_index]) for x in self.slice_dataset.values()])

    def _apply_pointwise_sampler_preprocessing(
        self, q_id: str, q_text: str, doc_text: str
    ) -> Sample:
        sample: Sample = {
            "id": q_id,
            "query_text": q_text,
            "doc_text": doc_text,
            "label": -1,  # this will be discarded later on
        }

        result = self.sample_preprocessing(sample)
        return result

    @override
    def __next__(self) -> ProcessedSample:
        # spot criteria
        if self.index > len(self):
            raise StopIteration
        while True:
            q_id, q_text = self.sampler.choose_question(self.index, self.epoch)

            # choose
            doc_pos_text = self.sampler.choose_positive_doc(self.index, self.epoch, q_id)
            if doc_pos_text is None:
                return self.__next__()

            pos_sample: Sample = self._apply_pointwise_sampler_preprocessing(
                q_id, q_text, doc_pos_text
            )

            doc_neg_text = self.sampler.choose_negative_doc(self.index, self.epoch, q_id)
            if doc_neg_text is None:
                return self.__next__()

            neg_sample = self._apply_pointwise_sampler_preprocessing(q_id, q_text, doc_neg_text)

            _ = pos_sample.pop("labels", None)
            _ = neg_sample.pop("labels", None)

            sample: ProcessedSample = {
                "pos_inputs": pos_sample,
                "neg_inputs": neg_sample,
            }

            break

        self.index: int = self.index + 1  # This is shit pyright

        return sample


class BioASQMultiNegativePairwiseIterator(BioASQPairwiseIterator):
    """
    Pairwise iterator that pairs ONE positive document with MULTIPLE negative documents.

    This is useful for training with more diverse negative samples per positive,
    giving the model a stronger signal for learning to distinguish relevant from
    non-relevant documents.

    Output format:
    {
        "pos_inputs": {...},           # The positive document sample
        "neg_inputs": [{...}, {...}]   # List of negative document samples
    }
    """

    def __init__(
        self,
        sample_preprocessing: BasicSamplePreprocessing,
        sampler_class: type[BasicSampler],
        num_neg_samples: int = 1,
        **kwargs: object,
    ):
        super().__init__(sample_preprocessing, sampler_class, num_neg_samples, **kwargs)
        self.num_neg_samples: int = num_neg_samples
        self.sampler_class: type[BasicSampler] = sampler_class
        assert num_neg_samples > 0, "num_neg_samples must be greater than 0"

    @override
    def __len__(self) -> int:
        """Length equals number of positive documents (each generates one sample with N negatives)."""
        return sum([len(x[self.positive_index]) for x in self.slice_dataset.values()])

    @override
    def __next__(self) -> ProcessedSample:
        # Check stopping criteria
        if self.index >= len(self):
            raise StopIteration

        while True:
            # Choose a random query
            q_id, q_text = self.sampler.choose_question(self.index, self.epoch)

            # Sample ONE positive document
            doc_pos_text = self.sampler.choose_positive_doc(self.index, self.epoch, q_id)
            if doc_pos_text is None:
                return self.__next__()

            # Apply preprocessing to positive sample
            pos_sample: Sample = self._apply_pointwise_sampler_preprocessing(
                q_id, q_text, doc_pos_text
            )

            # Sample MULTIPLE negative documents
            neg_doc_texts = [
                self.sampler.choose_negative_doc(self.index, self.epoch, q_id)
                for _ in range(self.num_neg_samples)
            ]
            if None in neg_doc_texts:
                return self.__next__()

            # Apply preprocessing to all negative samples
            neg_samples: list[Sample] = []
            for neg_text in neg_doc_texts:
                if neg_text is None:
                    break
                neg_sample = self._apply_pointwise_sampler_preprocessing(q_id, q_text, neg_text)
                neg_samples.append(neg_sample)

            # If we didn't get all negatives, retry
            if len(neg_samples) != self.num_neg_samples:
                return self.__next__()

            # Remove labels from all samples
            _ = pos_sample.pop("labels", None)
            _ = [neg_sample.pop("labels", None) for neg_sample in neg_samples]

            # Create the output structure
            sample: ProcessedSample = {
                "pos_inputs": pos_sample,
                "neg_inputs": neg_samples,
            }

            break

        self.index: int = self.index + 1  # This is shit pyright
        return sample


class BioASQDataset(torch.utils.data.IterableDataset[ProcessedSample]):
    dataset: SliceDataset
    epoch: int
    iterator: BioASQPointwiseIterator | Callable[..., BioASQPointwiseIterator]
    collection: Collection | None
    qrels_dict: QrelsDict | None

    def __init__(
        self,
        dataset: SliceDataset,
        iterator: BioASQPointwiseIterator | Callable[..., BioASQPointwiseIterator],
        collection: Collection | None = None,
        max_questions: int = 999999999,
    ) -> None:
        """
        dataset: {query_id: {}}
        """
        super().__init__()
        self.dataset = {}
        self.epoch = -1

        queries_ids = list(dataset.keys())

        if max_questions != -1:
            queries_ids = queries_ids[:max_questions]

        for q_id in queries_ids:
            self.dataset[q_id] = dataset[q_id]

        del dataset

        # Store the iterator factory/callable
        self.iterator = iterator
        self.collection = collection
        self.qrels_dict = None
        # exit()

    def get_n_questions(self) -> int:
        return len(self.dataset)

    def get_qrels(self) -> QrelsDict | None:
        return self.qrels_dict

    def __len__(self) -> int:
        # The iterator needs to be called with dataset first to get a Sized object
        _iterator = self.iterator(
            dataset=self.dataset, epoch=self.epoch, collection=self.collection
        )
        return len(_iterator)

    @override
    def __iter__(self) -> Iterator[ProcessedSample]:
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            iterator = self.iterator(
                dataset=self.dataset, epoch=self.epoch, collection=self.collection
            )
        else:  # in a worker process
            # split workload
            q_ids: list[str] = list(self.dataset.keys())
            this_worker_ids: list[str] = list(split_chunks(q_ids, worker_info.num_workers))[
                worker_info.id
            ]

            _dataset: SliceDataset = {q_id: self.dataset[q_id] for q_id in this_worker_ids}

            iterator = self.iterator(dataset=_dataset, epoch=self.epoch, collection=self.collection)

        self.epoch += 1

        return iter(iterator)

    @override
    def __repr__(self) -> str:
        # print all the data that is in the dataset
        final_string = ""
        final_string += f"BioASQDataset(size={len(self)})\n"

        for q_id in self.dataset:
            final_string += f"   - Query ID: {q_id}\n"
            final_string += f"   - Query Text: {self.dataset[q_id]['question'][:100]}...\n"
            final_string += f"   - 0 Docs: {len(self.dataset[q_id][0])}...\n"
            final_string += f"   - 1 Docs: {len(self.dataset[q_id][1])}\n"
            final_string += "-" * 100 + "\n"
        final_string += "\n\n"
        return final_string


class BioASQInferenceDataset(torch.utils.data.Dataset[ProcessedSample]):
    dataset_: list[Sample]
    sample_preprocessing: BasicSamplePreprocessing
    qrels_dict: QrelsDict

    def __init__(
        self,
        dataset_: list[ProcessedSample],
        sample_preprocessing: BasicSamplePreprocessing,
        qrels_dict: QrelsDict,
        max_docs: int = 999999999,
        add_labels: bool = False,
    ):
        """
        dataset: [{id bm25 query_text}]
        add_labels: if True, inject label from qrels for pointwise eval
        """
        super().__init__()
        self.dataset: list[Sample] = []
        self.sample_preprocessing = sample_preprocessing
        self.qrels_dict = qrels_dict
        self.add_labels = add_labels

        # build sequential dataset
        # if _key is "Neg_docs" then the dataset has the structure
        # dict_keys(['id', 'neg_docs', 'query_text'])
        # {id: str, query_text: str, neg_docs: [{"id": str, "text": str, "score": float}]}

        for q_data in dataset_:
            # fucking fix this shit
            _key: str | None = None
            if "bm25" in q_data.keys():
                _key = "bm25"
            elif "neg_docs" in q_data.keys():
                _key = "neg_docs"
            elif "documents" in q_data.keys():
                _key = "documents"

            if _key is None:
                continue

            docs = cast(list[dict[str, str]], q_data[_key])
            for doc in docs[:max_docs]:
                # for doc in q_data["neg_docs"][:max_docs]:

                # for doc in q_data["documents"][:max_docs]:

                self.dataset.append(
                    {
                        "id": str(q_data["id"]),
                        "doc_id": str(doc["id"]),
                        "doc_text": str(doc["text"]),
                        # "query_text": q_data["question"],
                        "query_text": str(q_data["query_text"]),
                    }
                )

    def get_qrels(self) -> QrelsDict:
        return self.qrels_dict

    @override
    def __getitem__(self, index: int) -> ProcessedSample:
        sample = dict(self.dataset[index])
        if self.add_labels:
            qid = str(sample["id"])
            doc_id = str(sample["doc_id"])
            sample["label"] = 1 if self.qrels_dict.get(qid, {}).get(doc_id, 0) >= 1 else 0
        result = self.sample_preprocessing(sample)
        return cast(ProcessedSample, dict(result))

    def __len__(self) -> int:
        return len(self.dataset)

    @override
    def __repr__(self) -> str:
        final_string = ""
        final_string += f"BioASQInferenceDataset(size={len(self)})\n"

        for _sample in self.dataset:
            sample = cast(dict[str, str], _sample)
            final_string += f"   - ID: {sample['id']}\n"
            final_string += f"   - Doc ID: {sample['doc_id']}\n"
            final_string += f"   - Doc Text: {sample['doc_text'][:100]}...\n"
            final_string += f"   - Query Text: {sample['query_text'][:100]}...\n"
            final_string += "-" * 100 + "\n"
        final_string += "\n\n"
        return final_string


class BioASQPairwiseEvalDataset(torch.utils.data.Dataset[ProcessedSample]):
    """Eval dataset that yields (pos_inputs, neg_inputs) pairs for pairwise Trainer eval."""

    def __init__(
        self,
        raw_test: list[ProcessedSample],
        qrels_dict: QrelsDict,
        sample_preprocessing: BasicSamplePreprocessing,
        pairs_per_query: int = 1,
        seed: int = 42,
        neg_as_list: bool = False,
    ):
        super().__init__()
        self.raw_test = raw_test
        self.qrels_dict = qrels_dict
        self.sample_preprocessing = sample_preprocessing
        self.pairs_per_query = pairs_per_query
        self.seed = seed
        self.neg_as_list = neg_as_list
        self._samples: list[ProcessedSample] = []
        self._build_samples()

    def _get_docs_key(self, entry: dict) -> str | None:
        if "neg_docs" in entry:
            return "neg_docs"
        if "bm25" in entry:
            return "bm25"
        return None

    def _build_samples(self) -> None:
        rng = random.Random(self.seed)
        for entry in self.raw_test:
            key = self._get_docs_key(entry)
            if key is None:
                continue
            qid = str(entry["id"])
            query_text = str(entry["query_text"])
            qrels = self.qrels_dict.get(qid, {})
            docs = cast(list[dict[str, str]], entry[key])
            rel_ids = {d for d, s in qrels.items() if s >= 1}
            pos_docs = [d for d in docs if str(d["id"]) in rel_ids]
            neg_docs = [d for d in docs if str(d["id"]) not in rel_ids]
            if not pos_docs or not neg_docs:
                continue
            for _ in range(self.pairs_per_query):
                pos_doc = rng.choice(pos_docs)
                neg_doc = rng.choice(neg_docs)
                pos_sample = self.sample_preprocessing(
                    {"id": qid, "query_text": query_text, "doc_text": pos_doc["text"], "label": 1}
                )
                neg_sample = self.sample_preprocessing(
                    {"id": qid, "query_text": query_text, "doc_text": neg_doc["text"], "label": 0}
                )
                pos_sample.pop("labels", None)
                neg_sample.pop("labels", None)
                neg_inputs: ProcessedSample | list[ProcessedSample] = (
                    [neg_sample] if self.neg_as_list else neg_sample
                )
                self._samples.append({"pos_inputs": pos_sample, "neg_inputs": neg_inputs})

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> ProcessedSample:
        return self._samples[index]


# Relevance mapping when using expanded positive training (semantic-similarity-expanded docs).
# Gold documents get highest weight (5), then 0.95, 0.9, 0.85, 0.8 similarity tiers.
EXPANDED_RELEVANCE_MAPPING: dict[str, int] = {
    "documents": 5,
    "expanded_docs_095": 4,
    "expanded_docs_09": 3,
    "expanded_docs_085": 2,
    "expanded_docs_08": 1,
}


def create_bioASQ_datasets(
    positive_data_path: str,
    all_data_path: str,
    iterator: BioASQPointwiseIterator,
    test_sample_preprocessing: BasicSamplePreprocessing,
    val_files: list[str] | None = None,
    relevance_mapping: dict[str, int] | None = None,
    use_expanded_pos: bool = False,
    **kwargs: object,
) -> tuple[
    BioASQDataset,
    BioASQInferenceDataset,
    BioASQInferenceDataset | None,
    BioASQPairwiseEvalDataset | None,
    BioASQPairwiseEvalDataset | None,
]:
    if val_files is None:
        val_files = []
    if relevance_mapping is None:
        relevance_mapping = EXPANDED_RELEVANCE_MAPPING if use_expanded_pos else {"documents": 1}

    train_dataset: SliceDataset = {}
    test_dataset: list[ProcessedSample] = []
    # qrels_train = {}
    qrels_test: QrelsDict = {}

    inverted_relevance_mapping = {v: k for k, v in relevance_mapping.items()}
    neg_index = min(inverted_relevance_mapping.keys()) - 1
    reminder_pos_index = sorted(list(inverted_relevance_mapping.items()), key=lambda x: -x[0])
    # print(reminder_pos_index)
    # build the qids for test set
    test_set_qids: list[str] = []
    test_set_questions: dict[str, str] = {}
    for file in val_files:
        with open(file) as f:
            data: dict[str, list[dict[str, str]]] = orjson.loads(f.read())
            for q in data["questions"]:
                test_set_qids.append(str(q["id"]))
                test_set_questions[q["id"]] = str(q["body"])

    test_ds_pos: defaultdict[str, list[dict[str, str]]] = defaultdict(lambda: list())

    with open(positive_data_path) as f:
        for line in f:
            sample: dict[str, str | list[dict[str, str]]] = orjson.loads(line)  # pyright: ignore[reportAny]
            sample_id = str(sample["id"])
            sample_body = str(sample["body"])

            # Skip to test set handling if this is a test query
            if sample_id in test_set_qids:
                docs: list[dict[str, str]] = cast(
                    list[dict[str, str]],
                    [sample[key_name] for _, key_name in reminder_pos_index][0],
                )
                test_ds_pos[sample_id] = docs
                doc_list: list[dict[str, str]] = cast(list[dict[str, str]], sample["documents"])
                qrels_test[sample_id] = {doc["id"]: 1 for doc in doc_list}
                continue

            # Add all positive docs to training set
            entry: dict[str | int, list[dict[str, str]] | str] = {"question": sample_body}
            for relevance_order, key_name in reminder_pos_index:
                # Use .get() for expanded keys; non-expanded files won't have them
                entry[relevance_order] = sample.get(key_name, [])
            train_dataset[sample_id] = entry
            # qrels_train[sample["id"]] = {doc["id"]:1 for doc in sample["documents"]}

    with open(all_data_path) as f:
        for line in f:
            sample_data: dict[str, list[dict[str, str]]] = orjson.loads(line)  # pyright: ignore[reportAny]
            sample_id = str(sample_data["id"])

            # Handle test set samples
            if sample_id in test_set_qids:
                neg_docs: list[dict[str, str]] = sample_data["neg_docs"]
                neg_docs.extend(test_ds_pos[sample_id])
                new_entry: ProcessedSample = cast(ProcessedSample, dict(sample_data))
                new_entry["query_text"] = test_set_questions[sample_id]
                test_dataset.append(new_entry)
                continue

            # Handle training set samples - add negative docs
            train_dataset[sample_id][neg_index] = []
            postive_keys: set[str] = {
                cast(dict[str, str], doc)["id"]
                for order_key, _ in reminder_pos_index
                for doc in train_dataset[sample_id][order_key]
            }
            # print(sample.keys())
            # for doc in sample["bm25"]:
            # TODO NEEDS TYPE HINTS
            neg_docs_list: list[dict[str, str]] = sample_data["neg_docs"]
            for doc in neg_docs_list:
                if str(doc["id"]) not in postive_keys:
                    train_list: list[dict[str, str]] = cast(
                        list[dict[str, str]], train_dataset[sample_id][neg_index]
                    )
                    train_list.append(doc)
        # Obtain 10 samples of each id (0 or 1) and save in a buffer that is a mini dataset

    train_ds = BioASQDataset(dataset=train_dataset, iterator=iterator)
    test_ds = BioASQInferenceDataset(
        dataset_=test_dataset,
        qrels_dict=qrels_test,
        sample_preprocessing=test_sample_preprocessing,
        add_labels=False,
    )
    eval_pointwise: BioASQInferenceDataset | None = None
    eval_pairwise: BioASQPairwiseEvalDataset | None = None
    eval_multi_neg: BioASQPairwiseEvalDataset | None = None
    if test_dataset:
        eval_pointwise = BioASQInferenceDataset(
            dataset_=test_dataset,
            qrels_dict=qrels_test,
            sample_preprocessing=test_sample_preprocessing,
            add_labels=True,
        )
        eval_pairwise = BioASQPairwiseEvalDataset(
            raw_test=test_dataset,
            qrels_dict=qrels_test,
            sample_preprocessing=test_sample_preprocessing,
            neg_as_list=False,
        )
        eval_multi_neg = BioASQPairwiseEvalDataset(
            raw_test=test_dataset,
            qrels_dict=qrels_test,
            sample_preprocessing=test_sample_preprocessing,
            neg_as_list=True,
        )
    return train_ds, test_ds, eval_pointwise, eval_pairwise, eval_multi_neg


def create_inference_dataset_from_bioasq_json(
    path: str | Path,
    sample_preprocessing: BasicSamplePreprocessing,
    max_docs: int = 999999999,
) -> BioASQInferenceDataset:
    """Load questions with document candidates from BioASQ JSON or JSONL for inference.

    JSON format: {"questions": [{"id", "body", "documents"|"neg_docs"|"bm25"}]}
    JSONL format: one object per line with {"id", "query_text"|"body", "neg_docs"|"bm25"}
    Each doc in the candidates list must have {"id": str, "text": str}.
    """
    path = Path(path)
    raw: list[dict] = []

    if path.suffix.lower() == ".jsonl":
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                q = orjson.loads(line)
                query_text = str(q.get("body", q.get("query_text", "")))
                docs_key = _get_docs_key(q)
                if docs_key is None:
                    continue
                raw.append({"id": str(q["id"]), "query_text": query_text, docs_key: q[docs_key]})
    else:
        with open(path) as f:
            data = orjson.loads(f.read())
        for q in data["questions"]:
            query_text = str(q.get("body", q.get("query_text", "")))
            docs_key = _get_docs_key(q)
            if docs_key is None:
                continue
            raw.append({"id": str(q["id"]), "query_text": query_text, docs_key: q[docs_key]})

    return BioASQInferenceDataset(
        dataset_=raw,
        qrels_dict={},
        sample_preprocessing=sample_preprocessing,
        add_labels=False,
        max_docs=max_docs,
    )


def _get_docs_key(q: dict) -> str | None:
    """Return the key for document candidates (documents, neg_docs, bm25) or None."""
    for key in ("documents", "neg_docs", "bm25"):
        if key not in q or not q[key]:
            continue
        docs = q[key]
        first = docs[0]
        if isinstance(first, str):
            continue
        if "text" not in first or "id" not in first:
            continue
        return key
    return None


def create_test_dataset(
    baselines_path: str,
    sample_preprocessing: BasicSamplePreprocessing,
    val_files: list[str] | None = None,
) -> BioASQInferenceDataset:
    if val_files is None:
        val_files = []

    test_dataset: list[ProcessedSample] = []
    # qrels_train = {}
    qrels_test: QrelsDict = {}

    # build the qids for test set
    for file in val_files:
        with open(file) as f:
            data: dict[str, list[dict[str, str | list[dict[str, str]]]]] = orjson.loads(f.read())  # pyright: ignore[reportAny]
            for q in data["questions"]:
                doc_list: list[dict[str, str]] = cast(list[dict[str, str]], q["documents"])
                qrels_test[str(q["id"])] = {doc["id"]: 1 for doc in doc_list}

    # for baseline_path in baselines_path:
    with open(baselines_path) as f:
        for sample in map(orjson.loads, f):
            test_dataset.append(sample)

    # test_dataset_args["qrels_dict"] = qrels_dict["test"]

    return BioASQInferenceDataset(
        dataset_=test_dataset,
        qrels_dict=qrels_test,
        sample_preprocessing=sample_preprocessing,
    )


if __name__ == "__main__":
    print("Starting the script")
    tokenizer = cast(TokenizersBackend, AutoTokenizer.from_pretrained("bert-base-uncased"))

    train_dataset, test_dataset, _, _, _ = create_bioASQ_datasets(
        positive_data_path="../../data/quality/training14b_inflated_clean_wContents.jsonl",
        all_data_path="../../data/quality/hard_negatives_IA_clean.jsonl",
        iterator=BioASQPairwiseIterator(
            sample_preprocessing=BasicSamplePreprocessing(tokenizer),
            sampler_class_type=BasicSampler,
            num_neg_samples=1,
            sampler_kwargs={},
        ),
        test_sample_preprocessing=BasicSamplePreprocessing(tokenizer),
        val_files=[
            "../../data/val_data/13B1_golden.json",
        ],
    )
    # print(train_dataset)
    # print(test_dataset)

    print("Script completed successfully")
