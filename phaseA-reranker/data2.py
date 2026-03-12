import torch
from typing import Union
from utils import (
    split_chunks,
    get_negative_positive_index_from_dataset,
    get_relevance_order_from_dataset,
)
from collections import defaultdict
import json


class BioASQPointwiseIterator:
    def __init__(
        self, sample_preprocessing, sampler_class, num_neg_samples=1, **kwargs
    ):
        self.sample_preprocessing = sample_preprocessing
        self.num_neg_samples = num_neg_samples
        self.sampler_class = sampler_class
        assert num_neg_samples > 0

    def __call__(self, dataset, epoch=0, collection=None):
        self.slice_dataset = dataset
        self.collection = collection
        self.sampler = self.sampler_class(slice_dataset=dataset, collection=collection)
        self.index = 0
        self.epoch = epoch
        self.relevance_order = get_relevance_order_from_dataset(dataset)
        self.negative_index, self.positive_index = (
            min(self.relevance_order),
            max(self.relevance_order),
        )
        return self

    def __len__(self):
        return sum(
            [len(x[self.positive_index]) for x in self.slice_dataset.values()]
        ) * (1 + self.num_neg_samples)

    def __next__(self):
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

            sample = {
                "id": q_id,
                "query_text": q_text,
                "doc_text": doc_text,
                "label": label,
            }

            sample = self.sample_preprocessing(sample)

            # use sample_preprocessing to see if samples are valid
            if sample is not None:
                break

        self.index += 1

        return sample


class BioASQPairwiseIterator(BioASQPointwiseIterator):
    def __len__(self):
        return sum([len(x[self.positive_index]) for x in self.slice_dataset.values()])

    def _apply_pointwise_sampler_preprocessing(self, q_id, q_text, doc_text):

        sample = {
            "id": q_id,
            "query_text": q_text,
            "doc_text": doc_text,
            "label": -1,  # this will be discarded later on, its just to use the basic sampler alredy implemented
        }

        return self.sample_preprocessing(sample)

    def __next__(self):
        # spot criteria
        if self.index > len(self):
            raise StopIteration

        while True:
            q_id, q_text = self.sampler.choose_question(self.index, self.epoch)

            # choose
            doc_pos_text = self.sampler.choose_positive_doc(
                self.index, self.epoch, q_id
            )
            if doc_pos_text is None:
                return self.__next__()

            pos_sample = self._apply_pointwise_sampler_preprocessing(
                q_id, q_text, doc_pos_text
            )

            if pos_sample is None:
                return self.__next__()

            doc_neg_text = self.sampler.choose_negative_doc(
                self.index, self.epoch, q_id
            )
            if doc_neg_text is None:
                return self.__next__()

            neg_sample = self._apply_pointwise_sampler_preprocessing(
                q_id, q_text, doc_neg_text
            )

            if neg_sample is None:
                return self.__next__()

            pos_sample.pop("labels")
            neg_sample.pop("labels")

            sample = {"pos_inputs": pos_sample, "neg_inputs": neg_sample}

            break

        self.index += 1

        return sample


class BioASQRelevanceAwarePairwiseIterator(BioASQPairwiseIterator):
    def __len__(self):
        only_pos_relevance_order = self.relevance_order[:-1]
        return sum(
            [
                len(docs)
                for q_data in self.slice_dataset.values()
                for k in only_pos_relevance_order
                for docs in q_data[k]
            ]
        )

    def __next__(self):
        # spot criteria
        if self.index > len(self):
            raise StopIteration

        while True:
            q_id, q_text = self.sampler.choose_question(self.index, self.epoch)

            # choose pos and neg
            doc_pos_text, doc_neg_text = self.sampler.choose_positive_and_negative_doc(
                self.index, self.epoch, q_id
            )
            if doc_pos_text is None or doc_neg_text is None:
                return self.__next__()

            pos_sample = self._apply_pointwise_sampler_preprocessing(
                q_id, q_text, doc_pos_text
            )
            if pos_sample is None:
                return self.__next__()

            neg_sample = self._apply_pointwise_sampler_preprocessing(
                q_id, q_text, doc_neg_text
            )
            if neg_sample is None:
                return self.__next__()

            pos_sample.pop("labels")
            neg_sample.pop("labels")

            sample = {"pos_inputs": pos_sample, "neg_inputs": neg_sample}

            break

        self.index += 1

        return sample


class BioASQDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, dataset, iterator, collection=None, max_questions: int = 999999999
    ):
        """
        dataset: {query_id: {}}
        """
        super().__init__()
        self.dataset = {}
        self.epoch = -1

        self.iterator = iterator

        queries_ids = list(dataset.keys())

        if max_questions != -1:
            queries_ids = queries_ids[:max_questions]

        for q_id in queries_ids:
            self.dataset[q_id] = dataset[q_id]

        del dataset

        self.iterator = self.iterator(self.dataset)
        self.collection = collection

        # exit()

    def get_n_questions(self):
        return len(self.dataset)

    def get_qrels(self):
        return self.qrels_dict

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            iterator = self.iterator(
                dataset=self.dataset, epoch=self.epoch, collection=self.collection
            )
        else:  # in a worker process
            # split workload
            q_ids = list(self.dataset.keys())

            this_worker_ids = list(split_chunks(q_ids, worker_info.num_workers))[
                worker_info.id
            ]

            _dataset = {q_id: self.dataset[q_id] for q_id in this_worker_ids}

            iterator = self.iterator(
                dataset=_dataset, epoch=self.epoch, collection=self.collection
            )

        self.epoch += 1

        return iterator


class BioASQInferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset, sample_preprocessing, qrels_dict, max_docs: int = 999999999
    ):
        """
        dataset: [{id bm25 query_text}]
        """
        super().__init__()
        self.dataset = []
        self.sample_preprocessing = sample_preprocessing
        self.qrels_dict = qrels_dict

        # build sequential dataset
        for q_data in dataset:
            # print(q_data.keys())

            # fucking fix this shit
            if "bm25" in q_data.keys():
                _key = "bm25"
            elif "neg_docs" in q_data.keys():
                _key = "neg_docs"

            for doc in q_data[_key][:max_docs]:
                # for doc in q_data["neg_docs"][:max_docs]:

                # for doc in q_data["documents"][:max_docs]:

                self.dataset.append(
                    {
                        "id": q_data["id"],
                        "doc_id": doc["id"],
                        "doc_text": doc["text"],
                        # "query_text": q_data["question"],
                        "query_text": q_data["query_text"],
                    }
                )

    def get_qrels(self):
        return self.qrels_dict

    def __getitem__(self, idx):

        sample = self.dataset[idx]
        return self.sample_preprocessing(sample)

    def __len__(self):
        return len(self.dataset)


def create_bioASQ_datasets(
    positive_data_path,
    all_data_path,
    iterator,
    test_sample_preprocessing,
    val_files: list = [],
    relevance_mapping={"documents": 1},
    **kwargs,
):

    train_dataset = {}
    test_dataset = []
    # qrels_train = {}
    qrels_test = {}

    inverted_relevance_mapping = {v: k for k, v in relevance_mapping.items()}
    neg_index = min(inverted_relevance_mapping.keys()) - 1
    reminder_pos_index = sorted(
        list(inverted_relevance_mapping.items()), key=lambda x: -x[0]
    )
    # print(reminder_pos_index)
    # build the qids for test set
    test_set_qids = []
    test_set_questions = {}
    for file in val_files:
        with open(file) as f:
            data = json.load(f)
            for q in data["questions"]:
                test_set_qids.append(q["id"])
                test_set_questions[q["id"]] = q["body"]

    test_ds_pos = defaultdict(lambda: list())
    with open(positive_data_path) as f:
        for line in f:
            sample = json.loads(line)
            if sample["id"] not in test_set_qids:
                # add all positive docs

                train_dataset[sample["id"]] = {
                    **{
                        relevance_order: sample[key_name]
                        for relevance_order, key_name in reminder_pos_index
                    },
                    "question": sample["body"],
                }
                # qrels_train[sample["id"]] = {doc["id"]:1 for doc in sample["documents"]}
            else:
                test_ds_pos[sample["id"]] = [
                    sample[key_name] for _, key_name in reminder_pos_index
                ][0]
                qrels_test[sample["id"]] = {doc["id"]: 1 for doc in sample["documents"]}

    with open(all_data_path) as f:
        for line in f:
            sample = json.loads(line)
            if sample["id"] in test_set_qids:
                # print(sample['neg_docs'][0].keys())
                # print(test_ds_pos[sample["id"]][0].keys())
                # return
                sample["neg_docs"].extend(test_ds_pos[sample["id"]])
                test_dataset.append(
                    sample | {"query_text": test_set_questions[sample["id"]]}
                )

            else:
                train_dataset[sample["id"]][neg_index] = []
                postive_keys = {
                    doc["id"]
                    for order_key, _ in reminder_pos_index
                    for doc in train_dataset[sample["id"]][order_key]
                }
                # print(sample.keys())
                # for doc in sample["bm25"]:
                for doc in sample["neg_docs"]:
                    if doc["id"] not in postive_keys:
                        train_dataset[sample["id"]][neg_index].append(doc)

    # split kwargs
    train_dataset_args = {}
    test_dataset_args = {}  # ?
    # for k in kwargs:
    #    if k.startswith("train_"):
    #        train_dataset_args[k[6:]] = kwargs[k]
    #    elif k.startswith("test_"):
    #        test_dataset_args[k[5:]] = kwargs[k]

    train_dataset_args["dataset"] = train_dataset
    train_dataset_args["iterator"] = iterator

    # train_dataset_args["dataset"] = train_dataset
    # train_dataset_args["qrels_dict"] = qrels_train

    test_dataset_args["dataset"] = test_dataset
    test_dataset_args["qrels_dict"] = qrels_test
    test_dataset_args["sample_preprocessing"] = test_sample_preprocessing

    # test_dataset_args["qrels_dict"] = qrels_dict["test"]

    # return train_dataset_args, test_dataset_args
    return BioASQDataset(**train_dataset_args), BioASQInferenceDataset(
        **test_dataset_args
    )


def create_test_dataset(baselines_path, sample_preprocessing, val_files=None):

    test_dataset = []
    # qrels_train = {}
    qrels_test = {}

    # build the qids for test set
    if val_files is not None:
        for file in val_files:
            with open(file) as f:
                data = json.load(f)
                for q in data["questions"]:
                    qrels_test[q["id"]] = {doc["id"]: 1 for doc in q["documents"]}

    test_dataset = []

    # for baseline_path in baselines_path:
    with open(baselines_path) as f:
        for sample in map(json.loads, f):
            test_dataset.append(sample)

    # test_dataset_args["qrels_dict"] = qrels_dict["test"]

    return BioASQInferenceDataset(
        dataset=test_dataset,
        qrels_dict=qrels_test,
        sample_preprocessing=sample_preprocessing,
    )
