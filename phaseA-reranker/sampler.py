import random
from utils import get_relevance_order_from_dataset


class BasicSampler:
    def __init__(self, slice_dataset, collection, *args, **kwargs):
        self.slice_dataset = slice_dataset
        self.collection = collection
        self.q_ids = list(self.slice_dataset.keys())

        self.relevance_order = get_relevance_order_from_dataset(self.slice_dataset)
        self.negative_index, self.positive_index = (
            min(self.relevance_order),
            max(self.relevance_order),
        )

    def _lookup_doc(self, document):
        if self.collection:
            pmid = document["id"]
            return self.collection[pmid]
        else:
            return document["text"]

    def choose_question(self, sample_index, epoch):
        q_id = random.choice(self.q_ids)
        q_text = self.slice_dataset[q_id]["question"]
        return q_id, q_text

    def choose_positive_doc(self, sample_index, epoch, q_id):
        valid_sample_groups = [
            ro
            for ro in self.relevance_order
            if len(self.slice_dataset[q_id][ro]) > 0 and ro != self.negative_index
        ]
        pos_index = random.choice(valid_sample_groups)

        return self._lookup_doc(random.choice(self.slice_dataset[q_id][pos_index]))

    def choose_negative_doc(self, sample_index, epoch, q_id):

        if len(self.slice_dataset[q_id][self.negative_index]) == 0:
            return None

        return self._lookup_doc(
            random.choice(self.slice_dataset[q_id][self.negative_index])
        )

    def choose_positive_and_negative_doc(self, sample_index, epoch, q_id):

        valid_sample_groups = [
            self.slice_dataset[q_id][ro]
            for ro in self.relevance_order
            if len(self.slice_dataset[q_id][ro]) > 0
        ]

        if len(valid_sample_groups) < 2:
            valid_sample_groups = [
                ro
                for ro in self.relevance_order
                if len(self.slice_dataset[q_id][ro]) > 0
            ]
            # its impossible to sample from this questions
            print(
                f"Warning question ({q_id}) only contain ({len(valid_sample_groups)}) valid_sample_groups ({valid_sample_groups}) groups for sampling.",
                flush=True,
            )
            return None, None

        pos_doc_index = random.randrange(len(valid_sample_groups[:-1]))
        pos_doc_text = self._lookup_doc(
            random.choice(valid_sample_groups[pos_doc_index])
        )

        neg_doc_list = random.choice(valid_sample_groups[pos_doc_index + 1 :])
        neg_doc_text = self._lookup_doc(random.choice(neg_doc_list))

        return pos_doc_text, neg_doc_text


class BasicV2Sampler(BasicSampler):
    def get_negatives_from_another_question(self, q_id):

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

        return random.choice(self.slice_dataset[new_q_id][self.negative_index])

    def choose_negative_doc(self, sample_index, epoch, q_id):

        if len(self.slice_dataset[q_id][self.negative_index]) == 0:
            return self._lookup_doc(self.get_negatives_from_another_question(q_id))

        return self._lookup_doc(
            random.choice(self.slice_dataset[q_id][self.negative_index])
        )

    def choose_positive_and_negative_doc(self, sample_index, epoch, q_id):

        valid_sample_groups = [
            self.slice_dataset[q_id][ro]
            for ro in self.relevance_order
            if len(self.slice_dataset[q_id][ro]) > 0
        ]

        if len(valid_sample_groups) < 2:
            pos_doc_text = self._lookup_doc(random.choice(valid_sample_groups[0]))
            neg_doc_text = self._lookup_doc(
                self.get_negatives_from_another_question(q_id)
            )

        else:
            pos_doc_index = random.randrange(len(valid_sample_groups[:-1]))
            pos_doc_text = self._lookup_doc(
                random.choice(valid_sample_groups[pos_doc_index])
            )

            neg_doc_list = random.choice(valid_sample_groups[pos_doc_index + 1 :])
            neg_doc_text = self._lookup_doc(random.choice(neg_doc_list))

        return pos_doc_text, neg_doc_text


class ExponentialWeightSampler(BasicSampler):
    def choose_positive_and_negative_doc(self, sample_index, epoch, q_id):

        valid_sample_groups = [
            (ro, self.slice_dataset[q_id][ro])
            for ro in self.relevance_order
            if len(self.slice_dataset[q_id][ro]) > 0
        ]

        valid_sample_groups_ro, valid_sample_groups = list(zip(*valid_sample_groups))

        if len(valid_sample_groups) < 2:
            valid_sample_groups = [
                ro
                for ro in self.relevance_order
                if len(self.slice_dataset[q_id][ro]) > 0
            ]
            # its impossible to sample from this questions
            print(
                f"Warning question ({q_id}) only contain ({len(valid_sample_groups)}) valid_sample_groups ({valid_sample_groups}) groups for sampling.",
                flush=True,
            )
            return None, None

        weights = [2**x for x in valid_sample_groups_ro]
        pos_doc_index = random.choices(
            range(len(valid_sample_groups[:-1])), weights=weights[:-1], k=1
        )[0]
        pos_doc_text = self._lookup_doc(
            random.choice(valid_sample_groups[pos_doc_index])
        )

        inverse_weights = [5 - x for x in valid_sample_groups_ro[pos_doc_index + 1 :]]
        # neg_doc_list = random.choice(valid_sample_groups[pos_doc_index+1:])
        neg_doc_list = random.choices(
            valid_sample_groups[pos_doc_index + 1 :], weights=inverse_weights, k=1
        )[0]
        neg_doc_text = self._lookup_doc(random.choice(neg_doc_list))

        return pos_doc_text, neg_doc_text


class HigherConfidenceNegativesSampler(BasicSampler):
    def choose_negative_doc(self, sample_index, epoch, q_id):

        if len(self.slice_dataset[q_id]["neg_docs"]) <= 10:
            # print("no_negative")
            return None

        if self.collection:
            neg_pmid = random.choice(self.slice_dataset[q_id]["neg_docs"][10:])["id"]
            # print(f"{self.slice_dataset[q_id]['neg_docs'].index(neg_pmid)} - {len(self.slice_dataset[q_id]['neg_docs'])}")

            return self.collection[neg_pmid]["text"]
        else:
            # print(f" - {len(self.slice_dataset[q_id]['neg_docs'])}")

            return random.choice(self.slice_dataset[q_id]["neg_docs"][10:])["text"]


class ShifterSampler:
    def __init__(self, slice_dataset, max_epoch, *args, **kwargs):
        self.slice_dataset = slice_dataset
        self.max_epoch = max_epoch
        self.q_ids = list(self.slice_dataset.keys())

    def choose_question(self, sample_index, epoch):
        q_id = random.choice(self.q_ids)
        q_text = self.slice_dataset[q_id]["question"]
        return q_id, q_text

    def choose_positive_doc(self, sample_index, epoch, q_id):
        return random.choice(self.slice_dataset[q_id]["pos_docs"])["text"]

    def choose_negative_doc(self, sample_index, epoch, q_id):

        neg_docs = self.slice_dataset[q_id]["neg_docs"]
        interval = len(neg_docs) // (self.max_epoch + 1)

        if len(self.slice_dataset[q_id]["neg_docs"]) < interval * epoch:
            return None

        return random.choice(neg_docs[interval * epoch :])["text"]
