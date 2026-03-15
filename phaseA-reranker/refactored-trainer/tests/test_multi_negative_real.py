"""
Test BioASQMultiNegativePairwiseIterator with REAL BioASQ data.
Demonstrates using multiple negatives per positive with actual training data.
"""

import sys

sys.path.insert(0, "/home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer")

import json
from typing import cast
from data import BioASQDataset, BioASQMultiNegativePairwiseIterator
from sampler import BasicSampler
from sample_preprocessing import BasicSamplePreprocessing


class RawTextPreprocessing(BasicSamplePreprocessing):
    """Preprocessing that keeps raw text instead of converting to tensors."""

    def __init__(self) -> None:
        # Bypass the parent __init__ since we don't need a tokenizer
        pass

    def __call__(
        self, sample: dict[str, str | int]
    ) -> dict[str, str | int]:
        sample = sample.copy()
        if "label" in sample:
            sample["labels"] = sample.pop("label")
        return sample


class IterableMultiNegIterator(BioASQMultiNegativePairwiseIterator):
    def __iter__(self) -> "IterableMultiNegIterator":
        return self


def load_real_dataset(max_queries: int = 20) -> dict[str, dict[str | int, list | str]]:
    """Load a subset of the real BioASQ dataset."""
    positives_path = "../../data/quality/training14b_inflated_clean_wContents.jsonl"
    all_data_path = "../../data/quality/hard_negatives_IA_clean.jsonl"

    train_dataset: dict[str, dict[str | int, list | str]] = {}

    # Load positive data
    with open(positives_path, "r") as f:
        for i, line in enumerate(f):
            if i >= max_queries:
                break
            sample_: dict[str, str | list] = json.loads(line)
            sample_id = str(sample_["id"])
            train_dataset[sample_id] = {
                1: sample_["documents"],
                "question": sample_["body"],
            }

    # Add negative docs
    with open(all_data_path, "r") as f:
        for line in f:
            sample: dict[str, str | list] = json.loads(line)
            sample_id = str(sample["id"])
            if sample_id in train_dataset:
                train_dataset[sample_id][0] = []
                positive_ids = {
                    doc["id"] for doc in train_dataset[sample_id][1]  # type: ignore
                }
                neg_docs: list = cast(list, sample["neg_docs"])
                for doc in neg_docs:
                    if doc["id"] not in positive_ids:
                        neg_list: list = cast(list, train_dataset[sample_id][0])
                        neg_list.append(doc)

    return train_dataset


def main() -> None:
    print("=" * 80)
    print("MULTI-NEGATIVE PAIRWISE WITH REAL BIOASQ DATA")
    print("=" * 80)

    print("\n1. Loading real dataset (first 20 queries)...")
    dataset_dict = load_real_dataset(max_queries=20)
    print(f"   Loaded {len(dataset_dict)} queries")

    # Show query statistics
    total_pos = sum(len(q[1]) for q in dataset_dict.values())  # type: ignore
    total_neg = sum(len(q[0]) for q in dataset_dict.values())  # type: ignore
    print(f"   Total positive docs: {total_pos}")
    print(f"   Total negative docs: {total_neg}")
    print(f"   Average negatives per positive: {total_neg // total_pos}")

    # Test with num_neg_samples=3
    print(f"\n{'='*80}")
    print("TESTING WITH num_neg_samples=3")
    print(f"{'='*80}")

    dataset = BioASQDataset(
        dataset=dataset_dict,
        iterator=lambda dataset, epoch=0, collection=None: IterableMultiNegIterator(
            sample_preprocessing=RawTextPreprocessing(),
            sampler_class=BasicSampler,
            num_neg_samples=3,  # 3 negatives per positive
        )(dataset=dataset, epoch=epoch, collection=collection),
    )

    print(f"\n   Dataset length: {len(dataset)}")
    print("   (Each positive generates 1 sample with 3 negatives)")

    # Get first 3 samples
    print("\n   First 3 samples:")
    print("-" * 80)

    iterator = iter(dataset)
    for i in range(3):
        try:
            sample = next(iterator)
        except StopIteration:
            break

        pos_inputs: dict = sample["pos_inputs"]  # type: ignore
        neg_inputs: list = sample["neg_inputs"]  # type: ignore

        print(f"\n   Sample {i+1}:")
        print(f"   Query ID: {pos_inputs['id']}")
        print(f"   Query: {pos_inputs['query_text'][:70]}...")

        print("\n   POSITIVE:")
        pos_text = pos_inputs["doc_text"]
        print(f"      {pos_text[:100]}...")

        print(f"\n   NEGATIVES ({len(neg_inputs)} total):")
        for j, neg in enumerate(neg_inputs):
            neg_text = neg["doc_text"]
            print(f"      {j+1}. {neg_text[:70]}...")

        print("-" * 80)

    # Explain loss computation
    print("\n" + "=" * 80)
    print("HOW TO COMPUTE LOSS WITH MULTIPLE NEGATIVES:")
    print("=" * 80)
    print("""
   For a sample with 1 positive and N negatives:

   scores_pos = model(query, pos_doc)          # [batch_size, 1]
   scores_negs = [model(query, neg_i) for neg_i in neg_docs]  # [batch_size, N]

   # Option 1: Average over all negative pairs
   loss = sum([max(0, margin - (score_pos - score_neg_i))
              for score_neg_i in scores_negs]) / N

   # Option 2: Hardest negative only
   loss = max(0, margin - (score_pos - min(scores_negs)))

   # Option 3: All negative sum (stronger signal)
   loss = sum([max(0, margin - (score_pos - score_neg_i))
              for score_neg_i in scores_negs])

   # In PyTorch:
   criterion = nn.MarginRankingLoss(margin=0.5)
   loss = sum([
       criterion(score_pos.expand_as(score_neg), score_neg, torch.ones_like(score_neg))
       for score_neg in scores_negs
   ])
    """)
    print("=" * 80)


if __name__ == "__main__":
    main()
