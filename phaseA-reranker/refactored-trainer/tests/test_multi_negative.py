"""
Test script to demonstrate the BioASQMultiNegativePairwiseIterator.
Shows how one positive is paired with multiple negatives.
"""

import sys

sys.path.insert(0, "/home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer")

from data import BioASQDataset, BioASQMultiNegativePairwiseIterator
from sampler import BasicSampler


class RawTextPreprocessing:
    """Preprocessing that keeps raw text instead of converting to tensors."""

    def __call__(self, sample):
        sample = sample.copy()
        if "label" in sample:
            sample["labels"] = sample.pop("label")
        return sample


class IterableMultiNegIterator(BioASQMultiNegativePairwiseIterator):
    """Wrapper to make the iterator actually iterable."""

    def __iter__(self):
        return self


def create_mock_dataset():
    """Create a small mock dataset for demonstration."""
    return {
        "query_001": {
            "question": "What is the treatment for diabetes?",
            0: [  # Negative docs
                {
                    "id": "doc_001",
                    "score": 25.5,
                    "text": "Diabetes is a chronic condition...",
                },
                {
                    "id": "doc_002",
                    "score": 24.1,
                    "text": "Metformin is commonly prescribed...",
                },
                {
                    "id": "doc_003",
                    "score": 22.8,
                    "text": "Regular exercise helps prevent...",
                },
                {
                    "id": "doc_004",
                    "score": 21.5,
                    "text": "Type 2 diabetes risk factors include...",
                },
                {
                    "id": "doc_005",
                    "score": 20.2,
                    "text": "Diet and lifestyle changes...",
                },
            ],
            1: [  # Positive docs
                {
                    "id": "doc_006",
                    "text": "Insulin therapy remains the primary treatment for type 1 diabetes...",
                },
            ],
        },
        "query_002": {
            "question": "What are the symptoms of COVID-19?",
            0: [  # Negative docs
                {
                    "id": "doc_007",
                    "score": 28.2,
                    "text": "The common cold and flu share many symptoms...",
                },
                {
                    "id": "doc_008",
                    "score": 26.7,
                    "text": "Vaccination programs have reduced severity...",
                },
                {
                    "id": "doc_009",
                    "score": 25.3,
                    "text": "Public health measures were implemented...",
                },
                {
                    "id": "doc_010",
                    "score": 24.1,
                    "text": "Respiratory viruses spread through droplets...",
                },
            ],
            1: [  # Positive docs
                {
                    "id": "doc_011",
                    "text": "COVID-19 symptoms include fever, cough, fatigue, loss of taste or smell...",
                },
                {
                    "id": "doc_012",
                    "text": "Severe cases may present with pneumonia, acute respiratory distress...",
                },
            ],
        },
    }


def main():
    print("=" * 80)
    print("MULTI-NEGATIVE PAIRWISE ITERATOR DEMONSTRATION")
    print("=" * 80)

    # Create mock data
    print("\n1. Creating mock dataset...")
    dataset_dict = create_mock_dataset()
    print(f"   Created {len(dataset_dict)} queries")

    for q_id, q_data in dataset_dict.items():
        print(f"   - {q_id}: {len(q_data[0])} negs, {len(q_data[1])} pos")

    # Test with different num_neg_samples values
    for num_negs in [1, 3, 5]:
        print(f"\n{'='*80}")
        print(f"TESTING WITH num_neg_samples={num_negs}")
        print(f"{'='*80}")

        def _create_iterator(sample_preprocessing, sampler_class, num_neg_samples):
            return BioASQMultiNegativePairwiseIterator(
                sample_preprocessing=sample_preprocessing,
                sampler_class=sampler_class,
                num_neg_samples=num_neg_samples,
            )

        dataset = BioASQDataset(
            dataset=dataset_dict.copy(),
            iterator=_create_iterator(
                sample_preprocessing=RawTextPreprocessing(),
                sampler_class=BasicSampler,
                num_neg_samples=num_negs,
            ),
        )

        print(f"\n   Dataset length: {len(dataset)}")

        # Get first 2 samples
        iterator = iter(dataset)
        for i in range(2):
            try:
                sample = next(iterator)
            except StopIteration:
                break

            from typing import cast, Any
            _sample = cast(dict[str, Any], sample)
            pos_inputs = cast(dict[str, str], _sample["pos_inputs"])
            neg_inputs = cast(list, _sample["neg_inputs"])

            print(f"\n   Sample {i+1}:")
            print(f"   - Query: {pos_inputs['query_text'][:50]}...")
            print(f"   - Positive doc ID: {pos_inputs['doc_text'][:20]}...")
            print(f"   - Number of negatives: {len(neg_inputs)}")

            for j, neg in enumerate(neg_inputs[:3]):  # Show first 3 negatives
                neg_entry = cast(dict[str, str], neg)
                print(f"      Neg {j+1}: {neg_entry['doc_text'][:30]}...")

            if len(neg_inputs) > 3:
                print(f"      ... and {len(neg_inputs) - 3} more")

    print("\n" + "=" * 80)
    print("COMPARISON WITH STANDARD PAIRWISE ITERATOR:")
    print("=" * 80)

    print("\nStandard BioASQPairwiseIterator (num_neg_samples=1, always):")
    print("   Output: {'pos_inputs': {...}, 'neg_inputs': {...}}  # Single neg")

    print("\nBioASQMultiNegativePairwiseIterator (num_neg_samples=N):")
    print(
        "   Output: {'pos_inputs': {...}, 'neg_inputs': [{...}, {...}, ...]}  # List of N negs"
    )

    print("\nBenefits of Multi-Negative:")
    print("   1. More diverse gradient signal per positive")
    print("   2. Better utilization of 999 available negatives")
    print("   3. Harder negative mining opportunities")
    print(
        "   4. Loss can be computed over all N pairs: sum(max(0, margin - (s_pos - s_neg_i)))"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
