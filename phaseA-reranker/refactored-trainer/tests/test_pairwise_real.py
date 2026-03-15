"""
Test script to demonstrate the pairwise iterator with REAL BioASQ data.
Shows raw text samples from the actual training dataset.
"""
from transformers import AutoTokenizer, TokenizersBackend

import sys
sys.path.insert(0, '/home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer')

import json
from data import BioASQDataset, BioASQPairwiseIterator
from sampler import ShifterSampler
from sample_preprocessing import BasicSamplePreprocessing
from typing import cast


# Create a sample preprocessing that just passes through raw text
# (no tokenization, no tensors)
class RawTextPreprocessing(BasicSamplePreprocessing):
    """Preprocessing that keeps raw text instead of converting to tensors."""
    def __init__(self,  model_max_length: int = -1) -> None:
        tokenizer = cast(TokenizersBackend, AutoTokenizer.from_pretrained("bert-base-uncased"))
        super().__init__(tokenizer, model_max_length)

    def __call__(self, sample: dict[str, str | int]) -> dict[str, object]:  # type: ignore[override]
        # sample = {"id": q_id, "query_text": q_text, "doc_text": doc_text, "label": label}
        # Just return as-is but rename "label" to "labels" to match expected interface
        result: dict[str, object] = dict(sample)
        if "label" in result:
            result["labels"] = result.pop("label")
        return result


class IterablePairwiseIterator(BioASQPairwiseIterator):
    """Wrapper to make the iterator actually iterable (adds __iter__ method)."""

    def __iter__(self):
        return self


def main():
    print("=" * 80)
    print("PAIRWISE ITERATOR WITH REAL BIOASQ DATA")
    print("=" * 80)

    # Paths to real data
    positives_path = "../../data/quality/training14b_inflated_clean_wContents.jsonl"
    all_data_path = "../../data/quality/hard_negatives_IA_clean.jsonl"

    print("\n1. Loading real BioASQ dataset...")
    print(f"   Positives: {positives_path}")
    print(f"   All data (with negatives): {all_data_path}")

    # Peek at the raw data structure
    print("\n2. Peeking at the raw data structure...")
    with open(positives_path, 'r') as f:
        first_pos = json.loads(f.readline())
        print(f"   Positives file keys: {list(first_pos.keys())}")
        print(f"   Query ID: {first_pos['id']}")
        print(f"   Query body: {first_pos['body'][:80]}...")
        print(f"   Number of positive docs: {len(first_pos['documents'])}")

    with open(all_data_path, 'r') as f:
        first_neg = json.loads(f.readline())
        print(f"\n   Negatives file keys: {list(first_neg.keys())}")
        print(f"   Query ID: {first_neg['id']}")
        print(f"   Number of neg_docs: {len(first_neg['neg_docs'])}")
        print(f"   First neg_doc keys: {list(first_neg['neg_docs'][0].keys()) if first_neg['neg_docs'] else 'N/A'}")

    # Count total queries
    print("\n3. Counting dataset size...")
    with open(positives_path, 'r') as f:
        total_queries = sum(1 for _ in f)
    print(f"   Total queries in dataset: {total_queries}")

    # Load the dataset
    print("\n4. Creating dataset with pairwise iterator (limited to first 50 queries for demo)...")

    # Manually create a small dataset matching the expected structure
    train_dataset = {}

    # First load all positive data (questions + positive docs)
    print("   Loading positive data...")
    with open(positives_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 50:  # Limit to 50 queries
                break
            sample = json.loads(line)
            train_dataset[sample["id"]] = {
                1: sample["documents"],  # positive docs at key 1
                "question": sample["body"],
            }

    # Then add negative docs from the all_data file
    print("   Loading negative data...")
    with open(all_data_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            if sample["id"] in train_dataset:
                train_dataset[sample["id"]][0] = []  # negative docs at key 0
                positive_ids = {doc["id"] for doc in train_dataset[sample["id"]][1]}
                for doc in sample["neg_docs"]:
                    if doc["id"] not in positive_ids:
                        train_dataset[sample["id"]][0].append(doc)

    print(f"   Loaded {len(train_dataset)} queries")

    # Show dataset structure
    print("\n5. Dataset structure (first 3 queries):")
    for i, (q_id, q_data) in enumerate(list(train_dataset.items())[:3]):
        print(f"\n   Query {i+1}: {q_id}")
        print(f"   - Question: {q_data['question'][:80]}...")
        print(f"   - 0 (negative/BM25) docs: {len(q_data[0])}")
        print(f"   - 1 (positive/gold) docs: {len(q_data[1])}")
    def _create_iterator(sample_preprocessing, sampler_class, num_neg_samples):
        return BioASQPairwiseIterator(
            sample_preprocessing=sample_preprocessing,
            sampler_class_type=sampler_class,
            num_neg_samples=num_neg_samples,
        )

    # Create the pairwise dataset
    dataset = BioASQDataset(
        dataset=train_dataset,
        iterator=_create_iterator(
            sample_preprocessing=RawTextPreprocessing(),
            sampler_class=ShifterSampler,
            num_neg_samples=1,
        ),
    )

    print("\n6. Dataset ready for iteration")
    print(f"   Dataset length: {len(dataset)} (equals total positive docs)")

    # Iterate and show samples
    print("\n7. First 5 pairwise samples:\n")
    print("=" * 80)

    from typing import cast, Any

    iterator = iter(dataset)
    for i, sample in enumerate(iterator):
        _sample = cast(dict[str, Any], sample)
        pos_inputs = cast(dict[str, str], _sample["pos_inputs"])
        neg_inputs = cast(dict[str, str], _sample["neg_inputs"])

        print(f"\nSAMPLE {i+1}:")
        print(f"  Query ID: {pos_inputs['id']}")
        print(f"  Query Text: {pos_inputs['query_text'][:100]}...")

        print("\n  POSITIVE DOC (relevant):")
        pos_text = pos_inputs['doc_text']
        print(f"    Doc ID: {pos_text[:20]}... (showing first 150 chars of text)")
        print(f"    Text: {pos_text[:150]}...")

        print("\n  NEGATIVE DOC (BM25-retrieved, not relevant):")
        neg_text = neg_inputs['doc_text']
        print(f"    Text: {neg_text[:150]}...")

        print("-" * 80)

    print("\n8. Summary:")
    print(f"   - Dataset has {len(train_dataset)} queries")
    total_pos = sum(len(q[1]) for q in train_dataset.values())
    total_neg = sum(len(q[0]) for q in train_dataset.values())
    print(f"   - Total positive docs: {total_pos}")
    print(f"   - Total negative docs: {total_neg}")
    print(f"   - Ratio: ~{total_neg//total_pos}:1 (negatives per positive)")
    print("=" * 80)


if __name__ == "__main__":
    main()
