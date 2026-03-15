"""
Comprehensive demo of all collator classes with detailed printed output.

Shows input structure, what each collator does, and output structure.
Run with: python tests/test_collator_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer

from collator import (
    MultiNegativePairwiseCollator,
    PairwiseCollator,
    PairwiseSentenceCollator,
    RankingCollator,
    RankingCollatorForCasualLM,
    RankingCollatorForSeq2Seq,
    RankingSentenceCollator,
    SentenceBatchSample,
    SentenceCollator,
)


def sep(title: str, char: str = "=") -> None:
    print(f"\n{char * 80}")
    print(f"  {title}")
    print(char * 80)


def subsep(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print("\n")
    sep("COLLATOR COMPREHENSIVE DEMO", "=")
    print("""
Each collator receives a BATCH (list of samples) from a DataLoader and returns
a dictionary of padded tensors + metadata ready for the model.

Batch structure varies by collator. We'll show INPUT → COLLATOR → OUTPUT for each.
""")

    # =========================================================================
    # 1. RankingCollator
    # =========================================================================
    sep("1. RankingCollator", "-")
    print("""
Use case: Single (query, doc) pair per sample. For evaluation or pointwise training.
Each sample = one tokenized (query, doc) encoding.
""")

    subsep("Input batch (2 samples)")
    sample1 = tokenizer("What is diabetes?", "Diabetes is a chronic disease.")
    sample2 = tokenizer("COVID symptoms?", "Fever and cough are common.")
    sample1_dict = dict(sample1)
    sample2_dict = dict(sample2)
    sample1_dict["labels"] = 1  # relevant
    sample2_dict["labels"] = 0  # not relevant
    sample1_dict["doc_id"] = "doc_001"
    sample2_dict["doc_id"] = "doc_002"
    batch = [sample1_dict, sample2_dict]
    print(f"  Sample 1 keys: {list(batch[0].keys())}")
    print(f"  Sample 2 keys: {list(batch[1].keys())}")
    print(f"  input_ids lengths: {len(batch[0]['input_ids'])}, {len(batch[1]['input_ids'])}")

    subsep("Collator call")
    collator = RankingCollator(tokenizer)
    out = collator(batch)

    subsep("Output")
    print(f"  Keys: {list(out.keys())}")
    print(f"  inputs['input_ids'] shape: {out['inputs']['input_ids'].shape}")
    print(f"  inputs['attention_mask'] shape: {out['inputs']['attention_mask'].shape}")
    print(f"  inputs['token_type_ids'] shape: {out['inputs']['token_type_ids'].shape}")
    print(f"  labels (passed through): {out['labels']}")
    print(f"  doc_id (passed through): {out['doc_id']}")
    print("""
  → Model inputs go under "inputs", metadata (labels, doc_id) passed through as-is.
  → Sequences padded to same length within the batch.
""")

    # =========================================================================
    # 2. PairwiseCollator
    # =========================================================================
    sep("2. PairwiseCollator", "-")
    print("""
Use case: Pairwise training. Each sample = (pos_inputs, neg_inputs).
For contrastive / margin loss: score(query, pos_doc) vs score(query, neg_doc).
""")

    subsep("Input batch (2 samples)")
    pos1 = tokenizer("query A", "relevant document")
    neg1 = tokenizer("query A", "irrelevant document")
    pos2 = tokenizer("query B", "relevant doc")
    neg2 = tokenizer("query B", "wrong doc")
    batch = [
        {"pos_inputs": dict(pos1), "neg_inputs": dict(neg1)},
        {"pos_inputs": dict(pos2), "neg_inputs": dict(neg2)},
    ]
    print(f"  Each sample has: pos_inputs, neg_inputs")
    print(f"  pos_inputs[0] input_ids len: {len(batch[0]['pos_inputs']['input_ids'])}")
    print(f"  neg_inputs[0] input_ids len: {len(batch[0]['neg_inputs']['input_ids'])}")

    subsep("Collator call")
    collator = PairwiseCollator(tokenizer)
    out = collator(batch)

    subsep("Output")
    print(f"  Keys: {list(out.keys())}")
    print(f"  pos_inputs['input_ids'] shape: {out['pos_inputs']['input_ids'].shape}")
    print(f"  neg_inputs['input_ids'] shape: {out['neg_inputs']['input_ids'].shape}")
    print("""
  → Positives and negatives padded separately into two batched tensors.
  → Forward: scores_pos = model(**pos_inputs), scores_neg = model(**neg_inputs)
""")

    # =========================================================================
    # 2b. MultiNegativePairwiseCollator (BioASQMultiNegativePairwiseIterator)
    # =========================================================================
    sep("2b. MultiNegativePairwiseCollator (BioASQMultiNegativePairwiseIterator)", "-")
    print("""
Use case: 1 positive + N negatives per sample. For BioASQMultiNegativePairwiseIterator.
neg_inputs is a LIST of tokenized samples (not a single dict).
""")

    subsep("Input batch (2 samples, 2 negatives each)")
    pos1 = tokenizer("query A", "relevant doc")
    neg1a = tokenizer("query A", "irrelevant doc A")
    neg1b = tokenizer("query A", "irrelevant doc B")
    pos2 = tokenizer("query B", "good doc")
    neg2a = tokenizer("query B", "bad doc A")
    neg2b = tokenizer("query B", "bad doc B")
    batch = [
        {"pos_inputs": dict(pos1), "neg_inputs": [dict(neg1a), dict(neg1b)]},
        {"pos_inputs": dict(pos2), "neg_inputs": [dict(neg2a), dict(neg2b)]},
    ]
    print(f"  Sample 1: 1 pos, {len(batch[0]['neg_inputs'])} negs")
    print(f"  Sample 2: 1 pos, {len(batch[1]['neg_inputs'])} negs")

    subsep("Collator call")
    collator = MultiNegativePairwiseCollator(tokenizer)
    out = collator(batch)

    subsep("Output")
    print(f"  pos_inputs['input_ids'] shape: {out['pos_inputs']['input_ids'].shape}")
    print(f"  neg_inputs: list of {len(out['neg_inputs'])} BatchEncodings")
    for i, neg in enumerate(out["neg_inputs"]):
        print(f"    neg_inputs[{i}]['input_ids'] shape: {neg['input_ids'].shape}")
    print("""
  → neg_inputs[i] = batch of the i-th negative from each sample.
  → scores_pos = model(**pos_inputs)
  → scores_negs = [model(**neg_i) for neg_i in neg_inputs]
  → loss = sum(margin_loss(scores_pos, s_neg) for s_neg in scores_negs) / N
""")

    # =========================================================================
    # 3. RankingCollatorForCasualLM
    # =========================================================================
    sep("3. RankingCollatorForCasualLM", "-")
    print("""
Use case: Decoder-only models (GPT-style). No token_type_ids.
""")

    subsep("Input batch")
    enc = tokenizer("prompt", "completion")
    batch = [dict(enc), dict(enc)]
    print(f"  Sample has token_type_ids: {'token_type_ids' in batch[0]}")

    subsep("Collator call (filters to input_ids + attention_mask only)")
    collator = RankingCollatorForCasualLM(tokenizer)
    out = collator(batch)

    subsep("Output")
    print(f"  inputs keys: {list(out['inputs'].keys())}")
    print(f"  token_type_ids in inputs: {'token_type_ids' in out['inputs']}")
    print("  → token_type_ids excluded (decoder-only models don't use it)")

    # =========================================================================
    # 4. RankingCollatorForSeq2Seq
    # =========================================================================
    sep("4. RankingCollatorForSeq2Seq", "-")
    print("""
Use case: Encoder-decoder models (T5, BART). Needs decoder_input_ids.
""")

    subsep("Input batch (mock decoder_input_ids)")
    enc = tokenizer("query", "doc")
    enc["decoder_input_ids"] = enc["input_ids"][:]
    batch = [dict(enc), dict(enc)]
    print(f"  Sample keys: {list(batch[0].keys())}")

    subsep("Collator call")
    collator = RankingCollatorForSeq2Seq(tokenizer)
    out = collator(batch)

    subsep("Output")
    print(f"  inputs keys: {list(out['inputs'].keys())}")
    print(f"  decoder_input_ids shape: {out['inputs']['decoder_input_ids'].shape}")

    # =========================================================================
    # 5. SentenceCollator
    # =========================================================================
    sep("5. SentenceCollator", "-")
    print("""
Use case: Multi-sentence per sample. Long docs split into chunks.
Each sample = dict with LISTS of encodings (multiple sentences).
Collator FLATTENS all sentences, pads them, and tracks sentences_count per sample.
""")

    subsep("Input batch (2 samples, 2 sentences each)")
    s1a = tokenizer("first sentence of doc 1", add_special_tokens=True)
    s1b = tokenizer("second sentence of doc 1", add_special_tokens=True)
    s2a = tokenizer("first sentence of doc 2", add_special_tokens=True)
    s2b = tokenizer("second sentence of doc 2", add_special_tokens=True)
    batch: list[SentenceBatchSample] = [
        {
            "input_ids": [s1a["input_ids"], s1b["input_ids"]],
            "attention_mask": [s1a["attention_mask"], s1b["attention_mask"]],
        },
        {
            "input_ids": [s2a["input_ids"], s2b["input_ids"]],
            "attention_mask": [s2a["attention_mask"], s2b["attention_mask"]],
        },
    ]
    print(f"  Sample 1: {len(batch[0]['input_ids'])} sentences")
    print(f"  Sample 2: {len(batch[1]['input_ids'])} sentences")
    print(f"  Total sentences: 4")

    subsep("Collator call")
    collator = SentenceCollator(tokenizer)
    out = collator(batch)

    subsep("Output")
    print(f"  input_ids shape: {out['input_ids'].shape}  (batch = 4 flattened sentences)")
    print(f"  attention_mask shape: {out['attention_mask'].shape}")
    print(f"  sentences_count: {out['sentences_count']}  (2 per original sample)")
    print("""
  → sentences_count tells you: sample 0 had 2 sentences, sample 1 had 2.
  → Use to reconstruct which embeddings belong to which original sample.
""")

    # =========================================================================
    # 6. PairwiseSentenceCollator
    # =========================================================================
    sep("6. PairwiseSentenceCollator", "-")
    print("""
Use case: Pairwise training with multi-sentence documents.
Each sample = pos_inputs (multi-sent) + neg_inputs (multi-sent).
Applies SentenceCollator to pos and neg separately.
""")

    subsep("Input batch")
    pos_sample = {"input_ids": [s1a["input_ids"], s1b["input_ids"]], "attention_mask": [s1a["attention_mask"], s1b["attention_mask"]]}
    neg_sample = {"input_ids": [s2a["input_ids"], s2b["input_ids"]], "attention_mask": [s2a["attention_mask"], s2b["attention_mask"]]}
    batch = [
        {"pos_inputs": pos_sample, "neg_inputs": neg_sample},
        {"pos_inputs": pos_sample, "neg_inputs": neg_sample},
    ]
    print(f"  2 samples, each with pos_inputs (2 sents) and neg_inputs (2 sents)")

    subsep("Collator call")
    collator = PairwiseSentenceCollator(tokenizer)
    out = collator(batch)

    subsep("Output")
    print(f"  pos_inputs['input_ids'] shape: {out['pos_inputs']['input_ids'].shape}")
    print(f"  pos_inputs['sentences_count']: {out['pos_inputs']['sentences_count']}")
    print(f"  neg_inputs['input_ids'] shape: {out['neg_inputs']['input_ids'].shape}")
    print(f"  neg_inputs['sentences_count']: {out['neg_inputs']['sentences_count']}")
    print("  → pos and neg each get their own flattened + padded tensors")

    # =========================================================================
    # 7. RankingSentenceCollator
    # =========================================================================
    sep("7. RankingSentenceCollator", "-")
    print("""
Use case: Ranking with multi-sentence docs + metadata passthrough.
Combines SentenceCollator (flatten sentences) + RankingCollator (pass metadata).
""")

    subsep("Input batch")
    batch = [
        {"input_ids": [s1a["input_ids"], s1b["input_ids"]], "attention_mask": [s1a["attention_mask"], s1b["attention_mask"]], "labels": 1, "doc_id": "d1"},
        {"input_ids": [s2a["input_ids"], s2b["input_ids"]], "attention_mask": [s2a["attention_mask"], s2b["attention_mask"]], "labels": 0, "doc_id": "d2"},
    ]
    print(f"  Sample keys: input_ids, attention_mask, labels, doc_id")
    print(f"  Model keys (flattened): input_ids, attention_mask")
    print(f"  Metadata (passed through): labels, doc_id")

    subsep("Collator call")
    collator = RankingSentenceCollator(tokenizer)
    out = collator(batch)

    subsep("Output")
    print(f"  Keys: {list(out.keys())}")
    print(f"  inputs (flattened sentences): input_ids shape {out['inputs']['input_ids'].shape}")
    print(f"  inputs['sentences_count']: {out['inputs']['sentences_count']}")
    print(f"  labels: {out['labels']}")
    print(f"  doc_id: {out['doc_id']}")
    print("  → Model gets padded sentences; you keep labels/doc_id for loss/metrics")

    # =========================================================================
    # Summary
    # =========================================================================
    sep("SUMMARY: Which collator when?", "=")
    print("""
  RankingCollator              → 1 encoding per sample, eval or pointwise train
  PairwiseCollator              → (pos, neg) pairs for contrastive training
  MultiNegativePairwiseCollator → 1 pos + N negs (BioASQMultiNegativePairwiseIterator)
  RankingCollatorForCasualLM    → Same as Ranking, decoder-only (no token_type_ids)
  RankingCollatorForSeq2Seq     → Same as Ranking, encoder-decoder (decoder_input_ids)
  SentenceCollator              → Multi-sentence per sample, flattens all
  PairwiseSentenceCollator     → Pairwise + multi-sentence
  RankingSentenceCollator      → Multi-sentence + metadata passthrough
""" + "=" * 80)


if __name__ == "__main__":
    main()
