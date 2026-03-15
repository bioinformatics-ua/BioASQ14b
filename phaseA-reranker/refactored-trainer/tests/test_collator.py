"""
Unit tests for collator classes.

Run with: python -m pytest tests/test_collator.py -v
Or:       python tests/test_collator.py
"""

import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
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


@pytest.fixture
def tokenizer():
    """Shared tokenizer for all tests."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


# --- RankingCollator ---


def test_ranking_collator_basic(tokenizer: AutoTokenizer) -> None:
    """RankingCollator pads inputs and passes through metadata."""
    collator = RankingCollator(tokenizer)
    batch = [
        {"input_ids": [101, 2023, 2003, 102], "attention_mask": [1, 1, 1, 1], "labels": 1},
        {"input_ids": [101, 2054, 102], "attention_mask": [1, 1, 1], "labels": 0},
    ]
    out = collator(batch)
    assert "inputs" in out
    assert "labels" in out
    assert out["inputs"]["input_ids"].shape[0] == 2
    assert out["inputs"]["attention_mask"].shape[0] == 2
    assert out["labels"] == [1, 0]


def test_ranking_collator_with_token_type_ids(tokenizer: AutoTokenizer) -> None:
    """RankingCollator auto-detects token_type_ids when present."""
    enc = tokenizer("query text", "document text")
    batch = [
        dict(enc),
        dict(tokenizer("other query", "other doc")),
    ]
    collator = RankingCollator(tokenizer)
    out = collator(batch)
    assert "inputs" in out
    assert "input_ids" in out["inputs"]
    assert "attention_mask" in out["inputs"]
    assert "token_type_ids" in out["inputs"]


def test_ranking_collator_explicit_model_inputs(tokenizer: AutoTokenizer) -> None:
    """RankingCollator respects explicit model_inputs."""
    enc = tokenizer("query", "doc")
    batch = [dict(enc), dict(enc)]
    batch[0]["labels"] = 1
    batch[1]["labels"] = 0
    collator = RankingCollator(tokenizer, model_inputs={"input_ids", "attention_mask"})
    out = collator(batch)
    assert "inputs" in out
    assert "labels" in out
    assert "token_type_ids" not in out["inputs"]


def test_ranking_collator_max_length(tokenizer: AutoTokenizer) -> None:
    """RankingCollator pads to max_length when padding='max_length'."""
    enc = tokenizer("short", "doc")  # Short sequence
    batch = [dict(enc), dict(enc)]
    collator = RankingCollator(tokenizer, padding="max_length", max_length=64)
    out = collator(batch)
    assert out["inputs"]["input_ids"].shape[1] == 64


# --- PairwiseCollator ---


def test_pairwise_collator_basic(tokenizer: AutoTokenizer) -> None:
    """PairwiseCollator pads pos and neg inputs separately."""
    pos_enc = tokenizer("query", "positive doc")
    neg_enc = tokenizer("query", "negative doc")
    batch = [
        {"pos_inputs": dict(pos_enc), "neg_inputs": dict(neg_enc)},
        {"pos_inputs": dict(pos_enc), "neg_inputs": dict(neg_enc)},
    ]
    collator = PairwiseCollator(tokenizer)
    out = collator(batch)
    assert "pos_inputs" in out
    assert "neg_inputs" in out
    assert out["pos_inputs"]["input_ids"].shape[0] == 2
    assert out["neg_inputs"]["input_ids"].shape[0] == 2
    assert isinstance(out["pos_inputs"]["input_ids"], torch.Tensor)
    assert isinstance(out["neg_inputs"]["input_ids"], torch.Tensor)


def test_multi_negative_pairwise_collator(tokenizer: AutoTokenizer) -> None:
    """MultiNegativePairwiseCollator pads pos + list of neg batches."""
    pos1 = tokenizer("q", "pos doc 1")
    neg1a = tokenizer("q", "neg 1a")
    neg1b = tokenizer("q", "neg 1b")
    pos2 = tokenizer("query", "pos doc 2")
    neg2a = tokenizer("query", "neg 2a")
    neg2b = tokenizer("query", "neg 2b")
    batch = [
        {"pos_inputs": dict(pos1), "neg_inputs": [dict(neg1a), dict(neg1b)]},
        {"pos_inputs": dict(pos2), "neg_inputs": [dict(neg2a), dict(neg2b)]},
    ]
    collator = MultiNegativePairwiseCollator(tokenizer)
    out = collator(batch)
    assert "pos_inputs" in out
    assert "neg_inputs" in out
    assert out["pos_inputs"]["input_ids"].shape[0] == 2
    assert len(out["neg_inputs"]) == 2  # 2 negatives per sample
    assert out["neg_inputs"][0]["input_ids"].shape[0] == 2
    assert out["neg_inputs"][1]["input_ids"].shape[0] == 2


def test_pairwise_collator_variable_lengths(tokenizer: AutoTokenizer) -> None:
    """PairwiseCollator handles variable-length sequences."""
    batch = [
        {"pos_inputs": tokenizer("short", "doc"), "neg_inputs": tokenizer("a", "b")},
        {"pos_inputs": tokenizer("longer query here", "longer doc here"), "neg_inputs": tokenizer("x", "y")},
    ]
    collator = PairwiseCollator(tokenizer)
    out = collator(batch)
    assert out["pos_inputs"]["input_ids"].shape[0] == 2
    assert out["neg_inputs"]["input_ids"].shape[0] == 2


# --- RankingCollatorForCasualLM ---


def test_ranking_collator_casual_lm(tokenizer: AutoTokenizer) -> None:
    """RankingCollatorForCasualLM uses only input_ids and attention_mask."""
    enc = tokenizer("query", "doc")
    batch = [dict(enc), dict(enc)]
    collator = RankingCollatorForCasualLM(tokenizer)
    out = collator(batch)
    assert "inputs" in out
    assert "input_ids" in out["inputs"]
    assert "attention_mask" in out["inputs"]
    # BERT tokenizer returns token_type_ids; collator filters to input_ids + attention_mask
    assert "token_type_ids" not in out["inputs"]


# --- RankingCollatorForSeq2Seq ---


def test_ranking_collator_seq2seq(tokenizer: AutoTokenizer) -> None:
    """RankingCollatorForSeq2Seq expects decoder_input_ids when provided."""
    enc = tokenizer("query", "doc")
    enc["decoder_input_ids"] = enc["input_ids"][:]  # Mimic seq2seq
    batch = [dict(enc), dict(enc)]
    collator = RankingCollatorForSeq2Seq(tokenizer)
    out = collator(batch)
    assert "inputs" in out
    assert "decoder_input_ids" in out["inputs"]


# --- SentenceCollator ---


def _make_sentence_batch(tokenizer: AutoTokenizer, n_samples: int = 2) -> list[SentenceBatchSample]:
    """Create batch of multi-sentence samples for SentenceCollator."""
    samples: list[SentenceBatchSample] = []
    for _ in range(n_samples):
        s1 = tokenizer("sentence one", add_special_tokens=True)
        s2 = tokenizer("sentence two", add_special_tokens=True)
        samples.append({
            "input_ids": [s1["input_ids"], s2["input_ids"]],
            "attention_mask": [s1["attention_mask"], s2["attention_mask"]],
        })
    return samples


def test_sentence_collator_basic(tokenizer: AutoTokenizer) -> None:
    """SentenceCollator flattens multi-sentence samples and tracks sentences_count."""
    batch = _make_sentence_batch(tokenizer)
    collator = SentenceCollator(tokenizer)
    out = collator(batch)
    assert "input_ids" in out
    assert "attention_mask" in out
    assert "sentences_count" in out
    assert out["sentences_count"] == [2, 2]  # 2 sentences per sample
    total_sentences = sum(out["sentences_count"])
    assert out["input_ids"].shape[0] == total_sentences


def test_sentence_collator_with_token_type_ids(tokenizer: AutoTokenizer) -> None:
    """SentenceCollator handles token_type_ids when present."""
    s1 = tokenizer("a", "b")
    s2 = tokenizer("c", "d")
    batch: list[SentenceBatchSample] = [{
        "input_ids": [s1["input_ids"], s2["input_ids"]],
        "attention_mask": [s1["attention_mask"], s2["attention_mask"]],
        "token_type_ids": [s1["token_type_ids"], s2["token_type_ids"]],
    }]
    collator = SentenceCollator(tokenizer)
    out = collator(batch)
    assert "token_type_ids" in out
    assert out["sentences_count"] == [2]


# --- PairwiseSentenceCollator ---


def test_pairwise_sentence_collator(tokenizer: AutoTokenizer) -> None:
    """PairwiseSentenceCollator applies SentenceCollator to pos and neg separately."""
    pos_batch = _make_sentence_batch(tokenizer)
    neg_batch = _make_sentence_batch(tokenizer)
    batch = [
        {"pos_inputs": pos_batch[0], "neg_inputs": neg_batch[0]},
        {"pos_inputs": pos_batch[1], "neg_inputs": neg_batch[1]},
    ]
    collator = PairwiseSentenceCollator(tokenizer)
    out = collator(batch)
    assert "pos_inputs" in out
    assert "neg_inputs" in out
    assert "sentences_count" in out["pos_inputs"]
    assert "sentences_count" in out["neg_inputs"]
    assert out["pos_inputs"]["sentences_count"] == [2, 2]
    assert out["neg_inputs"]["sentences_count"] == [2, 2]


# --- RankingSentenceCollator ---


def test_ranking_sentence_collator(tokenizer: AutoTokenizer) -> None:
    """RankingSentenceCollator preserves metadata while flattening sentences."""
    pos_batch = _make_sentence_batch(tokenizer)
    batch = [
        {"input_ids": pos_batch[0]["input_ids"], "attention_mask": pos_batch[0]["attention_mask"], "labels": 1},
        {"input_ids": pos_batch[1]["input_ids"], "attention_mask": pos_batch[1]["attention_mask"], "labels": 0},
    ]
    collator = RankingSentenceCollator(tokenizer)
    out = collator(batch)
    assert "inputs" in out
    assert "labels" in out
    assert out["labels"] == [1, 0]
    assert "sentences_count" in out["inputs"]
    assert out["inputs"]["sentences_count"] == [2, 2]


def test_ranking_sentence_collator_excludes_metadata_from_model_inputs(tokenizer: AutoTokenizer) -> None:
    """RankingSentenceCollator only feeds model keys to SentenceCollator."""
    pos_batch = _make_sentence_batch(tokenizer)
    batch = [
        {"input_ids": pos_batch[0]["input_ids"], "attention_mask": pos_batch[0]["attention_mask"], "doc_id": "d1", "labels": 1},
    ]
    collator = RankingSentenceCollator(tokenizer)
    out = collator(batch)
    assert "inputs" in out
    assert "doc_id" in out
    assert out["doc_id"] == ["d1"]
    assert out["labels"] == [1]


# --- Edge cases ---


def test_ranking_collator_empty_reminder(tokenizer: AutoTokenizer) -> None:
    """RankingCollator with no extra keys returns only inputs."""
    batch = [{"input_ids": [101, 102], "attention_mask": [1, 1]}]
    batch.append({"input_ids": [101, 2023, 102], "attention_mask": [1, 1, 1]})
    collator = RankingCollator(tokenizer, model_inputs={"input_ids", "attention_mask"})
    out = collator(batch)
    assert list(out.keys()) == ["inputs"]
    assert out["inputs"]["input_ids"].shape[0] == 2


def test_sentence_collator_single_sentence_per_sample(tokenizer: AutoTokenizer) -> None:
    """SentenceCollator works with one sentence per sample."""
    s = tokenizer("single", add_special_tokens=True)
    batch: list[SentenceBatchSample] = [
        {"input_ids": [s["input_ids"]], "attention_mask": [s["attention_mask"]]},
        {"input_ids": [s["input_ids"]], "attention_mask": [s["attention_mask"]]},
    ]
    collator = SentenceCollator(tokenizer)
    out = collator(batch)
    assert out["sentences_count"] == [1, 1]
    assert out["input_ids"].shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
