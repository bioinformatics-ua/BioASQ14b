"""Evaluate snippet extraction quality.

Compares predicted snippets against gold snippets from BioASQ training data.
Computes token-level precision/recall/F1, exact substring match rate,
and JSON parse success rate.

Usage:
    python -m bioasq.snippets.evaluate \
        --predictions data/snippets/extracted_snippets.jsonl \
        --gold data/training14b/training14b.json \
        --inflated data/quality/training14b_inflated_clean_wContents.jsonl
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


# ---------------------------------------------------------------------------
# Token-level metrics
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


def token_f1(prediction_tokens: list[str], gold_tokens: list[str]) -> dict[str, float]:
    """Compute token-level precision, recall, F1."""
    if not prediction_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not prediction_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counter = Counter(prediction_tokens)
    gold_counter = Counter(gold_tokens)

    common = sum((pred_counter & gold_counter).values())
    precision = common / len(prediction_tokens) if prediction_tokens else 0.0
    recall = common / len(gold_tokens) if gold_tokens else 0.0

    if precision + recall == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def snippet_set_f1(
    predicted_snippets: list[str],
    gold_snippets: list[str],
) -> dict[str, float]:
    """Compute F1 over the union of all snippet tokens for a single question."""
    pred_tokens = []
    for s in predicted_snippets:
        pred_tokens.extend(_tokenize(s))

    gold_tokens = []
    for s in gold_snippets:
        gold_tokens.extend(_tokenize(s))

    return token_f1(pred_tokens, gold_tokens)


def exact_substring_rate(predicted_snippets: list[str], doc_text: str) -> float:
    """Fraction of predicted snippets that are exact substrings of doc_text."""
    if not predicted_snippets:
        return 1.0
    matched = sum(1 for s in predicted_snippets if s in doc_text)
    return matched / len(predicted_snippets)


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------


def _extract_pmid(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


@app.command()
def main(
    predictions: Annotated[Path, typer.Option(help="Predicted snippets JSONL")] = Path(
        "data/snippets/extracted_snippets.jsonl"
    ),
    gold: Annotated[Path, typer.Option(help="Gold training14b.json")] = Path(
        "data/training14b/training14b.json"
    ),
    inflated: Annotated[Path, typer.Option(help="Inflated JSONL with doc texts")] = Path(
        "data/quality/training14b_inflated_clean_wContents.jsonl"
    ),
) -> None:
    """Evaluate predicted snippets against gold."""
    # Load gold
    with gold.open() as f:
        gold_data = json.load(f)
    gold_map = {q["id"]: q for q in gold_data["questions"]}
    print(f"Gold: {len(gold_map)} questions")

    # Load inflated doc texts
    inflated_docs: dict[str, dict[str, str]] = {}
    with inflated.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            docs = {}
            for d in obj.get("documents", []):
                if isinstance(d, dict):
                    docs[d["id"]] = d["text"]
            inflated_docs[obj["id"]] = docs

    # Load predictions
    pred_map: dict[str, dict] = {}
    with predictions.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pred_map[obj["id"]] = obj
    print(f"Predictions: {len(pred_map)} questions")

    # Evaluate
    all_f1: list[float] = []
    all_precision: list[float] = []
    all_recall: list[float] = []
    all_substring_rates: list[float] = []
    total_predicted = 0
    total_gold = 0
    questions_evaluated = 0

    for qid, pred_q in pred_map.items():
        gold_q = gold_map.get(qid)
        if gold_q is None:
            continue

        # Get gold snippet texts
        gold_snips = [s["text"] for s in gold_q.get("snippets", [])]
        # Get predicted snippet texts
        pred_snips = [s["text"] if isinstance(s, dict) else s for s in pred_q.get("snippets", [])]

        total_predicted += len(pred_snips)
        total_gold += len(gold_snips)
        questions_evaluated += 1

        # Token F1
        metrics = snippet_set_f1(pred_snips, gold_snips)
        all_f1.append(metrics["f1"])
        all_precision.append(metrics["precision"])
        all_recall.append(metrics["recall"])

        # Exact substring rate
        doc_texts = inflated_docs.get(qid, {})
        all_doc_text = " ".join(doc_texts.values())
        if all_doc_text:
            rate = exact_substring_rate(pred_snips, all_doc_text)
            all_substring_rates.append(rate)

    # Report
    n = max(questions_evaluated, 1)
    print(f"\n{'=' * 50}")
    print(f"Questions evaluated:   {questions_evaluated}")
    print(f"Total predicted snips: {total_predicted}")
    print(f"Total gold snips:      {total_gold}")
    print(f"Avg predicted/q:       {total_predicted / n:.1f}")
    print(f"Avg gold/q:            {total_gold / n:.1f}")
    print(f"{'=' * 50}")
    print(f"Token Precision:       {sum(all_precision) / n:.4f}")
    print(f"Token Recall:          {sum(all_recall) / n:.4f}")
    print(f"Token F1:              {sum(all_f1) / n:.4f}")
    if all_substring_rates:
        print(f"Exact substring rate:  {sum(all_substring_rates) / len(all_substring_rates):.4f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    app()
