"""
evaluation/metrics.py

Per-type scoring functions for BioASQ Phase B.

Each question type is scored differently by the competition:
    yesno   → macro F1 (F1 for "yes" class and "no" class, then averaged)
    factoid → MRR      (mean reciprocal rank of first correct answer in top-5)
    list    → mean F1  (precision/recall over entity sets per question)
    summary → ROUGE-2  (ideal answer only, no exact answer)

All types also produce a ROUGE-2 score for the ideal answer.

These functions are pure — they take predictions and gold dicts and return
floats. The evaluate.py script handles file I/O and calling these functions.
"""

import re
import string

from rouge_score import rouge_scorer as rouge_lib

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """
    Lowercase and strip punctuation/whitespace for fuzzy entity matching.
    Used when comparing exact answers in factoid and list questions.
    """
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)


def _rouge2_f1(prediction: str, references: list[str]) -> float:
    """
    Compute ROUGE-2 F1 between a predicted string and a list of reference strings.
    Returns the maximum score across all references (best-match strategy).
    """
    if not prediction or not references:
        return 0.0

    scorer = rouge_lib.RougeScorer(["rouge2"], use_stemmer=False)
    scores = [scorer.score(ref, prediction)["rouge2"].fmeasure for ref in references]
    return max(scores)


# ---------------------------------------------------------------------------
# Ideal answer metric (all types)
# ---------------------------------------------------------------------------


def rouge2_ideal(predictions: dict, ground_truth: list[dict]) -> dict:
    """
    Compute ROUGE-2 F1 for ideal answers across all questions.

    predictions:   {id: {"ideal_answer": str, ...}}
    ground_truth:  list of question dicts from the dataloader

    Returns a dict with per-question scores and the mean.
    """
    scores = {}

    for q in ground_truth:
        qid = q["id"]
        gold_list = q.get("ideal_answer") or []

        if qid not in predictions or not gold_list:
            continue

        predicted = predictions[qid].get("ideal_answer") or ""
        scores[qid] = _rouge2_f1(predicted, gold_list)

    mean = sum(scores.values()) / len(scores) if scores else 0.0
    return {"per_question": scores, "mean": mean}


# ---------------------------------------------------------------------------
# Yes/No — macro F1
# ---------------------------------------------------------------------------


def macro_f1_yesno(predictions: dict, ground_truth: list[dict]) -> dict:
    """
    Macro F1 for yes/no questions.

    Computes F1 separately for the "yes" class and "no" class, then averages.
    This penalises systems that always predict the majority class.

    predictions:   {id: {"exact_answer": "yes" | "no", ...}}
    ground_truth:  list of yesno question dicts
    """
    # Counts for each class: true positives, false positives, false negatives
    counts = {
        "yes": {"tp": 0, "fp": 0, "fn": 0},
        "no": {"tp": 0, "fp": 0, "fn": 0},
    }

    n_scored = 0

    for q in ground_truth:
        if q["type"] != "yesno":
            continue

        qid = q["id"]
        gold = (q.get("exact_answer") or "").lower().strip()

        if qid not in predictions or gold not in ("yes", "no"):
            continue

        pred = (predictions[qid].get("exact_answer") or "").lower().strip()
        if pred not in ("yes", "no"):
            # Invalid prediction — counts as wrong for the gold class
            pred = "yes" if gold == "no" else "no"

        n_scored += 1

        if pred == gold:
            counts[gold]["tp"] += 1
        else:
            # Predicted the wrong class
            counts[pred]["fp"] += 1
            counts[gold]["fn"] += 1

    def _f1(cls: str) -> float:
        tp = counts[cls]["tp"]
        fp = counts[cls]["fp"]
        fn = counts[cls]["fn"]
        precision: int | float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall: int | float = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    f1_yes = _f1("yes")
    f1_no = _f1("no")
    macro_f1 = (f1_yes + f1_no) / 2

    return {
        "f1_yes": f1_yes,
        "f1_no": f1_no,
        "macro_f1": macro_f1,
        "n_scored": n_scored,
    }


# ---------------------------------------------------------------------------
# Factoid — Mean Reciprocal Rank (MRR)
# ---------------------------------------------------------------------------


def mrr_factoid(predictions: dict, ground_truth: list[dict]) -> dict:
    """
    Mean Reciprocal Rank for factoid questions.

    Systems submit up to 5 candidate answers ranked by confidence.
    MRR = mean of 1/rank where rank is the position of the first correct answer.
    If no correct answer appears in the top 5, score is 0 for that question.

    Matching is case-insensitive with punctuation stripped.

    predictions:   {id: {"exact_answer": ["candidate1", "candidate2", ...], ...}}
    ground_truth:  list of factoid question dicts
    """
    reciprocal_ranks = []
    strict_hits = []  # 1 if gold is first candidate, else 0
    lenient_hits = []  # 1 if gold appears anywhere in top-5, else 0

    for q in ground_truth:
        if q["type"] != "factoid":
            continue

        qid = q["id"]
        gold_items = q.get("exact_answer") or []

        if qid not in predictions or not gold_items:
            continue

        # Gold answers — normalise for comparison
        # Factoid gold is a flat list like ["Bazex syndrome"]
        gold_set = {_normalise(g) for g in gold_items}

        candidates = predictions[qid].get("exact_answer") or []
        if isinstance(candidates, str):
            candidates = [candidates]

        rr = 0.0
        strict_correct = False  # gold is first candidate (SAcc)
        lenient_correct = False  # gold appears anywhere in top-5 (LAcc)

        for rank, candidate in enumerate(candidates[:5], start=1):
            if _normalise(str(candidate)) in gold_set:
                rr = 1.0 / rank
                lenient_correct = True
                if rank == 1:
                    strict_correct = True
                break

        reciprocal_ranks.append(rr)
        strict_hits.append(1 if strict_correct else 0)
        lenient_hits.append(1 if lenient_correct else 0)

    n = len(reciprocal_ranks)
    mrr = sum(reciprocal_ranks) / n if n else 0.0
    sacc = sum(strict_hits) / n if n else 0.0  # strict accuracy
    lacc = sum(lenient_hits) / n if n else 0.0  # lenient accuracy
    return {"mrr": mrr, "strict_acc": sacc, "lenient_acc": lacc, "n_scored": n}


# ---------------------------------------------------------------------------
# List — Mean F1
# ---------------------------------------------------------------------------


def mean_f1_list(predictions: dict, ground_truth: list[dict]) -> dict:
    """
    Mean F1 for list questions.

    For each question, compute precision and recall between the predicted entity
    set and the gold entity set, then take the F1. Average across all questions.

    Gold exact_answer is a list of lists: [["EGF"], ["betacellulin"], ...]
    Flatten to a set for comparison.

    predictions:   {id: {"exact_answer": [["entity1"], ["entity2"], ...], ...}}
    ground_truth:  list of list question dicts
    """
    f1_scores: list[float] = []

    for q in ground_truth:
        if q["type"] != "list":
            continue

        qid = q["id"]
        gold_items = q.get("exact_answer") or []

        if qid not in predictions or not gold_items:
            continue

        # Gold is nested: [["EGF"], ["betacellulin"]] — flatten and normalise
        gold_set = set()
        for item in gold_items:
            if isinstance(item, list):
                for subitem in item:
                    gold_set.add(_normalise(str(subitem)))
            else:
                gold_set.add(_normalise(str(item)))

        # Predictions can be nested or flat — flatten and normalise
        pred_items = predictions[qid].get("exact_answer") or []
        pred_set = set()
        for item in pred_items:
            if isinstance(item, list):
                for subitem in item:
                    pred_set.add(_normalise(str(subitem)))
            else:
                pred_set.add(_normalise(str(item)))

        if not pred_set:
            f1_scores.append(0.0)
            continue

        tp = len(pred_set & gold_set)
        precision = tp / len(pred_set)
        recall = tp / len(gold_set) if gold_set else 0.0

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return {"mean_f1": mean_f1, "n_scored": len(f1_scores)}


# ---------------------------------------------------------------------------
# Combined scorer
# ---------------------------------------------------------------------------


def evaluate_all(predictions: dict, ground_truth: list[dict]) -> dict:
    """
    Run all metrics and return a single results dict.

    This is the main entry point used by evaluate.py.
    """
    return {
        "rouge2_ideal": rouge2_ideal(predictions, ground_truth),
        "yesno": macro_f1_yesno(predictions, ground_truth),
        "factoid": mrr_factoid(predictions, ground_truth),
        "list": mean_f1_list(predictions, ground_truth),
    }
