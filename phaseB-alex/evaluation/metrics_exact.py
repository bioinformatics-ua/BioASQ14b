"""
evaluation/metrics_exact.py

Exact-answer scoring functions for BioASQ Phase B, strictly per the
official evaluation document (D4.1, Section 3.1).

  yes/no  → accuracy (completeness) + macro-averaged F1 [official]  (eq. 3.1, 3.2)
  factoid → strict/lenient accuracy (completeness) + MRR [official]  (eq. 3.3–3.5)
  list    → mean precision, recall, F1 [official = mean F1]           (eq. 2.1–2.4)

Gold answer formats (as they appear in the val file):
  yes/no  : "yes" | "no"
  factoid : [["best answer", "synonym", ...]]   ← one entity, multiple synonyms
  list    : [["entity1"], ["entity2", "syn"], ...]  ← one inner list per entity
"""

import re
import string


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _flatten_synonyms(nested) -> set:
    """
    Flatten a nested gold answer list to a set of normalised strings.
    Handles both flat lists and lists-of-lists.
    """
    out = set()
    for item in nested:
        if isinstance(item, list):
            for s in item:
                out.add(_norm(str(s)))
        else:
            out.add(_norm(str(item)))
    return out


# ---------------------------------------------------------------------------
# Yes/No — accuracy + macro F1  (Section 3.1, eq. 3.1–3.2)
# ---------------------------------------------------------------------------

def score_yesno(predictions: dict, ground_truth: list[dict]) -> dict:
    """
    predictions  : {qid: {"exact_answer": "yes"|"no", ...}}
    ground_truth : list of question dicts (type == "yesno")

    Returns:
      accuracy   — Acc = c/n  (eq. 3.1, for completeness)
      f1_yes     — F1 for the "yes" class
      f1_no      — F1 for the "no" class
      macro_f1   — maF1 = (F1y + F1n) / 2  (eq. 3.2)  [OFFICIAL]
      n_scored
    """
    # tp/fp/fn per class
    counts = {
        "yes": {"tp": 0, "fp": 0, "fn": 0},
        "no":  {"tp": 0, "fp": 0, "fn": 0},
    }
    correct = 0
    n = 0

    for q in ground_truth:
        if q["type"] != "yesno":
            continue
        qid  = q["id"]
        gold = (q.get("exact_answer") or "").lower().strip()
        if gold not in ("yes", "no") or qid not in predictions:
            continue

        pred = (predictions[qid].get("exact_answer") or "").lower().strip()
        if pred not in ("yes", "no"):
            pred = "yes" if gold == "no" else "no"   # invalid → worst case

        n += 1
        if pred == gold:
            correct += 1
            counts[gold]["tp"] += 1
        else:
            counts[pred]["fp"] += 1
            counts[gold]["fn"] += 1

    def _f1(cls):
        tp, fp, fn = counts[cls]["tp"], counts[cls]["fp"], counts[cls]["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    f1y = _f1("yes")
    f1n = _f1("no")

    return {
        "accuracy":  correct / n if n else 0.0,   # eq. 3.1
        "f1_yes":    f1y,
        "f1_no":     f1n,
        "macro_f1":  (f1y + f1n) / 2,             # eq. 3.2  [OFFICIAL]
        "n_scored":  n,
    }


# ---------------------------------------------------------------------------
# Factoid — SAcc, LAcc, MRR  (Section 3.1, eq. 3.3–3.5)
# ---------------------------------------------------------------------------

def score_factoid(predictions: dict, ground_truth: list[dict]) -> dict:
    """
    predictions  : {qid: {"exact_answer": ["candidate1", ...], ...}}
    ground_truth : list of question dicts (type == "factoid")

    Gold exact_answer is a nested list [[entity, synonym, ...]] — all strings
    in all inner lists are treated as valid synonyms.

    Returns:
      strict_acc  — SAcc: gold is rank-1 candidate  (eq. 3.3, completeness)
      lenient_acc — LAcc: gold anywhere in top-5    (eq. 3.4, completeness)
      mrr         — MRR = mean(1/rank)              (eq. 3.5)  [OFFICIAL]
      n_scored
    """
    rr_sum      = 0.0
    strict_hits = 0
    lenient_hits = 0
    n = 0

    for q in ground_truth:
        if q["type"] != "factoid":
            continue
        qid        = q["id"]
        gold_raw   = q.get("exact_answer") or []
        if not gold_raw or qid not in predictions:
            continue

        gold_set   = _flatten_synonyms(gold_raw)   # all valid answer strings

        candidates = predictions[qid].get("exact_answer") or []
        if isinstance(candidates, str):
            candidates = [candidates]

        n += 1
        rr = 0.0
        for rank, cand in enumerate(candidates[:5], start=1):
            if _norm(str(cand)) in gold_set:
                rr = 1.0 / rank         # eq. 3.5
                if rank == 1:
                    strict_hits += 1    # eq. 3.3
                lenient_hits += 1       # eq. 3.4
                break

        rr_sum += rr

    return {
        "strict_acc":  strict_hits  / n if n else 0.0,   # eq. 3.3
        "lenient_acc": lenient_hits / n if n else 0.0,   # eq. 3.4
        "mrr":         rr_sum       / n if n else 0.0,   # eq. 3.5  [OFFICIAL]
        "n_scored":    n,
    }


# ---------------------------------------------------------------------------
# List — mean precision, recall, F1  (Section 3.1, eq. 2.1–2.4)
# ---------------------------------------------------------------------------

def score_list(predictions: dict, ground_truth: list[dict]) -> dict:
    """
    predictions  : {qid: {"exact_answer": [["entity1"], ["entity2"], ...], ...}}
    ground_truth : list of question dicts (type == "list")

    Gold exact_answer is a nested list — each inner list is one entity with
    possible synonyms.  Flatten to a set for comparison; synonyms count as
    matches; duplicates (same entity via different synonym) counted once.

    Returns:
      mean_precision  (completeness)
      mean_recall     (completeness)
      mean_f1         [OFFICIAL]
      n_scored
    """
    precisions = []
    recalls    = []
    f1_scores  = []

    for q in ground_truth:
        if q["type"] != "list":
            continue
        qid      = q["id"]
        gold_raw = q.get("exact_answer") or []
        if not gold_raw or qid not in predictions:
            continue

        gold_set = _flatten_synonyms(gold_raw)

        pred_raw = predictions[qid].get("exact_answer") or []
        pred_set = _flatten_synonyms(pred_raw)

        if not pred_set:
            precisions.append(0.0)
            recalls.append(0.0)
            f1_scores.append(0.0)
            continue

        tp = len(pred_set & gold_set)
        p  = tp / len(pred_set)
        r  = tp / len(gold_set) if gold_set else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0

        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

    n = len(f1_scores)
    return {
        "mean_precision": sum(precisions) / n if n else 0.0,
        "mean_recall":    sum(recalls)    / n if n else 0.0,
        "mean_f1":        sum(f1_scores)  / n if n else 0.0,   # [OFFICIAL]
        "n_scored":       n,
    }