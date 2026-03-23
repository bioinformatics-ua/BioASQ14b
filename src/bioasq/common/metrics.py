"""Evaluation metrics for the BioASQ pipeline.

Merges Phase A ranx-based retrieval metrics with Phase B answer-quality
metrics (ROUGE, F1, MRR, yes/no accuracy).  All functions accept abstract
parameter types and return concrete containers.
"""

from __future__ import annotations

import re
import string
from collections.abc import Mapping, Sequence

from rouge_score import rouge_scorer


# ═══════════════════════════════════════════════════════════════════════════
# Phase A — retrieval metrics (via ranx)
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_RETRIEVAL_METRICS: list[str] = [
    "ndcg@5",
    "mrr",
    "recall@10",
    "recall@100",
    "map@10",
    "map-bioasq@10",
]


def evaluate_retrieval_run(
    run_dict: Mapping[str, Mapping[str, float]],
    qrels_dict: Mapping[str, Mapping[str, int]],
    metrics: Sequence[str] | None = None,
    per_file_results: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate a retrieval run against qrels using ranx.

    Parameters
    ----------
    run_dict:
        ``{question_id: {doc_id: score}}``.
    qrels_dict:
        ``{question_id: {doc_id: relevance}}``.
    metrics:
        Metric names understood by :func:`ranx.evaluate`.
        Defaults to :data:`DEFAULT_RETRIEVAL_METRICS`.
    per_file_results:
        Optional ``{filename: [qid, …]}`` for per-file breakdowns.

    Returns
    -------
    ``{"total": {metric: value}, "file.json": {metric: value}, …}``
    """
    from ranx import Qrels, Run, evaluate  # heavy import; keep lazy

    used_metrics: Sequence[str] = metrics if metrics is not None else DEFAULT_RETRIEVAL_METRICS
    qrels: Qrels = Qrels(dict(qrels_dict))
    run: Run = Run(dict(run_dict))
    results: dict[str, dict[str, float]] = {}
    results["total"] = evaluate(qrels, run, list(used_metrics))

    if per_file_results is not None:
        for filename, qids in per_file_results.items():
            qid_set: set[str] = set(qids)
            sub_qrels: dict[str, dict[str, int]] = {
                k: dict(v) for k, v in qrels_dict.items() if k in qid_set
            }
            sub_run: dict[str, dict[str, float]] = {
                k: dict(v) for k, v in run_dict.items() if k in qid_set
            }
            if sub_qrels and sub_run:
                results[filename] = evaluate(
                    Qrels(sub_qrels), Run(sub_run), list(used_metrics)
                )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase B — answer-quality metrics
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------


def _normalize_answer(s: str) -> str:
    """Lower-case, remove punctuation, articles, extra whitespace."""
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def _gold_exact_to_normalised_set(
    gold_exact: list[str] | list[list[str]] | str | None,
) -> set[str]:
    """Normalise gold exact answers into a comparable set of strings.

    Handles the different BioASQ exact_answer formats:
    - ``str``: single answer         → ``{normalised}``
    - ``list[str]``: flat list        → ``{normalised, …}``
    - ``list[list[str]]``: synonyms   → ``{normalised, …}``  (flattened)
    """
    if gold_exact is None:
        return set()
    if isinstance(gold_exact, str):
        return {_normalize_answer(gold_exact)}
    result: set[str] = set()
    for item in gold_exact:
        if isinstance(item, list):
            for sub in item:
                normalised: str = _normalize_answer(str(sub))
                if normalised:
                    result.add(normalised)
        else:
            normalised = _normalize_answer(str(item))
            if normalised:
                result.add(normalised)
    return result


# ---------------------------------------------------------------------------
# Per-type evaluation functions
# ---------------------------------------------------------------------------


def accuracy_yesno(
    predictions: Mapping[str, str],
    gold: Mapping[str, str],
) -> dict[str, float]:
    """Yes/no question macro accuracy and per-class F1.

    Parameters
    ----------
    predictions:
        ``{question_id: "yes"|"no"}``.
    gold:
        ``{question_id: "yes"|"no"}``.
    """
    if not predictions:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    tp_yes: int = 0
    fp_yes: int = 0
    fn_yes: int = 0
    tp_no: int = 0
    fp_no: int = 0
    fn_no: int = 0
    correct: int = 0

    for qid, pred in predictions.items():
        truth: str = gold.get(qid, "")
        pred_norm: str = pred.strip().lower()
        truth_norm: str = truth.strip().lower()

        if pred_norm == truth_norm:
            correct += 1

        if pred_norm == "yes":
            if truth_norm == "yes":
                tp_yes += 1
            else:
                fp_yes += 1
        else:
            if truth_norm == "yes":
                fn_yes += 1

        if pred_norm == "no":
            if truth_norm == "no":
                tp_no += 1
            else:
                fp_no += 1
        else:
            if truth_norm == "no":
                fn_no += 1

    def _f1(tp: int, fp: int, fn: int) -> float:
        if tp == 0:
            return 0.0
        precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall: float = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    f1_yes: float = _f1(tp_yes, fp_yes, fn_yes)
    f1_no: float = _f1(tp_no, fp_no, fn_no)

    return {
        "accuracy": correct / len(predictions),
        "macro_f1": (f1_yes + f1_no) / 2,
    }


def mrr_factoid(
    predictions: Mapping[str, Sequence[str]],
    gold: Mapping[str, list[str] | list[list[str]] | str | None],
) -> dict[str, float]:
    """Mean Reciprocal Rank for factoid questions.

    Parameters
    ----------
    predictions:
        ``{question_id: [candidate_1, candidate_2, …]}``.
    gold:
        ``{question_id: exact_answer}`` in any BioASQ format.
    """
    if not predictions:
        return {"mrr": 0.0}

    total_rr: float = 0.0
    for qid, pred_list in predictions.items():
        gold_set: set[str] = _gold_exact_to_normalised_set(gold.get(qid))
        if not gold_set:
            continue
        for rank, candidate in enumerate(pred_list, start=1):
            if _normalize_answer(candidate) in gold_set:
                total_rr += 1.0 / rank
                break

    return {"mrr": total_rr / len(predictions) if predictions else 0.0}


def mean_f1_list(
    predictions: Mapping[str, Sequence[str] | Sequence[Sequence[str]]],
    gold: Mapping[str, list[str] | list[list[str]] | str | None],
) -> dict[str, float]:
    """Mean F1 for list-type questions.

    Parameters
    ----------
    predictions:
        ``{question_id: [entity, …]}`` or ``{question_id: [[synonym, …], …]}``.
    gold:
        ``{question_id: exact_answer}`` in any BioASQ format.
    """
    if not predictions:
        return {"mean_f1": 0.0, "mean_precision": 0.0, "mean_recall": 0.0}

    total_f1: float = 0.0
    total_p: float = 0.0
    total_r: float = 0.0

    for qid, pred_entities in predictions.items():
        gold_set: set[str] = _gold_exact_to_normalised_set(gold.get(qid))
        if not gold_set:
            continue

        # Flatten predicted entities
        pred_normalised: set[str] = set()
        for entity in pred_entities:
            if isinstance(entity, list):
                for synonym in entity:
                    pred_normalised.add(_normalize_answer(str(synonym)))
            else:
                pred_normalised.add(_normalize_answer(str(entity)))

        if not pred_normalised:
            continue

        tp: int = len(pred_normalised & gold_set)
        precision: float = tp / len(pred_normalised)
        recall: float = tp / len(gold_set)
        f1: float = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        total_f1 += f1
        total_p += precision
        total_r += recall

    n: int = len(predictions)
    return {
        "mean_f1": total_f1 / n,
        "mean_precision": total_p / n,
        "mean_recall": total_r / n,
    }


def rouge2_summary(
    predictions: Mapping[str, str],
    gold: Mapping[str, str | list[str]],
) -> dict[str, float]:
    """ROUGE-2 F-measure for summary-type questions.

    Parameters
    ----------
    predictions:
        ``{question_id: ideal_answer_text}``.
    gold:
        ``{question_id: gold_ideal_answer}`` — may be a string or list
        of reference strings (we take the best).
    """
    if not predictions:
        return {"rouge2_f": 0.0, "rouge2_p": 0.0, "rouge2_r": 0.0}

    scorer: rouge_scorer.RougeScorer = rouge_scorer.RougeScorer(
        ["rouge2"], use_stemmer=True
    )
    total_f: float = 0.0
    total_p: float = 0.0
    total_r: float = 0.0

    for qid, pred_text in predictions.items():
        gold_text: str | list[str] = gold.get(qid, "")
        if isinstance(gold_text, list):
            # Take best ROUGE among reference answers
            best_f: float = 0.0
            best_p: float = 0.0
            best_r: float = 0.0
            for ref in gold_text:
                scores = scorer.score(ref, pred_text)
                f: float = scores["rouge2"].fmeasure
                if f > best_f:
                    best_f = f
                    best_p = scores["rouge2"].precision
                    best_r = scores["rouge2"].recall
            total_f += best_f
            total_p += best_p
            total_r += best_r
        else:
            scores = scorer.score(gold_text, pred_text)
            total_f += scores["rouge2"].fmeasure
            total_p += scores["rouge2"].precision
            total_r += scores["rouge2"].recall

    n: int = len(predictions)
    return {
        "rouge2_f": total_f / n,
        "rouge2_p": total_p / n,
        "rouge2_r": total_r / n,
    }
