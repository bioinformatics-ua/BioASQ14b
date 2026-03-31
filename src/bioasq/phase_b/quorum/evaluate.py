"""Evaluation system for the agent quorum debate pipeline.

Computes two categories of metrics:

**Answer quality** — how good are the final answers?
  - ROUGE-1 / ROUGE-2 / ROUGE-L for ideal answers (overall + per-type)
  - BERTScore F1 (optional, requires model download)
  - Exact-answer metrics reused from the existing evaluation module
    (yesno macro-F1, factoid MRR, list mean-F1)

**Debate process** — how does the debate itself behave?
  - Convergence speed (mean/median rounds to termination)
  - Consensus rate (fraction reaching ``strongly_agree`` unanimity)
  - Agreement trajectory (per-round mean agreement level)
  - Context efficiency (documents injected / documents available)
  - Per-focus agreement analysis
  - Per-model agreement analysis
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Annotated, Any

import orjson
import typer
from rouge_score import rouge_scorer as rouge_lib

from bioasq.phase_b.evaluation.metrics import (
    macro_f1_yesno,
    mean_f1_list,
    mrr_factoid,
)
from bioasq.phase_b.quorum.types import AGREEMENT_RANK

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_quorum_output(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    obj = orjson.loads(raw)
    if isinstance(obj, dict) and "questions" in obj:
        return obj["questions"]
    if isinstance(obj, list):
        return obj
    msg = f"Unexpected format in {path}"
    raise ValueError(msg)


def _load_gold(path: Path) -> list[dict[str, Any]]:
    """Load gold standard from JSONL or JSON wrapper."""
    raw = path.read_bytes()
    try:
        obj = orjson.loads(raw)
        if isinstance(obj, dict) and "questions" in obj:
            return obj["questions"]
        if isinstance(obj, list):
            return obj
    except orjson.JSONDecodeError:
        pass

    records: list[dict[str, Any]] = []
    for line in path.open("rb"):
        line = line.strip()
        if line:
            records.append(orjson.loads(line))
    return records


def _build_gold_index(gold: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(q["id"]): q for q in gold}


def _build_predictions_dict(questions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Reshape quorum output list into ``{qid: {ideal_answer, exact_answer, …}}``."""
    out: dict[str, dict[str, Any]] = {}
    for q in questions:
        qid = str(q.get("id", ""))
        out[qid] = {
            "ideal_answer": q.get("ideal_answer", ""),
            "exact_answer": q.get("exact_answer"),
            "type": q.get("type", "summary"),
        }
    return out


# ---------------------------------------------------------------------------
# Answer quality: ROUGE
# ---------------------------------------------------------------------------


def _rouge_scores(
    predictions: list[dict[str, Any]],
    gold_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute ROUGE-1/2/L F1 for ideal answers, overall and per question type."""
    scorer = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    per_question: dict[str, dict[str, float]] = {}
    by_type: dict[str, list[dict[str, float]]] = {}

    for q in predictions:
        qid = str(q.get("id", ""))
        gold = gold_index.get(qid)
        if gold is None:
            continue

        gold_ideals = gold.get("ideal_answer") or []
        if isinstance(gold_ideals, str):
            gold_ideals = [gold_ideals]
        if not gold_ideals:
            continue

        predicted = q.get("ideal_answer") or ""
        if not predicted:
            continue

        best: dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for ref in gold_ideals:
            result = scorer.score(ref, predicted)
            for metric in best:
                best[metric] = max(best[metric], result[metric].fmeasure)

        per_question[qid] = best
        qtype = str(q.get("type", gold.get("type", "summary")))
        by_type.setdefault(qtype, []).append(best)

    def _mean_scores(items: list[dict[str, float]]) -> dict[str, float]:
        if not items:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        return {
            metric: sum(s[metric] for s in items) / len(items)
            for metric in ("rouge1", "rouge2", "rougeL")
        }

    all_scores = list(per_question.values())
    overall = _mean_scores(all_scores)
    per_type = {t: _mean_scores(items) for t, items in sorted(by_type.items())}

    return {
        "overall": overall,
        "per_type": per_type,
        "n_scored": len(per_question),
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# Answer quality: BERTScore
# ---------------------------------------------------------------------------


def _bertscore(
    predictions: list[dict[str, Any]],
    gold_index: dict[str, dict[str, Any]],
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> dict[str, Any]:
    """Compute BERTScore F1 for ideal answers."""
    try:
        import evaluate

        bertscore_metric = evaluate.load("bertscore")
    except Exception as exc:  # noqa: BLE001
        return {"error": f"BERTScore unavailable: {exc}"}

    preds: list[str] = []
    refs: list[str] = []
    qids: list[str] = []

    for q in predictions:
        qid = str(q.get("id", ""))
        gold = gold_index.get(qid)
        if gold is None:
            continue

        gold_ideals = gold.get("ideal_answer") or []
        if isinstance(gold_ideals, str):
            gold_ideals = [gold_ideals]
        if not gold_ideals:
            continue

        predicted = q.get("ideal_answer") or ""
        if not predicted:
            continue

        preds.append(predicted)
        refs.append(gold_ideals[0])
        qids.append(qid)

    if not preds:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_scored": 0}

    results = bertscore_metric.compute(
        predictions=preds,
        references=refs,
        model_type=model_type,
    )

    per_question: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(qids):
        per_question[qid] = {
            "precision": results["precision"][i],
            "recall": results["recall"][i],
            "f1": results["f1"][i],
        }

    n = len(preds)
    return {
        "precision": sum(results["precision"]) / n,
        "recall": sum(results["recall"]) / n,
        "f1": sum(results["f1"]) / n,
        "n_scored": n,
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# Answer quality: exact-answer metrics (reuse existing)
# ---------------------------------------------------------------------------


def _exact_metrics(
    predictions_dict: dict[str, dict[str, Any]],
    gold_list: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "yesno": macro_f1_yesno(predictions_dict, gold_list),
        "factoid": mrr_factoid(predictions_dict, gold_list),
        "list": mean_f1_list(predictions_dict, gold_list),
    }


# ---------------------------------------------------------------------------
# Debate process metrics
# ---------------------------------------------------------------------------


def _process_metrics(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute debate process metrics from quorum_meta fields."""
    rounds_list: list[int] = []
    consensus_count = 0
    context_ratios: list[float] = []
    all_turns: list[dict[str, Any]] = []
    total_questions = 0

    for q in questions:
        meta = q.get("quorum_meta")
        if meta is None:
            continue
        total_questions += 1

        n_rounds = meta.get("rounds", 0)
        rounds_list.append(n_rounds)

        if meta.get("consensus_reached", False):
            consensus_count += 1

        docs_injected = meta.get("docs_injected", 0)
        docs_available = len(q.get("documents", []))
        if docs_available > 0:
            context_ratios.append(docs_injected / docs_available)

        for turn in meta.get("debate", []):
            all_turns.append(turn)

    if total_questions == 0:
        return {"error": "no questions with quorum_meta found"}

    convergence = _convergence_metrics(rounds_list)
    consensus_rate = consensus_count / total_questions
    context_efficiency = _context_efficiency(context_ratios)
    trajectory = _agreement_trajectory(all_turns)
    per_focus = _per_focus_analysis(all_turns)
    per_model = _per_model_analysis(all_turns)

    return {
        "convergence_speed": convergence,
        "consensus_rate": consensus_rate,
        "consensus_count": consensus_count,
        "total_questions": total_questions,
        "context_efficiency": context_efficiency,
        "agreement_trajectory": trajectory,
        "per_focus": per_focus,
        "per_model": per_model,
    }


def _convergence_metrics(rounds_list: list[int]) -> dict[str, float]:
    if not rounds_list:
        return {"mean_rounds": 0.0, "median_rounds": 0.0, "min_rounds": 0, "max_rounds": 0}
    return {
        "mean_rounds": statistics.mean(rounds_list),
        "median_rounds": statistics.median(rounds_list),
        "min_rounds": min(rounds_list),
        "max_rounds": max(rounds_list),
    }


def _context_efficiency(ratios: list[float]) -> dict[str, float]:
    if not ratios:
        return {"mean_ratio": 0.0, "median_ratio": 0.0}
    return {
        "mean_ratio": statistics.mean(ratios),
        "median_ratio": statistics.median(ratios),
    }


def _agreement_trajectory(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-round mean agreement level across all questions."""
    by_round: dict[int, list[int]] = {}
    for turn in turns:
        rnd = turn.get("round", 0)
        agreement_str = turn.get("agreement", "disagree")
        rank = AGREEMENT_RANK.get(agreement_str, 1)  # type: ignore[arg-type]
        by_round.setdefault(rnd, []).append(rank)

    trajectory: list[dict[str, float]] = []
    for rnd in sorted(by_round):
        values = by_round[rnd]
        trajectory.append({
            "round": rnd,
            "mean_agreement": statistics.mean(values),
            "n_turns": len(values),
        })

    return {"per_round": trajectory}


def _per_focus_analysis(turns: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean agreement level per thinking focus across all debates."""
    by_focus: dict[str, list[int]] = {}
    for turn in turns:
        focus = turn.get("agent_focus", "unknown")
        rank = AGREEMENT_RANK.get(turn.get("agreement", "disagree"), 1)  # type: ignore[arg-type]
        by_focus.setdefault(focus, []).append(rank)

    return {
        focus: {
            "mean_agreement": statistics.mean(vals),
            "n_turns": len(vals),
        }
        for focus, vals in sorted(by_focus.items())
    }


def _per_model_analysis(turns: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean agreement level per model across all debates."""
    by_model: dict[str, list[int]] = {}
    for turn in turns:
        model = turn.get("model", "unknown")
        rank = AGREEMENT_RANK.get(turn.get("agreement", "disagree"), 1)  # type: ignore[arg-type]
        by_model.setdefault(model, []).append(rank)

    return {
        model: {
            "mean_agreement": statistics.mean(vals),
            "n_turns": len(vals),
        }
        for model, vals in sorted(by_model.items())
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def _print_report(
    rouge: dict[str, Any],
    bertscore_result: dict[str, Any] | None,
    exact: dict[str, Any],
    process: dict[str, Any],
) -> None:
    W = 60
    sep = "-" * W

    print(f"\n{'=' * W}")
    print("  Quorum Evaluation Report")
    print(f"{'=' * W}")

    # ROUGE
    print(f"\n{sep}")
    print("  ROUGE (ideal answers)")
    print(sep)
    overall = rouge.get("overall", {})
    print(f"  ROUGE-1 F1  : {overall.get('rouge1', 0):.4f}")
    print(f"  ROUGE-2 F1  : {overall.get('rouge2', 0):.4f}")
    print(f"  ROUGE-L F1  : {overall.get('rougeL', 0):.4f}")
    print(f"  Questions   : {rouge.get('n_scored', 0)}")
    per_type = rouge.get("per_type", {})
    if per_type:
        print("  Per-type breakdown:")
        for qtype, scores in per_type.items():
            print(
                f"    {qtype:10s}  R1={scores['rouge1']:.4f}  "
                f"R2={scores['rouge2']:.4f}  RL={scores['rougeL']:.4f}"
            )

    # BERTScore
    if bertscore_result and "error" not in bertscore_result:
        print(f"\n{sep}")
        print("  BERTScore (ideal answers)")
        print(sep)
        print(f"  Precision   : {bertscore_result.get('precision', 0):.4f}")
        print(f"  Recall      : {bertscore_result.get('recall', 0):.4f}")
        print(f"  F1          : {bertscore_result.get('f1', 0):.4f}")
        print(f"  Questions   : {bertscore_result.get('n_scored', 0)}")

    # Exact-answer metrics
    yn = exact.get("yesno", {})
    if yn.get("n_scored", 0) > 0:
        print(f"\n{sep}")
        print("  Yes/No (exact)")
        print(sep)
        print(f"  Macro F1    : {yn['macro_f1']:.4f}")
        print(f"  Questions   : {yn['n_scored']}")

    fa = exact.get("factoid", {})
    if fa.get("n_scored", 0) > 0:
        print(f"\n{sep}")
        print("  Factoid (exact)")
        print(sep)
        print(f"  MRR         : {fa['mrr']:.4f}")
        print(f"  Strict acc  : {fa['strict_acc']:.4f}")
        print(f"  Lenient acc : {fa['lenient_acc']:.4f}")
        print(f"  Questions   : {fa['n_scored']}")

    li = exact.get("list", {})
    if li.get("n_scored", 0) > 0:
        print(f"\n{sep}")
        print("  List (exact)")
        print(sep)
        print(f"  Mean F1     : {li['mean_f1']:.4f}")
        print(f"  Questions   : {li['n_scored']}")

    # Process metrics
    print(f"\n{sep}")
    print("  Debate Process")
    print(sep)
    conv = process.get("convergence_speed", {})
    print(f"  Mean rounds       : {conv.get('mean_rounds', 0):.2f}")
    print(f"  Median rounds     : {conv.get('median_rounds', 0):.1f}")
    print(f"  Min/max rounds    : {conv.get('min_rounds', 0)} / {conv.get('max_rounds', 0)}")
    print(
        f"  Consensus rate    : {process.get('consensus_rate', 0):.1%} "
        f"({process.get('consensus_count', 0)}/{process.get('total_questions', 0)})"
    )
    ctx = process.get("context_efficiency", {})
    print(f"  Context efficiency: {ctx.get('mean_ratio', 0):.1%} of available docs injected")

    trajectory = process.get("agreement_trajectory", {}).get("per_round", [])
    if trajectory:
        labels = {0: "SD", 1: "D ", 2: "A ", 3: "SA"}
        vals = [f"R{t['round']}={t['mean_agreement']:.2f}" for t in trajectory[:10]]
        print(f"  Agreement curve   : {', '.join(vals)}")
        print(f"    (scale: 0=strongly_disagree … 3=strongly_agree)")

    per_focus = process.get("per_focus", {})
    if per_focus:
        print("  Per-focus agreement:")
        for focus, data in per_focus.items():
            print(f"    {focus:16s}  mean={data['mean_agreement']:.2f}  n={data['n_turns']}")

    per_model = process.get("per_model", {})
    if per_model:
        print("  Per-model agreement:")
        for model, data in per_model.items():
            print(f"    {model:40s}  mean={data['mean_agreement']:.2f}  n={data['n_turns']}")

    print(f"\n{'=' * W}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

evaluate_app = typer.Typer(
    name="evaluate",
    help="Evaluate quorum debate results against gold standard.",
    no_args_is_help=True,
)


@evaluate_app.command(name="run")
def evaluate_command(
    predictions: Annotated[
        Path,
        typer.Option("--predictions", "-p", help="Quorum output JSON file."),
    ],
    golden: Annotated[
        Path,
        typer.Option("--golden", "-g", help="Gold-standard JSONL or JSON file."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Save report as JSON."),
    ] = None,
    bertscore: Annotated[
        bool,
        typer.Option(help="Compute BERTScore (requires GPU and model download)."),
    ] = False,
    bertscore_model: Annotated[
        str,
        typer.Option("--bertscore-model", help="BERTScore model to use."),
    ] = "microsoft/deberta-xlarge-mnli",
) -> None:
    """Evaluate quorum results: answer quality + debate process metrics."""
    typer.echo(f"Loading predictions: {predictions}")
    quorum_questions = _load_quorum_output(predictions)

    typer.echo(f"Loading gold standard: {golden}")
    gold_questions = _load_gold(golden)
    gold_index = _build_gold_index(gold_questions)

    typer.echo(f"Predictions: {len(quorum_questions)}  |  Gold: {len(gold_questions)}")

    # Answer quality
    typer.echo("Computing ROUGE scores…")
    rouge = _rouge_scores(quorum_questions, gold_index)

    bertscore_result: dict[str, Any] | None = None
    if bertscore:
        typer.echo("Computing BERTScore…")
        bertscore_result = _bertscore(quorum_questions, gold_index, model_type=bertscore_model)

    predictions_dict = _build_predictions_dict(quorum_questions)
    typer.echo("Computing exact-answer metrics…")
    exact = _exact_metrics(predictions_dict, gold_questions)

    # Process metrics
    typer.echo("Computing debate process metrics…")
    process = _process_metrics(quorum_questions)

    # Print report
    _print_report(rouge, bertscore_result, exact, process)

    # Save JSON report
    if output is not None:
        report: dict[str, Any] = {
            "answer_quality": {
                "rouge": {k: v for k, v in rouge.items() if k != "per_question"},
                "exact": exact,
            },
            "debate_process": process,
            "per_question": {
                "rouge": rouge.get("per_question", {}),
            },
        }
        if bertscore_result and "error" not in bertscore_result:
            bs_summary = {k: v for k, v in bertscore_result.items() if k != "per_question"}
            report["answer_quality"]["bertscore"] = bs_summary
            report["per_question"]["bertscore"] = bertscore_result.get("per_question", {})
        elif bertscore_result:
            report["answer_quality"]["bertscore"] = bertscore_result

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float))
        typer.echo(f"Report saved to {output}")
