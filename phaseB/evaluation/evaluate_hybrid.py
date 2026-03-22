r"""
Hybrid evaluation: BioASQ Phase B classical metrics plus LLM-as-judge scores.

This combines the official-style metrics in ``evaluation.metrics`` with a
rubric-based judge implemented through ``phaseB/loaders`` (local vLLM or
OpenRouter), similar in spirit to automated RAG/QA evaluation toolkits that use
a strong model to score correctness and faithfulness.

Reference frameworks and papers (for methodology context, not runtime deps):

- RAGAS — automated evaluation for RAG: https://arxiv.org/abs/2309.15217
  Docs: https://docs.ragas.io/
- MT-Bench / Chatbot Arena — LLM-as-judge practice and limits: https://arxiv.org/abs/2306.05685
- G-Eval — LLM evaluation with chain-of-thought rubrics: https://arxiv.org/abs/2303.16634

Example (from ``phaseB/``, classical metrics only):

.. code-block:: bash

   uv run python evaluation/evaluate_hybrid.py classical \\
     --predictions outputs/Mistral_abstracts_5_1.json \\
     --golden ../data/val_data/13B1_golden_documents.jsonl \\
     --output results/classical.json

Example (LLM judge via OpenRouter; requires ``OPENROUTER_API_KEY``):

.. code-block:: bash

   uv run python evaluation/evaluate_hybrid.py judge \\
     --predictions outputs/Mistral_abstracts_5_1.json \\
     --golden ../data/val_data/13B1_golden_documents.jsonl \\
     --backend openrouter \\
     --model anthropic/claude-sonnet-4-6 \\
     --output results/judge.json

Example (both):

.. code-block:: bash

      uv run python evaluation/evaluate_hybrid.py all \\
        --predictions outputs/Mistral_abstracts_5_1.json \\
        --golden ../data/val_data/13B1_golden_documents.jsonl \\
        --backend openrouter \\
        --model anthropic/claude-sonnet-4-6 \\
        --output results/hybrid.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate import print_report
from evaluation.llm_judge import build_backend, judge_score_means, run_judge_batch
from evaluation.metrics import evaluate_all
from evaluation.prediction_normalize import metrics_ready_predictions
from loaders.dataloader import BioASQDataLoader

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _parse_types(spec: Optional[str]) -> Optional[set[str]]:
    if spec is None or not spec.strip():
        return None
    return {x.strip() for x in spec.split(",") if x.strip()}


def _filter_golden(
    loader: BioASQDataLoader,
    types: Optional[set[str]],
    limit: Optional[int],
) -> list[dict[str, Any]]:
    rows = [q for q in loader if q.get("ideal_answer")]
    if types is not None:
        rows = [q for q in rows if q["type"] in types]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    with open(path) as f:
        raw = json.load(f)
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _align_predictions(
    predictions: dict[str, dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    ids = {str(q["id"]) for q in ground_truth}
    return {k: v for k, v in predictions.items() if k in ids}


@app.command("classical")
def classical_cmd(
    predictions: Annotated[
        Path, typer.Option("--predictions", exists=True, dir_okay=False)
    ],
    golden: Annotated[Path, typer.Option("--golden", exists=True, dir_okay=False)],
    output: Annotated[Optional[Path], typer.Option("--output", dir_okay=False)] = None,
    question_types: Annotated[
        Optional[str],
        typer.Option("--question-types", help="Comma list: yesno,factoid,list,summary"),
    ] = None,
    limit: Annotated[Optional[int], typer.Option("--limit", min=1)] = None,
) -> None:
    types = _parse_types(question_types)
    loader = BioASQDataLoader(str(golden))
    gt = _filter_golden(loader, types, limit)
    pred_raw = _load_predictions(predictions)
    pred = _align_predictions(pred_raw, gt)
    scored = metrics_ready_predictions(pred)
    n_total = len(scored)
    n_valid = sum(1 for v in scored.values() if v.get("valid", False))
    results = evaluate_all(scored, gt)
    print_report(results, n_total, n_valid)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": "classical",
            "predictions_file": str(predictions),
            "golden_file": str(golden),
            "n_total": n_total,
            "n_valid": n_valid,
            "metrics": results,
        }
        with open(output, "w") as f:
            json.dump(payload, f, indent=2)
        typer.echo(f"Wrote {output}")


@app.command("judge")
def judge_cmd(
    predictions: Annotated[
        Path, typer.Option("--predictions", exists=True, dir_okay=False)
    ],
    golden: Annotated[Path, typer.Option("--golden", exists=True, dir_okay=False)],
    backend: Annotated[str, typer.Option("--backend")] = "openrouter",
    model: Annotated[
        str, typer.Option("--model", "-m")
    ] = "anthropic/claude-sonnet-4-6",
    output: Annotated[Optional[Path], typer.Option("--output", dir_okay=False)] = None,
    question_types: Annotated[Optional[str], typer.Option("--question-types")] = None,
    limit: Annotated[Optional[int], typer.Option("--limit", min=1)] = None,
    max_tokens: Annotated[int, typer.Option("--max-tokens")] = 2048,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.0,
    context_max_chars: Annotated[int, typer.Option("--context-max-chars")] = 6000,
    tensor_parallel_size: Annotated[int, typer.Option("--tensor-parallel-size")] = 1,
    gpu_memory_utilization: Annotated[
        float, typer.Option("--gpu-memory-utilization")
    ] = 0.9,
    max_model_len: Annotated[int, typer.Option("--max-model-len")] = 8192,
) -> None:
    types = _parse_types(question_types)
    loader = BioASQDataLoader(str(golden))
    gt = _filter_golden(loader, types, limit)
    pred_raw = _load_predictions(predictions)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for q in gt:
        qid = str(q["id"])
        if qid in pred_raw:
            pairs.append((q, pred_raw[qid]))
    b = build_backend(
        backend,
        model,
        max_tokens,
        temperature,
        tensor_parallel_size,
        gpu_memory_utilization,
        max_model_len,
    )
    scores = run_judge_batch(b, pairs, context_max_chars)
    means = judge_score_means(scores)
    typer.echo(json.dumps({"judge_means": means}, indent=2))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        detail = {qid: asdict(s) for qid, s in scores.items()}
        payload = {
            "mode": "judge",
            "predictions_file": str(predictions),
            "golden_file": str(golden),
            "backend": backend,
            "judge_model": model,
            "judge_means": means,
            "per_question": detail,
        }
        with open(output, "w") as f:
            json.dump(payload, f, indent=2)
        typer.echo(f"Wrote {output}")


@app.command("all")
def all_cmd(
    predictions: Annotated[
        Path, typer.Option("--predictions", exists=True, dir_okay=False)
    ],
    golden: Annotated[Path, typer.Option("--golden", exists=True, dir_okay=False)],
    backend: Annotated[str, typer.Option("--backend")] = "openrouter",
    model: Annotated[
        str, typer.Option("--model", "-m")
    ] = "anthropic/claude-sonnet-4-6",
    output: Annotated[Optional[Path], typer.Option("--output", dir_okay=False)] = None,
    question_types: Annotated[Optional[str], typer.Option("--question-types")] = None,
    limit: Annotated[Optional[int], typer.Option("--limit", min=1)] = None,
    max_tokens: Annotated[int, typer.Option("--max-tokens")] = 2048,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.0,
    context_max_chars: Annotated[int, typer.Option("--context-max-chars")] = 6000,
    tensor_parallel_size: Annotated[int, typer.Option("--tensor-parallel-size")] = 1,
    gpu_memory_utilization: Annotated[
        float, typer.Option("--gpu-memory-utilization")
    ] = 0.9,
    max_model_len: Annotated[int, typer.Option("--max-model-len")] = 8192,
) -> None:
    types = _parse_types(question_types)
    print(types)
    loader = BioASQDataLoader(str(golden))
    print(loader)
    gt = _filter_golden(loader, types, limit)
    print("gt:", gt)
    pred_raw = _load_predictions(predictions)
    print("pred_raw:", pred_raw)
    pred = _align_predictions(pred_raw, gt)
    print("pred:", pred)
    scored = metrics_ready_predictions(pred)
    print("scored:", scored)
    n_total = len(scored)
    n_valid = sum(1 for v in scored.values() if v.get("valid", False))
    classical = evaluate_all(scored, gt)
    print_report(classical, n_total, n_valid)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (q, pred_raw[str(q["id"])]) for q in gt if str(q["id"]) in pred_raw
    ]
    b = build_backend(
        backend,
        model,
        max_tokens,
        temperature,
        tensor_parallel_size,
        gpu_memory_utilization,
        max_model_len,
    )
    judge_scores = run_judge_batch(b, pairs, context_max_chars)
    means = judge_score_means(judge_scores)
    typer.echo(json.dumps({"judge_means": means}, indent=2))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        detail = {qid: asdict(s) for qid, s in judge_scores.items()}
        payload = {
            "mode": "all",
            "predictions_file": str(predictions),
            "golden_file": str(golden),
            "backend": backend,
            "judge_model": model,
            "n_total": n_total,
            "n_valid": n_valid,
            "metrics": classical,
            "judge_means": means,
            "per_question_judge": detail,
        }
        with open(output, "w") as f:
            json.dump(payload, f, indent=2)
        typer.echo(f"Wrote {output}")


if __name__ == "__main__":
    app()
