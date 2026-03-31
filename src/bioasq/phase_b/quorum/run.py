"""CLI entry point for the agent quorum answer generation pipeline.

Usage examples::

    # 4 agents, 2 models, read from JSONL, write results to out.json
    bioasq phase-b quorum run \\
        --data data/batch1.jsonl \\
        --models "openrouter:google/gemini-2.5-flash,local:google/medgemma-27b-text-it" \\
        --num-agents 4 \\
        --max-rounds 6 \\
        --out results/quorum_out.json

    # Single model, verbose, limit to 3 documents per question
    bioasq phase-b quorum run \\
        --data data/batch1.jsonl \\
        --models "openrouter:google/gemini-2.5-flash" \\
        --num-agents 3 \\
        --max-docs 3 \\
        --out results/quorum_out.json
"""

import json
import random
from pathlib import Path
from typing import Annotated, Any

import orjson
import typer

from bioasq.phase_b.quorum.evaluate import evaluate_app

app = typer.Typer(
    name="quorum",
    help="Agent quorum: multi-agent debate for BioASQ answer generation.",
    no_args_is_help=True,
)
app.add_typer(evaluate_app, name="evaluate")


def _load_questions(data_path: Path) -> list[dict[str, Any]]:
    """Load questions from JSONL or JSON (``{"questions": [...]}`` wrapper)."""
    raw = data_path.read_bytes()
    # Try wrapped JSON first.
    try:
        obj = orjson.loads(raw)
        if isinstance(obj, dict) and "questions" in obj:
            return obj["questions"]
        if isinstance(obj, list):
            return obj
    except orjson.JSONDecodeError:
        pass

    # Fall back to JSONL.
    questions: list[dict[str, Any]] = []
    for line in data_path.open("rb"):
        line = line.strip()
        if line:
            questions.append(orjson.loads(line))
    return questions


def _build_backends(
    models: list[str],
    max_tokens: int,
    temperature: float,
    request_delay: float,
) -> list[Any]:
    """Instantiate and load one backend per model.

    Model strings use ``"backend:model_name"`` format, e.g.
    ``"openrouter:google/gemini-2.5-flash"`` or ``"local:google/medgemma-27b-text-it"``.
    If no prefix is given, defaults to ``"openrouter"``.
    """
    from bioasq.phase_b.backends.cloud import OpenRouterBackend

    backends = []
    for spec in models:
        if ":" in spec:
            backend_type, model_name = spec.split(":", 1)
        else:
            backend_type, model_name = "openrouter", spec

        if backend_type == "local":
            from bioasq.phase_b.backends.local import VLLMBackend

            backend: Any = VLLMBackend(
                model_path=model_name,
                max_new_tokens=max_tokens,
                temperature=temperature,
                tensor_parallel_size=2,
            )
        else:
            backend = OpenRouterBackend(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                request_delay=request_delay,
            )
        backend.load()
        backends.append(backend)
    return backends


def _extract_documents(question: dict[str, Any]) -> list[str]:
    """Extract ordered document texts from a question dict."""
    raw_docs = question.get("documents", [])
    texts: list[str] = []
    for doc in raw_docs:
        if isinstance(doc, dict):
            text = doc.get("text", "")
            if text:
                texts.append(str(text))
        elif isinstance(doc, str) and doc.strip():
            texts.append(doc)
    return texts


@app.command(name="run")
def run_quorum(
    data: Annotated[Path, typer.Option("--data", "-d", help="BioASQ JSONL or JSON input file.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output JSON file for results.")],
    models: Annotated[
        str,
        typer.Option(
            "--models",
            "-m",
            help="Comma-separated list of OpenRouter model names.",
        ),
    ] = "google/gemini-2.5-flash",
    num_agents: Annotated[
        int, typer.Option("--num-agents", "-n", help="Number of debate agents.")
    ] = 4,
    max_rounds: Annotated[
        int, typer.Option("--max-rounds", help="Maximum debate rounds per question.")
    ] = 8,
    max_docs: Annotated[
        int | None,
        typer.Option("--max-docs", help="Maximum documents to inject per question (default: all)."),
    ] = None,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", help="Max tokens per LLM response.")
    ] = 1024,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Sampling temperature.")
    ] = 0.5,
    request_delay: Annotated[
        float,
        typer.Option("--request-delay", help="Seconds between API requests (rate limiting)."),
    ] = 0.5,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Random seed for reproducibility.")
    ] = None,
    verbose: Annotated[bool, typer.Option(help="Print debate progress.")] = True,
    question_ids: Annotated[
        str | None,
        typer.Option("--ids", help="Comma-separated question IDs to process (default: all)."),
    ] = None,
) -> None:
    """Run the agent quorum debate to generate answers for BioASQ questions."""
    from bioasq.phase_b.quorum.agent import build_agents
    from bioasq.phase_b.quorum.debate import Debate

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        typer.echo("Error: at least one model must be specified.", err=True)
        raise typer.Exit(1)

    id_filter: set[str] | None = None
    if question_ids:
        id_filter = {qid.strip() for qid in question_ids.split(",") if qid.strip()}

    questions = _load_questions(data)
    if id_filter:
        questions = [q for q in questions if str(q.get("id", "")) in id_filter]

    if not questions:
        typer.echo("No questions to process.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(questions)} question(s).")
    typer.echo(f"Models: {model_list}")
    typer.echo(
        f"Agents: {num_agents}  |  Max rounds: {max_rounds}  |  Max docs: {max_docs or 'all'}"
    )

    backends = _build_backends(model_list, max_tokens, temperature, request_delay)
    rng = random.Random(seed)

    results: list[dict[str, Any]] = []

    for idx, question in enumerate(questions, start=1):
        q_id = str(question.get("id", f"q{idx}"))
        q_body = str(question.get("body", ""))
        q_type = str(question.get("type", "summary"))
        documents = _extract_documents(question)

        if not documents:
            typer.echo(f"[{idx}/{len(questions)}] {q_id}: no documents — skipping.")
            results.append(
                {
                    "id": q_id,
                    "type": q_type,
                    "body": q_body,
                    "error": "no_documents",
                    "ideal_answer": "",
                    "exact_answer": None,
                }
            )
            continue

        typer.echo(
            f"\n[{idx}/{len(questions)}] {q_id} ({q_type})  — {len(documents)} docs available"
        )

        # Fresh agents per question (resets participation flags).
        agents = build_agents(num_agents, model_list, backends)

        debate = Debate(
            question_id=q_id,
            question_body=q_body,
            question_type=q_type,
            documents=documents,
            agents=agents,
            max_rounds=max_rounds,
            max_docs=max_docs or len(documents),
            verbose=verbose,
            rng=random.Random(rng.randint(0, 2**32 - 1)),
        )

        result = debate.run()

        if verbose:
            typer.echo(
                f"\n  → Consensus: {result['consensus_reached']}  "
                f"Rounds: {result['rounds']}  "
                f"Docs used: {result['docs_injected']}"
            )
            typer.echo(f"  → Ideal answer: {result['ideal_answer'][:120]}…")

        results.append(
            {
                "id": q_id,
                "type": q_type,
                "body": q_body,
                "ideal_answer": result["ideal_answer"],
                "exact_answer": result["exact_answer"],
                "quorum_meta": {
                    "rounds": result["rounds"],
                    "consensus_reached": result["consensus_reached"],
                    "docs_injected": result["docs_injected"],
                    "num_agents": num_agents,
                    "models": model_list,
                    "debate": result["debate"],
                },
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"questions": results}, indent=2, ensure_ascii=False))
    typer.echo(f"\nResults written to {out}")

    for backend in backends:
        backend.unload()
