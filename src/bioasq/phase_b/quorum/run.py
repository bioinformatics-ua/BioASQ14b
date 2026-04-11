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

import random
from pathlib import Path
from typing import Annotated, Any

import orjson
import typer

from bioasq.phase_b.backends.base import BaseModelBackend
from bioasq.phase_b.quorum._types import QuorumDocument
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
    local_max_tokens: int | None = None,
) -> list[BaseModelBackend]:
    """Instantiate and load one backend per model.

    Model strings use ``"backend|model_name"`` format, e.g.
    ``"openrouter|google/gemini-2.5-flash"`` or ``"local|google/medgemma-27b-text-it"``, or
    ``"external|192.168.1.2:8080|google/medgemma-27b-text-it"`` for custom OpenAI-based endpoints.
    If no prefix is given, defaults to ``"openrouter"``.
    """
    from bioasq.phase_b.backends.cloud import OpenRouterBackend

    backends: list[BaseModelBackend] = []
    for spec in models:
        if "|" in spec:
            backend_type, model_name = spec.split("|", 1)
        else:
            backend_type, model_name = "openrouter", spec

        if backend_type == "local":
            from bioasq.phase_b.backends.local import VLLMBackend

            backend = VLLMBackend(
                model_path=model_name,
                max_new_tokens=local_max_tokens if local_max_tokens is not None else max_tokens,
                temperature=temperature,
                tensor_parallel_size=2,
            )
        else:
            base_url, model_name = (
                model_name.split("|", 1) if backend_type == "external" else (None, model_name)
            )
            backend = OpenRouterBackend(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                request_delay=request_delay,
                base_url=base_url,
            )
        backend.load()
        backends.append(backend)
    return backends


def _extract_documents(question: dict[str, Any]) -> list[QuorumDocument]:
    """Extract ordered documents and bind snippets by source document ID."""
    raw_docs = question.get("documents", [])
    snippets_by_doc_id: dict[str, list[str]] = {}

    for raw_snippet in question.get("snippets", []):
        doc_id = raw_snippet.get("document").rstrip("/").split("/")[-1]
        snippet = raw_snippet.get("text")
        if doc_id is None or snippet is None:
            continue
        snippets_by_doc_id.setdefault(doc_id, []).append(snippet)

    documents: list[QuorumDocument] = []
    for doc in raw_docs:
        if not isinstance(doc, dict):
            print(f"Warning: skipping invalid document (not a dict): {doc}")
            continue

        text = doc.get("text", "")
        if not isinstance(text, str) or not text.strip():
            print(f"Warning: skipping document with invalid or empty text: {doc}")
            continue

        source_id = doc.get("id").rstrip("/").split("/")[-1]
        documents.append(
            {
                "text": text.strip(),
                "snippets": list(snippets_by_doc_id.get(source_id, []))
                if source_id is not None
                else [],
            }
        )
    return documents


@app.command(name="run")
def run_quorum(
    data: Annotated[Path, typer.Option("--data", "-d", help="BioASQ JSONL or JSON input file.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output JSONL file for results.")],
    models: Annotated[
        list[str],
        typer.Option(
            "--models",
            "-m",
            help="Comma-separated list of OpenRouter model names.",
        ),
    ],
    num_agents: Annotated[
        int, typer.Option("--num-agents", "-n", help="Number of debate agents.")
    ] = 6,
    max_rounds: Annotated[
        int, typer.Option("--max-rounds", help="Maximum debate rounds per question.")
    ] = 8,
    docs_per_sample: Annotated[
        int,
        typer.Option(
            "--docs-per-sample",
            help="Number of documents each agent sees per round (initial n).",
        ),
    ] = 3,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", help="Max tokens per LLM response (cloud backends).")
    ] = 64000,
    local_max_tokens: Annotated[
        int,
        typer.Option(
            "--local-max-tokens",
            help="Max tokens for local vLLM backends (defaults to --max-tokens if omitted).",
        ),
    ] = 128000,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Sampling temperature.")
    ] = 0.5,
    request_delay: Annotated[
        float,
        typer.Option("--request-delay", help="Seconds between API requests (rate limiting)."),
    ] = 0.05,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Random seed for reproducibility.")
    ] = None,
    verbose: Annotated[bool, typer.Option(help="Print debate progress.")] = True,
    question_ids: Annotated[
        str | None,
        typer.Option("--ids", help="Comma-separated question IDs to process (default: all)."),
    ] = None,
    synthesizer_model: Annotated[
        str | None,
        typer.Option(
            "--synthesizer-model",
            help=(
                "Model for the final synthesis step (e.g. 'openrouter:google/gemini-2.5-pro'). "
                "If omitted, the first debate agent is used."
            ),
        ),
    ] = None,
    generation_timeout: Annotated[
        float,
        typer.Option(
            ...,
            help=(
                "Wall-clock seconds allowed per LLM call. "
                "Responses that exceed this (or are detected as repetitive) are discarded "
                "and treated as invalid JSON. Use 0 to disable."
            ),
        ),
    ] = 60.0,
    repetition_threshold: Annotated[
        float,
        typer.Option(
            ...,
            help=(
                "Fraction of identical 6-word n-grams that triggers the hallucination "
                "filter (0.0-1.0). Default 0.15 = 15%%. Use 1.0 to disable."
            ),
        ),
    ] = 0.15,
    start_index: Annotated[
        int, typer.Option("--start-index", help="Starting question index (for resuming).")
    ] = 0,
    end_index: Annotated[
        int | None,
        typer.Option("--end-index", help="Ending question index (exclusive, for resuming)."),
    ] = None,
) -> None:
    """Run the agent quorum debate to generate answers for BioASQ questions."""
    from bioasq.phase_b.quorum.agent import build_agents
    from bioasq.phase_b.quorum.debate import Debate

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
    typer.echo(f"Models: {models}")
    typer.echo(
        f"Agents: {num_agents}  |  Max rounds: {max_rounds}  |  Docs/sample: {docs_per_sample}"
    )

    backends = _build_backends(models, max_tokens, temperature, request_delay, local_max_tokens)
    rng = random.Random(seed)

    synthesizer_backend: BaseModelBackend | None = None
    if synthesizer_model:
        [synthesizer_backend] = _build_backends(
            [synthesizer_model], max_tokens, temperature, request_delay, local_max_tokens
        )

    results: list[dict[str, Any]] = []
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w+b") as f:
        for idx, question in enumerate(questions[start_index:end_index], start=start_index):
            q_id = str(question.get("id", f"q{idx}"))
            q_body = str(question.get("body", ""))
            q_type = str(question.get("type", "summary"))
            documents = _extract_documents(question)

            if not documents:
                typer.echo(f"[{idx}/{len(questions) - 1}] {q_id}: no documents — skipping.")
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
                f"\n[{idx}/{len(questions) - 1}] {q_id} ({q_type}) — {len(documents)} docs in total"
            )

            # Fresh agents per question (resets participation flags).
            agents = build_agents(
                num_agents,
                models,
                backends,
                generation_timeout=generation_timeout,
                repetition_threshold=repetition_threshold,
            )

            debate = Debate(
                question_id=q_id,
                question_body=q_body,
                question_type=q_type,
                documents=documents,
                agents=agents,
                docs_per_sample=docs_per_sample,
                max_rounds=max_rounds,
                verbose=verbose,
                rng=random.Random(rng.randint(0, 2**32 - 1)),
                synthesizer_backend=synthesizer_backend,
            )

            result = debate.run()

            if verbose:
                typer.echo(
                    f"\n  → Consensus: {result['consensus_reached']}  "
                    f"Rounds: {result['rounds']}  "
                    f"Total docs: {result['total_docs']}"
                )
                typer.echo(f"  → Ideal answer: {result['ideal_answer'][:120]}…")

            result = {
                "id": q_id,
                "type": q_type,
                "body": q_body,
                "ideal_answer": result["ideal_answer"],
                "exact_answer": result["exact_answer"],
                "quorum_meta": {
                    "rounds": result["rounds"],
                    "consensus_reached": result["consensus_reached"],
                    "total_docs": result["total_docs"],
                    "docs_per_sample": result["docs_per_sample"],
                    "num_agents": num_agents,
                    "models": models,
                    "debate": result["debate"],
                },
            }
            results.append(result)

            f.write(orjson.dumps(result, option=orjson.OPT_NON_STR_KEYS) + b"\n")

    out.with_suffix(".final.json").write_bytes(
        orjson.dumps(
            {"questions": results},
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_INDENT_2,
        )
    )

    typer.echo(f"\nResults written to {out}")

    for backend in backends:
        backend.unload()
    if synthesizer_backend is not None:
        synthesizer_backend.unload()


if __name__ == "__main__":
    app()
