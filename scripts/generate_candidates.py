"""Generate hybrid retrieval candidates (BM25 + Dense → RRF) for a BioASQ testset.

Requires Docker Compose services running: PostgreSQL (db), Qdrant, TEI.

Outputs a BioASQ JSON where each question has a ``documents`` list of
``{id, text}`` dicts — ready for the reranker inference step.

Usage::

    uv run python scripts/generate_candidates.py \
        --testset data/BioASQ-task14bPhaseA-testset1.json \
        --output data/candidates_batch01.json

    uv run python scripts/generate_candidates.py \
        --testset data/BioASQ-task14bPhaseA-testset1.json \
        --output data/candidates_batch01.json \
        --bm25-topk 200 --semantic-topk 200 --rrf-k 60
"""

import asyncio
import copy
from pathlib import Path
from typing import Annotated

import orjson
import typer

from bioasq.data.database import close_pool, close_qdrant_client
from bioasq.phase_a.retrieval.pipeline import hybrid_retrieve

app = typer.Typer()


@app.command()
def main(
    testset: Annotated[Path, typer.Option(..., "--testset", "-t", help="BioASQ testset JSON")],
    output: Annotated[
        Path, typer.Option(..., "--output", "-o", help="Output BioASQ JSON with candidates")
    ],
    bm25_topk: Annotated[int, typer.Option("--bm25-topk", help="BM25 top-k")] = 200,
    semantic_topk: Annotated[
        int, typer.Option("--semantic-topk", help="Dense retrieval top-k")
    ] = 200,
    rrf_k: Annotated[int, typer.Option("--rrf-k", help="RRF k parameter")] = 60,
) -> None:
    """Retrieve candidate documents for each question with hybrid BM25+Dense search."""
    asyncio.run(_run(testset, output, bm25_topk, semantic_topk, rrf_k))


async def _run(
    testset: Path,
    output: Path,
    bm25_topk: int,
    semantic_topk: int,
    rrf_k: int,
) -> None:
    rrf_data: dict[str, list] = orjson.loads(testset.read_bytes())
    wsum_data = copy.deepcopy(rrf_data)
    bm25_data = copy.deepcopy(rrf_data)

    rrf_questions: list[dict] = rrf_data["questions"]
    wsum_questions: list[dict] = wsum_data["questions"]
    bm25_questions: list[dict] = bm25_data["questions"]
    typer.echo(f"Retrieving candidates for {len(rrf_questions)} questions...")

    try:
        for rrfq, wsumq, bm25q in zip(rrf_questions, wsum_questions, bm25_questions, strict=True):
            qid = rrfq["id"]
            body = rrfq["body"]

            rrf_docs, wsum_docs, bm25_docs = await hybrid_retrieve(
                qid,
                body,
                bm25_topk=bm25_topk,
                semantic_topk=semantic_topk,
                rrf_k=rrf_k,
            )

            # Store as {id, text} dicts for reranker consumption
            rrfq["documents"] = [{"id": d.pmid, "text": d.full_text} for d in rrf_docs]
            wsumq["documents"] = [{"id": d.pmid, "text": d.full_text} for d in wsum_docs]
            bm25q["documents"] = [{"id": d.pmid, "text": d.full_text} for d in bm25_docs]
            typer.echo(f"  [{qid}] {len(rrf_docs)} candidates")
    finally:
        await close_pool()
        await close_qdrant_client()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".rrf.json").write_bytes(orjson.dumps(rrf_data))
    output.with_suffix(".wsum.json").write_bytes(orjson.dumps(wsum_data))
    output.with_suffix(".bm25.json").write_bytes(orjson.dumps(bm25_data))

    typer.echo(f"Wrote {len(rrf_questions)} questions with candidates to {output}")


if __name__ == "__main__":
    app()
