"""CLI: BM25 + dense retrieval with RRF over a question JSONL testset."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Annotated

import msgspec
import orjson
import typer
from tqdm.asyncio import tqdm

from bioasq.common import PROJECT_DATA_BM25_DIR
from bioasq.common.utils import typer_async
from bioasq.phase_a.retrieval.pipeline import hybrid_retrieve_rrf
from bioasq.phase_a.retrieval.query_encoder import default_tei_embed_url

app = typer.Typer(no_args_is_help=True)


@app.command("run-jsonl")
@typer_async
async def hybrid_retrieve_jsonl(
    testset_file: Annotated[
        Path,
        typer.Argument(..., help="JSONL with id + body per line.", exists=True),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            PROJECT_DATA_BM25_DIR / "hybrid_rrf_run.jsonl",
            "-o",
            "--output",
            help="Output JSONL: qid + results (pmid, full_text, score).",
        ),
    ],
    bm25_topk: Annotated[int, typer.Option(100, help="BM25 candidate depth.")] = 100,
    semantic_topk: Annotated[int, typer.Option(100, help="Dense candidate depth.")] = 100,
    rrf_k: Annotated[int, typer.Option(60, help="RRF smoothing constant k.")] = 60,
    tei_embed_url: Annotated[
        str | None,
        typer.Option(None, help=f"Full TEI URL (default: {default_tei_embed_url()})."),
    ] = None,
) -> None:
    """Run hybrid retrieval (BM25 ∥ dense) and fuse with ranx RRF; writes one line per question."""
    questions: list[dict[str, str]] = [orjson.loads(line) for line in testset_file.open("rb")]
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("wb") as out:
        for question in tqdm(questions, desc="Hybrid retrieve", unit="q"):
            qid = str(question["id"])
            body = str(question["body"])
            fused = await hybrid_retrieve_rrf(
                qid,
                body,
                bm25_topk=bm25_topk,
                semantic_topk=semantic_topk,
                embed_url=tei_embed_url,
                rrf_k=rrf_k,
            )
            row = {
                "qid": qid,
                "results": [msgspec.to_builtins(d) for d in fused],
            }
            out.write(orjson.dumps(row) + b"\n")
