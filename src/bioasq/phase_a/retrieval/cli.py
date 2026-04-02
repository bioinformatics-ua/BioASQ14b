"""CLI: BM25 + dense retrieval with RRF over a question JSONL testset."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Annotated, Literal

import msgspec
import orjson
import typer
from ranx import Run, fuse
from tqdm.asyncio import tqdm

from bioasq.common import PROJECT_DATA_BM25_DIR
from bioasq.common.utils import typer_async
from bioasq.phase_a.retrieval.pipeline import hybrid_retrieve
from bioasq.phase_a.retrieval.query_encoder import default_tei_embed_url

app = typer.Typer()


@app.command("retrieve")
@typer_async
async def retrieve(
    testset_file: Annotated[
        Path,
        typer.Argument(..., help="JSONL with id + body per line.", exists=True),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            ...,
            "-o",
            "--output",
            help="Output JSONL: qid + results (pmid, full_text, score).",
        ),
    ] = PROJECT_DATA_BM25_DIR / "retrieval_run.jsonl",
    bm25_topk: Annotated[int, typer.Option(help="BM25 candidate depth.")] = 200,
    semantic_topk: Annotated[int, typer.Option(help="Dense candidate depth.")] = 200,
    rrf_k: Annotated[int, typer.Option(help="RRF smoothing constant k.")] = 60,
    tei_embed_url: Annotated[
        str | None,
        typer.Option(help=f"Full TEI URL (default: {default_tei_embed_url()})."),
    ] = None,
) -> None:
    """Run hybrid retrieval (BM25 ∥ dense) and fuse with ranx RRF; writes one line per question."""
    questions: list[dict[str, str]] = [orjson.loads(line) for line in testset_file.open("rb")]
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with (
        output_file.with_suffix(".rrf.jsonl").open("wb") as out,
        output_file.with_suffix(".wsum.jsonl").open("wb") as wsum_out,
        output_file.with_suffix(".bm25.jsonl").open("wb") as bm25_out,
    ):
        for question in tqdm(questions, desc="Hybrid retrieve", unit="q"):
            qid = str(question["id"])
            body = str(question["body"])
            rrf, wsum, bm25 = await hybrid_retrieve(
                qid,
                body,
                bm25_topk=bm25_topk,
                semantic_topk=semantic_topk,
                embed_url=tei_embed_url,
                rrf_k=rrf_k,
            )
            for out_docs, out_f in [(rrf, out), (wsum, wsum_out), (bm25, bm25_out)]:
                row = {
                    "qid": qid,
                    "results": [msgspec.to_builtins(d) for d in out_docs],
                }
                out_f.write(orjson.dumps(row) + b"\n")

    # await close_pool()
    # await close_qdrant_client()


@app.command("fuse")
def fuse_runs(
    runs_dir: Annotated[Path, typer.Argument(..., help="Directory containing run files to fuse.")],
    output: Annotated[
        Path | None,
        typer.Option(
            ...,
            "-o",
            "--output",
            help='Output file path. If default, writes to "fuse_<method>.json".',
        ),
    ] = None,
    method: Annotated[
        Literal[
            "bayesfuse",
            "bordafuse",
            "anz",
            "gmnz",
            "max",
            "med",
            "min",
            "mnz",
            "sum",
            "condorcet",
            "isr",
            "log_isr",
            "logn_isr",
            "mapfuse",
            "mixed",
            "posfuse",
            "probfuse",
            "rbc",
            "rrf",
            "segfuse",
            "slidefuse",
            "w_bordafuse",
            "w_condorcet",
            "wmnz",
            "wsum",
        ],
        typer.Option(..., help="Fusion method."),
    ] = "rrf",
) -> None:
    ranx_runs: list[Run] = [
        Run.from_file(str(run)) for run in runs_dir.glob("[!fuse_]*.json") if run.is_file()
    ]
    result: Run = fuse(ranx_runs, method=method)

    if output is None:
        output = runs_dir / f"fuse_{method}.json"

    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output))


if __name__ == "__main__":
    app()
