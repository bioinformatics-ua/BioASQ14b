"""Run BM25 inference using the database."""

from pathlib import Path
from typing import Annotated

import orjson
import typer

from bioasq.common import PROJECT_DATA_BM25_DIR
from bioasq.common.utils import typer_async
from bioasq.data.database import bm25_search

app = typer.Typer()


@app.command()
@typer_async
async def main(
    testset_file: Annotated[Path, typer.Argument(..., help="The testset file to use.")],
    output_file: Annotated[
        Path,
        typer.Option(
            PROJECT_DATA_BM25_DIR / "bm25_run.jsonl",
            "-o",
            "--output-file",
            help="The file to save the results to.",
        ),
    ],
) -> None:
    questions: list[dict[str, str]] = [orjson.loads(line) for line in testset_file.open("rb")]
    results = [await bm25_search(question["body"], topk=100) for question in questions]

    with output_file.open("wb") as f:
        for question, result in zip(questions, results, strict=True):
            f.write(
                orjson.dumps({"qid": question["id"], "results": [doc.to_dict() for doc in result]})
                + "\n"
            )
