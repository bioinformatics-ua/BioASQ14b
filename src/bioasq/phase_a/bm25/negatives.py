"""Negative document mining via BM25 retrieval.

For each training question, retrieves top-K BM25 results from the
question's baseline index, excludes positive documents, and outputs
negatives with text and scores.

Refactored from ``phaseA-BM25/negatives.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import msgspec
import typer
from tqdm import tqdm

from bioasq.common.utils import typer_async
from bioasq.data.database import bm25_search

if TYPE_CHECKING:
    from pathlib import Path

    from bioasq.common.aliases import DocumentId, QuestionId
    from bioasq.common.types import DocumentWithScore


class TrainingQuestion(msgspec.Struct):
    """A training question with positive docs and a baseline reference."""

    id: str
    body: str
    documents: list[PositiveDoc] = []
    baseline: str = ""


class PositiveDoc(msgspec.Struct, frozen=True):
    """A positive document reference."""

    id: DocumentId
    text: str = ""


class NegativeMiningOutput(msgspec.Struct):
    """Output record: a question with its positive and negative documents."""

    id: QuestionId
    body: str
    pos_docs: list[PositiveDoc] = []
    neg_docs: list[DocumentWithScore] = []


app = typer.Typer()


@app.command()
@typer_async
async def mine_negatives(
    training_file: Annotated[Path, typer.Argument(..., help="Training JSONL file.", exists=True)],
    output_file: Annotated[
        Path, typer.Option("../data/negatives.jsonl", "-o", help="Output JSONL.")
    ],
    num_results: Annotated[int, typer.Option(100, "-n", help="Negatives per question.")],
) -> None:
    """Mine BM25 negatives for training questions."""

    print(f"Loading training data from '{training_file}'...")
    decoder = msgspec.json.Decoder(TrainingQuestion)
    training_data = [decoder.decode(line) for line in training_file.open("rb")]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb+") as f:
        for question in tqdm(training_data, desc="Processing questions", unit="question"):
            pos_docs_ids: set[DocumentId] = {doc.id for doc in question.documents}
            neg_docs = (
                await bm25_search(question.body, topk=num_results + 100, exclude_ids=pos_docs_ids)
            )[:num_results]

            output = NegativeMiningOutput(
                id=question.id,
                body=question.body,
                pos_docs=[PositiveDoc(id=d.id, text=d.text) for d in question.documents],
                neg_docs=neg_docs,
            )
            f.write(msgspec.json.encode(output) + b"\n")

    print(f"Saved results to '{output_file}'.")


if __name__ == "__main__":
    app()
