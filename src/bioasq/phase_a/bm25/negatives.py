"""Negative document mining via hybrid retrieval (BM25 + dense, year-scoped).

For each training question, retrieves top-K candidates from the question's
baseline year using hybrid RRF (BM25 + Qdrant dense), excludes positive
documents, and outputs negatives with text and scores.
"""

import re
from pathlib import Path
from typing import Annotated

import msgspec
import typer
from tqdm import tqdm

from bioasq.common import PROJECT_DATA_DIR
from bioasq.common.aliases import DocumentId, QuestionId
from bioasq.common.types import DocumentWithScore
from bioasq.common.utils import typer_async
from bioasq.phase_a.retrieval.pipeline import hybrid_retrieve_rrf
from bioasq.phase_a.retrieval.query_encoder import default_tei_embed_url


class PositiveDoc(msgspec.Struct, frozen=True):
    """A positive document reference."""

    id: DocumentId
    text: str = ""


class TrainingQuestion(msgspec.Struct):
    """A training question with positive docs and a baseline reference."""

    id: str
    body: str
    documents: list[PositiveDoc] = []
    baseline: str = ""


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
    training_file: Annotated[Path, typer.Argument(..., help="Training JSONL file.")],
    output_file: Annotated[
        Path,
        typer.Option(..., "-o", "--output", help="Output JSONL."),
    ] = PROJECT_DATA_DIR / "negatives.jsonl",
    num_results: Annotated[
        int, typer.Option(..., "-n", "--num-results", help="Negatives per question.")
    ] = 100,
    embed_url: Annotated[
        str,
        typer.Option(..., "-e", "--embed-url", help="TEI embed endpoint URL."),
    ] = default_tei_embed_url(),
) -> None:
    """Mine hybrid (BM25 + dense) negatives for training questions, scoped to baseline year."""

    print(f"Loading training data from '{training_file}'...")
    decoder = msgspec.json.Decoder(TrainingQuestion)
    training_data = [decoder.decode(line) for line in training_file.open("rb")]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb+") as f:
        for question in tqdm(training_data, desc="Processing questions", unit="question"):
            year_match = re.search(r"\b(20\d{2})\b", question.baseline)
            year: int | None = int(year_match.group(1)) if year_match else None

            pos_docs_ids: set[DocumentId] = {doc.id for doc in question.documents}
            candidates = await hybrid_retrieve_rrf(
                question.id,
                question.body,
                year=year,
                bm25_topk=num_results + 100,
                semantic_topk=num_results + 100,
                embed_url=embed_url,
                exclude_ids=pos_docs_ids,
                rrf_k=100,
            )
            neg_docs = candidates[:num_results]

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
