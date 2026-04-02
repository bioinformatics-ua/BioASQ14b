"""
Hydrate BioASQ predictions with baseline document text (title + abstract).

Uses database for fast batch lookups instead of file seeking (process_golden.py style).
Same logic as data/process_golden.py but significantly faster via indexed DB queries.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import orjson
import typer
from tqdm import tqdm

from bioasq.common.utils import typer_async
from bioasq.data.database import get_article_by_id

if TYPE_CHECKING:
    from bioasq.common.aliases import DocumentId

app = typer.Typer()


def _pmid_from_url(url: str) -> str:
    """Extract PMID from BioASQ document URL."""
    return url.rstrip("/").split("/")[-1]


@app.command()
@typer_async
async def main(
    file: Annotated[
        Path,
        typer.Argument(..., help="The BioASQ JSON file to hydrate (questions with documents)."),
    ],
    out_file: Annotated[
        Path | None,
        typer.Option(
            ...,
            "-o",
            "--out",
            help="Output jsonl path. Default: input name with '.hydrated.jsonl'.",
        ),
    ] = None,
) -> None:
    """Inject baseline document text into BioASQ predictions using database for fast lookups."""
    if not file.exists():
        raise FileNotFoundError(f"Predictions file '{file}' not found.")

    if out_file is None:
        out_file = file.with_suffix(".hydrated.jsonl")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("Load BioASQ file...")
    with file.open("rb") as f:
        data = orjson.loads(f.read())

    questions: list[dict[str, Any]] = data.get("questions", [])
    if not questions:
        raise ValueError(f"No questions found in '{file}'.")

    # Collect all PMIDs
    all_pmids: set[DocumentId] = set()
    for q in questions:
        for doc in q.get("documents", []):
            pmid = _pmid_from_url(doc) if isinstance(doc, str) else doc.get("id", "")
            if pmid:
                all_pmids.add(str(pmid))

    print(f"Batch fetch {len(all_pmids)} documents from database...")

    batch_texts = {pmid: await get_article_by_id(int(pmid)) for pmid in all_pmids}
    not_found: list[tuple[str, str]] = []

    print("Hydrating questions...")
    with out_file.open("wb") as out:
        for question in tqdm(questions, desc="Hydrating", unit="question"):
            for key in ("concepts", "triples"):
                if key in question:
                    del question[key]

            document_ids: list[dict[str, str]] = [
                {"id": _pmid_from_url(url) if isinstance(url, str) else url.get("id", "")}
                for url in question.get("documents", [])
            ]

            for idx, doc_ent in enumerate(document_ids):
                pmid = doc_ent["id"]
                if not pmid:
                    continue
                text = batch_texts[pmid].full_text
                if text is None:
                    not_found.append((question.get("id", "?"), pmid))
                    continue
                document_ids[idx]["text"] = text

            question["documents"] = document_ids
            out.write(orjson.dumps(question) + b"\n")

    print(f"Wrote '{out_file}'.")
    if not_found:
        print(f"Note: {len(not_found)} documents not found in baseline.")
        nf_path = out_file.parent / "not_found.jsonl"
        with nf_path.open("wb") as f:
            for qid, pmid in not_found:
                f.write(orjson.dumps({"question_id": qid, "pmid": pmid}) + b"\n")
        print(f"Logged to '{nf_path}'.")


if __name__ == "__main__":
    app()
