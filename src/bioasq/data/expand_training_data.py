"""
Expand positive training data via semantic similarity lookup.

Takes gold documents per question and adds semantically similar documents at
various similarity thresholds (0.95, 0.9, 0.85, 0.8). Excludes original positives
and baseline-retrieved documents (known negatives) from expansion.

Output format: Same as input, with additional keys:
  - expanded_docs_095, expanded_docs_09, expanded_docs_085, expanded_docs_08
  Each is a list of {id, text} dicts.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Annotated, Any

import orjson
import typer
from tqdm import tqdm

from bioasq.data.database import get_article_by_id, get_pmids_per_baseline, lookup_by_pmid

if TYPE_CHECKING:
    from pathlib import Path

    from bioasq.common.types import Document

app = typer.Typer()

TrainingRecord = dict[str, Any]


async def semantic_search_from_ids(
    doc_ids: set[str],
    thresholds: tuple[float, ...],
) -> dict[float, set[str]]:
    """Return pmids of documents similar to any in doc_ids with score > threshold."""
    expanded_docs: defaultdict[float, set[str]] = defaultdict(set)
    for doc_id in doc_ids:
        for other_id, score in await lookup_by_pmid(doc_id):
            for threshold in thresholds:
                if score > threshold:
                    expanded_docs[threshold].add(other_id)
    return expanded_docs


async def expand_question(
    qdata: TrainingRecord,
    thresholds: tuple[float, ...] = (0.95, 0.9),
) -> TrainingRecord:
    """Expand positives for one question and attach expanded_docs_* keys."""
    all_pos: set[str] = {str(doc["id"]) for doc in qdata.get("documents", [])}
    baseline: str = str(qdata.get("baseline", ""))
    exclude_pmids: set[str] = set(await get_pmids_per_baseline(baseline))

    exclude = all_pos | exclude_pmids

    async def to_docs(pmids: set[str]) -> list[Document]:
        return [
            {"id": pmid, "text": p.full_text}
            for pmid in pmids
            if (p := await get_article_by_id(pmid)) is not None
        ]

    expanded_docs = await semantic_search_from_ids(all_pos, thresholds)
    for threshold in thresholds:
        expanded_docs[threshold] = expanded_docs[threshold] - exclude
        qdata[f"expanded_docs_{str(threshold).replace('.', '')}"] = await to_docs(
            expanded_docs[threshold]
        )

    return qdata


@app.command()
def main(
    training_path: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Input training JSONL (e.g. training14b_inflated_clean_wContents.jsonl).",
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            ...,
            "--output",
            "-o",
            help="Output JSONL with expanded_docs_* attached.",
        ),
    ],
    thresholds: Annotated[
        list[float],
        typer.Option(
            [0.95, 0.9],
            "--thresholds",
            "-t",
            help="Similarity thresholds for tiers (default: 0.95, 0.9, 0.85, 0.8).",
        ),
    ],
) -> None:
    """Expand positive training data using semantic similarity lookup."""
    if not training_path.exists():
        raise FileNotFoundError(f"Training file not found: {training_path}")

    th_list = tuple(float(x.strip()) for x in thresholds.split(","))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    typer.echo("Expanding training data...", err=True)
    count = 0
    with training_path.open("rb") as fin, output_path.open("wb") as fout:
        for line in tqdm(fin, desc="Expanding training data"):
            if not line.strip():
                continue
            qdata = orjson.loads(line)
            expanded = expand_question(qdata, thresholds=th_list)
            fout.write(orjson.dumps(expanded) + b"\n")
            count += 1

    typer.echo(f"Wrote {count} expanded records to {output_path}", err=True)


if __name__ == "__main__":
    app()
