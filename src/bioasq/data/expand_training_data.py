"""
Expand positive training data via semantic similarity lookup.

Takes gold documents per question and adds semantically similar documents at
various similarity thresholds (0.95, 0.9, 0.85, 0.8). Excludes original positives
and baseline-retrieved documents (known negatives) from expansion.

Output format: Same as input, with additional keys:
  - expanded_docs_095, expanded_docs_09, expanded_docs_085, expanded_docs_08
  Each is a list of {id, text} dicts.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import orjson
import typer
from tqdm import tqdm

from bioasq.common.types import Document
from bioasq.common.utils import typer_async
from bioasq.data.database import get_article_by_id, get_pmids_per_baseline, lookup_by_pmid

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
    year_match = re.search(r"\b(20\d{2})\b", baseline)
    year: int | None = int(year_match.group(1)) if year_match else None
    exclude_pmids: set[str] = set(await get_pmids_per_baseline(year)) if year is not None else set()

    exclude = all_pos | exclude_pmids

    async def to_docs(pmids: set[str]) -> list[Document]:
        result = []
        for pmid in pmids:
            p = await get_article_by_id(int(pmid))
            if p is not None:
                result.append({"id": pmid, "text": p.full_text})
        return result

    expanded_docs = await semantic_search_from_ids(all_pos, thresholds)
    for threshold in tqdm(thresholds, desc="Expanding documents, thresholds:"):
        expanded_docs[threshold] = expanded_docs[threshold] - exclude
        qdata[f"expanded_docs_{str(threshold).replace('.', '')}"] = await to_docs(
            expanded_docs[threshold]
        )

    return qdata


@app.command()
@typer_async
async def main(
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
            ...,
            "--thresholds",
            "-t",
            help="Similarity thresholds for tiers (default: 0.95, 0.9, 0.85, 0.8).",
        ),
    ] = [0.9, 0.85],
    baseline_filter: Annotated[
        str | None,
        typer.Option(
            "--baseline",
            "-b",
            help="Only process questions with this baseline value (e.g. '2025').",
        ),
    ] = None,
) -> None:
    """Expand positive training data using semantic similarity lookup."""
    if thresholds is None:
        thresholds = [0.95, 0.9, 0.85, 0.8]
    if not training_path.exists():
        raise FileNotFoundError(f"Training file not found: {training_path}")

    th_list = tuple(thresholds)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    typer.echo("Expanding training data...", err=True)
    count = 0
    skipped = 0
    with training_path.open("rb") as fin, output_path.open("wb") as fout:
        for i, line in enumerate(tqdm(fin, desc="Expanding training data"), 1):
            if not line.strip():
                continue
            try:
                qdata = orjson.loads(line)
            except orjson.JSONDecodeError:
                typer.echo(f"WARNING: skipping malformed JSON at line {i}", err=True)
                skipped += 1
                continue
            if baseline_filter and str(qdata.get("baseline", "")) != baseline_filter:
                skipped += 1
                continue
            expanded = await expand_question(qdata, thresholds=th_list)
            fout.write(orjson.dumps(expanded) + b"\n")
            count += 1

    typer.echo(f"Wrote {count} expanded records to {output_path} (skipped {skipped})", err=True)


if __name__ == "__main__":
    app()
