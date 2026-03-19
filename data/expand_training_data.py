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
import time

from pathlib import Path
from typing import Any

import orjson
import typer
from tqdm import tqdm

app = typer.Typer()

# Lookup: pmid -> list of (pmid, score) pairs. JSON stores tuples as [str, float].
SimilarityLookup = dict[str, list[tuple[str, float]]]
Collection = dict[str, str]  # pmid -> full text
IDsPerBaseline = dict[str, dict[str, Any]]  # baseline_year -> pmid -> line_idx (or similar)
TrainingRecord = dict[str, Any]
ExpandedDoc = dict[str, str]  # {"id": pmid, "text": ...}


def _load_lookup(path: Path) -> SimilarityLookup:
    """Load similarity lookup from JSON. Format: {pmid: [[pmid, score], ...]}."""
    data = orjson.loads(path.read_bytes())
    # JSON arrays come as lists; convert inner [pmid, score] to tuple
    result: SimilarityLookup = {}
    for k, v in data.items():
        result[str(k)] = [(str(p), float(s)) for p, s in v]
    return result


def _load_ids_per_baseline(path: Path | None) -> dict[str, set[str]]:
    """Load ids_per_baseline; return baseline_year -> set of pmids."""
    if path is None:
        return {}
    data: IDsPerBaseline = orjson.loads(path.read_bytes())
    return {year: set(ids.keys()) for year, ids in data.items()}


def _load_collection(path: Path) -> Collection:
    """Load document collection from JSONL. Expects {pmid, title?, abstract?, text?}."""
    result: Collection = {}
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            doc = orjson.loads(line)
            pmid = str(doc.get("pmid", doc.get("id", "")))
            text = doc.get("text")
            if text is None:
                title = doc.get("title", "")
                abstract = doc.get("abstract", "")
                text = f"{title} {abstract}".strip()
            result[pmid] = str(text)
    return result


def semantic_search_from_ids(
    doc_ids: set[str],
    lookup: SimilarityLookup,
    thresholds: tuple[float, ...],
) -> dict[float, set[str]]:
    """Return pmids of documents similar to any in doc_ids with score > threshold."""
    expanded_docs: defaultdict[float, set[str]] = defaultdict(set)
    for doc_id in doc_ids:
        for other_id, score in lookup.get(doc_id, []):
            for threshold in thresholds:
                if score > threshold:
                    expanded_docs[threshold].add(other_id)
    return expanded_docs


def expand_question(
    qdata: TrainingRecord,
    lookup: SimilarityLookup,
    ids_per_baseline: dict[str, set[str]],
    collection: Collection,
    thresholds: tuple[float, ...] = (0.95, 0.9),
) -> TrainingRecord:
    """Expand positives for one question and attach expanded_docs_* keys."""
    all_pos: set[str] = {str(doc["id"]) for doc in qdata.get("documents", [])}
    baseline: str = str(qdata.get("baseline", ""))
    exclude_baseline: set[str] = ids_per_baseline.get(baseline, set())

    exclude = all_pos | exclude_baseline
    
    def to_docs(pmids: set[str]) -> list[ExpandedDoc]:
        return [{"id": pmid, "text": doc} for pmid in pmids if (doc := collection.get(pmid)) is not None]

    expanded_docs = semantic_search_from_ids(all_pos, lookup, thresholds)
    expanded_095 = expanded_docs[0.95] - exclude
    expanded_09 = expanded_docs[0.9] - exclude - expanded_095
    # expanded_085 = expanded_docs[0.85] - exclude - expanded_095 - expanded_09
    # expanded_08 = expanded_docs[0.8] - exclude - expanded_095 - expanded_09 - expanded_085
    out = dict(qdata)
    out["expanded_docs_095"] = to_docs(expanded_095)
    out["expanded_docs_09"] = to_docs(expanded_09)
    # out["expanded_docs_085"] = to_docs(expanded_085)
    # out["expanded_docs_08"] = to_docs(expanded_08)
    return out


@app.command()
def main(
    training_path: Path = typer.Argument(
        ...,
        help="Input training JSONL (e.g. training14b_inflated_clean_wContents.jsonl).",
    ),
    lookup_path: Path = typer.Option(
        ...,
        "--lookup",
        "-l",
        help="JSON file with similarity lookup: {pmid: [[pmid, score], ...]}.",
    ),
    collection_path: Path = typer.Option(
        ...,
        "--collection",
        "-c",
        help="JSONL with pmid/text for document text (e.g. pubmed_baseline jsonl).",
    ),
    ids_per_baseline_path: Path | None = typer.Option(
        None,
        "--ids-per-baseline",
        "-i",
        help="Optional JSON: baseline year -> {pmid: ...}. Used to exclude baseline docs from expansion.",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output JSONL with expanded_docs_* attached.",
    ),
    thresholds: str = typer.Option(
        "0.95,0.9",
        "--thresholds",
        "-t",
        help="Comma-separated similarity thresholds for tiers (default: 0.95,0.9,0.85,0.8).",
    ),
) -> None:
    """Expand positive training data using semantic similarity lookup."""
    if not training_path.exists():
        raise FileNotFoundError(f"Training file not found: {training_path}")
    if not lookup_path.exists():
        raise FileNotFoundError(f"Lookup file not found: {lookup_path}")
    if not collection_path.exists():
        raise FileNotFoundError(f"Collection file not found: {collection_path}")
    if ids_per_baseline_path is not None and not ids_per_baseline_path.exists():
        raise FileNotFoundError(f"ids_per_baseline file not found: {ids_per_baseline_path}")

    th_list = tuple(float(x.strip()) for x in thresholds.split(","))

    typer.echo("Loading lookup...", err=True)
    lookup = _load_lookup(lookup_path)

    typer.echo("Loading ids_per_baseline...", err=True)
    ids_per_baseline = _load_ids_per_baseline(ids_per_baseline_path)

    typer.echo("Loading collection...", err=True)
    collection = _load_collection(collection_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    typer.echo("Expanding training data...", err=True)
    count = 0
    with training_path.open("rb") as fin, output_path.open("wb") as fout:
        for line in tqdm(fin, desc="Expanding training data"):
            if not line.strip():
                continue
            qdata = orjson.loads(line)
            expanded = expand_question(
                qdata, lookup, ids_per_baseline, collection, thresholds=th_list
            )
            fout.write(orjson.dumps(expanded) + b"\n")
            count += 1

    typer.echo(f"Wrote {count} expanded records to {output_path}", err=True)


if __name__ == "__main__":
    app()
