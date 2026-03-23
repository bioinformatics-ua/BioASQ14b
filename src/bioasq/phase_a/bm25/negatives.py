"""Negative document mining via BM25 retrieval.

For each training question, retrieves top-K BM25 results from the
question's baseline index, excludes positive documents, and outputs
negatives with text and scores.

Refactored from ``phaseA-BM25/negatives.py``.
"""

from __future__ import annotations

from pathlib import Path

import msgspec
import orjson
import pandas as pd
from tqdm import tqdm

from bioasq.phase_a.bm25.index import load_index


# ---------------------------------------------------------------------------
# Typed structures for negative mining
# ---------------------------------------------------------------------------


class TrainingQuestion(msgspec.Struct):
    """A training question with positive docs and a baseline reference."""

    id: str
    body: str
    documents: list[PositiveDoc] = []
    baseline: str = ""


class PositiveDoc(msgspec.Struct, frozen=True):
    """A positive document reference."""

    id: str
    text: str = ""


class MinedNegativeDoc(msgspec.Struct):
    """A mined negative document with BM25 score."""

    id: str
    text: str = ""
    score: float = 0.0


class NegativeMiningOutput(msgspec.Struct):
    """Output record: a question with its positive and negative documents."""

    id: str
    body: str
    pos_docs: list[PositiveDoc] = []
    neg_docs: list[MinedNegativeDoc] = []


# ---------------------------------------------------------------------------
# Text filling
# ---------------------------------------------------------------------------


def _fill_neg_docs_text(
    baselines_dir: Path,
    ids_per_baseline: dict[str, dict[str, int]],
    baseline: str,
    neg_docs: list[MinedNegativeDoc],
) -> None:
    """Fill ``text`` for each neg_doc using byte-offset lookup.  Mutates in place."""
    baseline_ids: dict[str, int] = ids_per_baseline.get(str(baseline), {})
    baseline_path: Path = baselines_dir / f"pubmed_baseline_{baseline}.jsonl"
    if not baseline_path.exists():
        for doc in neg_docs:
            doc.text = ""
        return
    with baseline_path.open("rb") as f:
        for doc in neg_docs:
            pmid: str = str(doc.id)
            offset: int | None = baseline_ids.get(pmid)
            if offset is None:
                doc.text = ""
                continue
            f.seek(offset)
            line: bytes = f.readline()
            if not line:
                doc.text = ""
                continue
            pub: dict[str, str] = orjson.loads(line)
            title: str = pub.get("title", "")
            abstract: str = pub.get("abstract", "")
            doc.text = title + "  " + abstract


# ---------------------------------------------------------------------------
# Core mining function
# ---------------------------------------------------------------------------


def mine_negatives(
    training_file: Path,
    indexes_dir: Path,
    output_file: Path,
    baselines_dir: Path,
    ids_per_baseline_file: Path,
    *,
    k1: float = 0.4,
    b: float = 0.3,
    num_results: int = 100,
) -> None:
    """Mine BM25 negatives for all training questions and write to JSONL.

    Parameters
    ----------
    training_file:
        JSONL training data (each line: ``{id, body, documents, baseline}``).
    indexes_dir:
        Directory containing per-year PISA indexes.
    output_file:
        Destination JSONL for negatives.
    baselines_dir:
        Directory with ``pubmed_baseline_*.jsonl`` files.
    ids_per_baseline_file:
        JSON mapping ``{year: {pmid: byte_offset}}``.
    k1, b:
        BM25 parameters.
    num_results:
        Number of negative documents per question.
    """
    if not training_file.exists():
        msg: str = f"Training file '{training_file}' doesn't exist."
        raise FileNotFoundError(msg)
    if not indexes_dir.exists():
        msg = f"Indexes directory '{indexes_dir}' doesn't exist."
        raise FileNotFoundError(msg)
    if not baselines_dir.exists():
        msg = f"Baselines directory '{baselines_dir}' doesn't exist."
        raise FileNotFoundError(msg)
    if not ids_per_baseline_file.exists():
        msg = f"IDs-per-baseline file '{ids_per_baseline_file}' doesn't exist."
        raise FileNotFoundError(msg)

    print(f"Loading ids_per_baseline from '{ids_per_baseline_file}'...")
    ids_per_baseline: dict[str, dict[str, int]] = orjson.loads(
        ids_per_baseline_file.read_bytes()
    )

    print(f"Loading training data from '{training_file}'...")
    training_data: list[TrainingQuestion] = []
    with training_file.open("rb") as f:
        for line in f:
            training_data.append(msgspec.json.decode(line, type=TrainingQuestion))

    print(f"Loading indexes from '{indexes_dir}'...")
    fetch_size: int = num_results + 100
    index_pools: dict[str, object] = {
        index_dir.name.split("_")[-1]: load_index(index_dir).bm25(  # type: ignore[attr-defined]
            k1=k1, b=b, num_results=fetch_size, threads=32
        )
        for index_dir in tqdm(
            indexes_dir.iterdir(), desc="Loading indexes", unit="index"
        )
        if index_dir.is_dir()
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb+") as f:
        for question in tqdm(
            training_data, desc="Processing questions", unit="question"
        ):
            if not question.baseline:
                msg = (
                    f"Question '{question.id}' has no 'baseline' field."
                )
                raise ValueError(msg)

            bm25: object | None = index_pools.get(str(question.baseline))
            if bm25 is None:
                msg = (
                    f"No index found for baseline '{question.baseline}' "
                    f"(question '{question.id}'). "
                    f"Available: {sorted(index_pools.keys())}."
                )
                raise FileNotFoundError(msg)

            pos_docs_ids: set[str] = {
                str(doc.id) for doc in question.documents
            }

            query_df: pd.DataFrame = pd.DataFrame(
                [{"qid": question.id, "query": question.body}]
            )
            results: pd.DataFrame = bm25.transform(query_df)  # type: ignore[attr-defined]

            neg_docs: list[MinedNegativeDoc] = [
                MinedNegativeDoc(
                    id=str(row["docno"]),
                    text="",
                    score=float(row["score"]),
                )
                for _, row in results.iterrows()
                if str(row["docno"]) not in pos_docs_ids
            ]
            neg_docs = sorted(neg_docs, key=lambda x: -x.score)[:num_results]

            _fill_neg_docs_text(
                baselines_dir, ids_per_baseline, str(question.baseline), neg_docs
            )

            output: NegativeMiningOutput = NegativeMiningOutput(
                id=question.id,
                body=question.body,
                pos_docs=[
                    PositiveDoc(id=d.id, text=d.text)
                    for d in question.documents
                ],
                neg_docs=neg_docs,
            )
            f.write(msgspec.json.encode(output) + b"\n")

    print(f"Saved results to '{output_file}'.")
