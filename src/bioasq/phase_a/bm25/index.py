"""BM25 index management for Phase A retrieval.

Wraps :mod:`pyterrier_pisa` for indexing PubMed baselines.

Refactored from ``phaseA-BM25/index.py``.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import msgspec
import orjson
from tqdm import tqdm


class PubMedArticle(msgspec.Struct, frozen=True):
    """Minimal PubMed article for indexing."""

    pmid: str
    title: str = ""
    abstract: str = ""


class IndexDocument(msgspec.Struct, frozen=True):
    """Document record for PISA indexing."""

    docno: str
    text: str


def load_index(index_dir: Path) -> object:
    """Load a PISA index from disk.

    Parameters
    ----------
    index_dir:
        Path to the index directory.

    Returns
    -------
    :class:`pyterrier_pisa.PisaIndex` instance.

    Raises
    ------
    FileNotFoundError
        If the index directory does not exist.
    """
    from pyterrier_pisa import PisaIndex

    if not index_dir.exists():
        msg: str = f"Index directory '{index_dir}' doesn't exist."
        raise FileNotFoundError(msg)

    return PisaIndex(str(index_dir), text_field="text", threads=32)


def create_index(baseline: Path, output_dir: Path) -> None:
    """Create a PISA index from a PubMed baseline JSONL file.

    Parameters
    ----------
    baseline:
        Path to the ``pubmed_baseline_*.jsonl`` file.
    output_dir:
        Directory to store the index.
    """
    from pyterrier_pisa import PisaIndex

    output_dir.mkdir(parents=True, exist_ok=True)
    index: object = PisaIndex(str(output_dir), text_field="text")
    index.index(_load_collection(baseline))  # type: ignore[attr-defined]


def _load_collection(
    baseline: Path,
) -> Generator[dict[str, str], None, None]:
    """Yield ``{"docno": pmid, "text": title + abstract}`` from baseline JSONL."""
    with open(baseline, "rb") as f:
        for line in tqdm(f, desc="Loading collection"):
            pub: PubMedArticle = msgspec.json.decode(line, type=PubMedArticle)
            yield {
                "docno": pub.pmid,
                "text": " ".join([pub.title, pub.abstract]),
            }
