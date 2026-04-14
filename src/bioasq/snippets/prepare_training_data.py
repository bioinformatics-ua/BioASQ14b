"""Prepare training data for snippet extraction LoRA.

Reads BioASQ training data (training14b.json with gold snippets) and the
inflated JSONL (with full document text), joins them by question ID, and
produces (question, document_text, gold_snippets) triples suitable for
training a generative snippet extractor.

Output format (JSONL):
    {
        "question_id": str,
        "question_body": str,
        "question_type": str,
        "doc_pmid": str,
        "doc_text": str,
        "snippets": [str, ...]      # gold snippet texts that appear in doc_text
    }

Usage:
    python -m bioasq.snippets.prepare_training_data \
        --training-json data/training14b/training14b.json \
        --inflated-jsonl data/quality/training14b_inflated_clean_wContents.jsonl \
        --output data/training/snippet_extraction/gold_pairs.jsonl
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any

import typer
from tqdm import tqdm

app = typer.Typer()


# ---------------------------------------------------------------------------
# Fuzzy substring matching
# ---------------------------------------------------------------------------


def _normalise_ws(text: str) -> str:
    """Collapse all whitespace to single spaces for matching."""
    return re.sub(r"\s+", " ", text).strip()


def fuzzy_find(snippet: str, doc_text: str) -> tuple[bool, int, int]:
    """Try to locate *snippet* inside *doc_text*.

    Returns (found, start, end) where start/end are character offsets
    in the **original** doc_text.  Uses whitespace-normalized matching
    so minor newline / double-space differences are tolerated.
    """
    # Fast path: exact match
    idx = doc_text.find(snippet)
    if idx >= 0:
        return True, idx, idx + len(snippet)

    # Normalised match — build a position map from normalised -> original
    norm_doc = _normalise_ws(doc_text)
    norm_snip = _normalise_ws(snippet)
    idx = norm_doc.find(norm_snip)
    if idx >= 0:
        return True, idx, idx + len(norm_snip)

    # Try with first 60 chars as anchor (handles truncation differences)
    anchor = norm_snip[:60]
    idx = norm_doc.find(anchor)
    if idx >= 0:
        return True, idx, idx + len(norm_snip)

    return False, -1, -1


def _extract_pmid(url: str) -> str:
    """Extract PubMed ID from a URL like http://...pubmed/12345."""
    return url.rstrip("/").split("/")[-1]


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------


def _load_training_questions(path: Path) -> dict[str, dict[str, Any]]:
    """Load training14b.json and index by question ID."""
    with path.open() as f:
        data = json.load(f)
    return {q["id"]: q for q in data["questions"]}


def _load_inflated_docs(path: Path) -> dict[str, dict[str, str]]:
    """Load inflated JSONL and build {question_id: {pmid: doc_text}} map."""
    result: dict[str, dict[str, str]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj["id"]
            docs: dict[str, str] = {}
            for d in obj.get("documents", []):
                if isinstance(d, dict):
                    docs[d["id"]] = d["text"]
            result[qid] = docs
    return result


def build_training_pairs(
    training_questions: dict[str, dict[str, Any]],
    inflated_docs: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Join snippets from training with document texts from inflated."""
    pairs: list[dict[str, Any]] = []
    stats = {"total_snippets": 0, "matched": 0, "missed": 0, "no_doc_text": 0}

    for qid, tq in tqdm(training_questions.items(), desc="Building pairs"):
        doc_texts = inflated_docs.get(qid, {})
        if not doc_texts:
            continue

        snippets_by_doc: dict[str, list[str]] = {}
        for s in tq.get("snippets", []):
            stats["total_snippets"] += 1
            pmid = _extract_pmid(s.get("document", ""))
            doc_text = doc_texts.get(pmid)
            if doc_text is None:
                stats["no_doc_text"] += 1
                continue

            found, _, _ = fuzzy_find(s["text"], doc_text)
            if found:
                stats["matched"] += 1
                snippets_by_doc.setdefault(pmid, []).append(s["text"])
            else:
                stats["missed"] += 1

        # Emit one training example per (question, document) pair
        for pmid, snip_texts in snippets_by_doc.items():
            pairs.append(
                {
                    "question_id": qid,
                    "question_body": tq["body"],
                    "question_type": tq["type"],
                    "doc_pmid": pmid,
                    "doc_text": doc_texts[pmid],
                    "snippets": snip_texts,
                }
            )

    print("\n--- Stats ---")
    print(f"Total snippets:   {stats['total_snippets']}")
    print(f"Matched:          {stats['matched']}")
    print(f"Missed (fuzzy):   {stats['missed']}")
    print(f"No doc text:      {stats['no_doc_text']}")
    print(f"Training pairs:   {len(pairs)}")

    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    training_json: Annotated[Path, typer.Option(help="Path to training14b.json")] = Path(
        "data/training14b/training14b.json"
    ),
    inflated_jsonl: Annotated[
        Path, typer.Option(help="Path to inflated JSONL with doc texts")
    ] = Path("data/quality/training14b_inflated_clean_wContents.jsonl"),
    output: Annotated[Path, typer.Option(help="Output JSONL path")] = Path(
        "data/training/snippet_extraction/gold_pairs.jsonl"
    ),
) -> None:
    """Build (question, doc_text, snippets) training pairs."""
    training_questions = _load_training_questions(training_json)
    print(f"Loaded {len(training_questions)} training questions")

    inflated_docs = _load_inflated_docs(inflated_jsonl)
    print(f"Loaded doc texts for {len(inflated_docs)} questions")

    pairs = build_training_pairs(training_questions, inflated_docs)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(pairs)} training pairs to {output}")


if __name__ == "__main__":
    app()
