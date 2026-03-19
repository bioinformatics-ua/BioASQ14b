"""
Hydrate BioASQ predictions with baseline document text (title + abstract).

Uses DuckDB for fast batch lookups instead of file seeking (process_golden.py style).
Same logic as data/process_golden.py but significantly faster via indexed DB queries.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import duckdb
import orjson
import typer
from tqdm import tqdm

from data.baseline_duckdb import get_articles_batch, init_pubmed_db

app = typer.Typer()


def _connect_baseline_db(baseline: Path, db_path: Path) -> duckdb.DuckDBPyConnection:
    """Connect to DuckDB, using read-only if db exists (avoids lock conflicts)."""
    if db_path.exists():
        return duckdb.connect(str(db_path), read_only=True)
    return init_pubmed_db(baseline, db_path)


def _pmid_from_url(url: str) -> str:
    """Extract PMID from BioASQ document URL."""
    return url.rstrip("/").split("/")[-1]


@app.command()
def main(
    file: Path = typer.Argument(
        ..., help="The BioASQ JSON file to hydrate (questions with documents)."
    ),
    baseline: Path = typer.Option(
        Path("../data/baselines/pubmed_baseline_2026.jsonl"),
        "-b",
        "--baseline",
        help="Path to baseline JSONL (used to init DuckDB if needed).",
    ),
    baseline_db: Path | None = typer.Option(
        Path("../data/db_baselines/pubmed_baseline_2026.db"),
        "--baseline-db",
        help="Path to DuckDB file. Default: data/db_baselines/{baseline.stem}.db",
    ),
    out_file: Path | None = typer.Option(
        None,
        "-o",
        "--out",
        help="Output jsonl path. Default: input name with '_hydrated.jsonl'.",
    ),
):
    """Inject baseline document text into BioASQ predictions using DuckDB for fast lookups."""
    if not file.exists():
        raise FileNotFoundError(f"Predictions file '{file}' not found.")
    if not baseline.exists():
        raise FileNotFoundError(f"Baseline file '{baseline}' not found.")

    db_path = baseline_db or (
        _PROJECT_ROOT / "data" / "db_baselines" / f"{baseline.stem}.db"
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if out_file is None:
        out_file = file.parent / file.name.replace(".json", "_hydrated.jsonl")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("Connect to DuckDB...")
    con = _connect_baseline_db(baseline, db_path)

    print("Load predictions...")
    with file.open("rb") as f:
        data = orjson.loads(f.read())

    questions = data.get("questions", [])
    if not questions:
        raise ValueError(f"No questions found in '{file}'.")

    # Collect all PMIDs
    all_pmids: set[str] = set()
    for q in questions:
        for doc in q.get("documents", []):
            pmid = _pmid_from_url(doc) if isinstance(doc, str) else doc.get("id", "")
            if pmid:
                all_pmids.add(str(pmid))

    print(f"Batch fetch {len(all_pmids)} documents from DuckDB...")
    batch_texts = get_articles_batch(con, list(all_pmids))
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
                text = batch_texts.get(pmid)
                if text is None:
                    not_found.append((question.get("id", "?"), pmid))
                    continue
                document_ids[idx]["text"] = text

            question["documents"] = document_ids
            out.write(orjson.dumps(question) + b"\n")

    con.close()
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
