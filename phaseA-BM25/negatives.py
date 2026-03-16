"""
Search for negatives in the training data.

Each training line must have: id, body, documents (positives), baseline.
Negatives are the top 500 BM25 results from the question's baseline index
that are NOT in documents (i.e. not positive docs).

Output format of a JSONL line:
{
    "id": "question_id",
    "body": "query",
    "pos_docs": [{"id": "pmid", "text": "text"}],
    "neg_docs": [{"id": "pmid", "text": "text", "score": score}]
}
"""

from typing import Any, Literal, TypeAlias, cast

import orjson
from pyterrier_pisa import PisaRetrieve
import typer
import pandas as pd
from index import load_index
from pathlib import Path
from tqdm import tqdm
from __init__ import QuestionWithNegatives

app = typer.Typer()

Year: TypeAlias = Literal[
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023", #11
    "2024", #12
    "2025", #13
    "2026",
]

# This shit makes me wanna make C code,,, changing the reference mf
def _fill_neg_docs_text(
    baselines_dir: Path,
    ids_per_baseline: dict[str, dict[str, int]],
    baseline: str,
    neg_docs: list[dict[str, Any]],
) -> None:
    """Fill 'text' for each neg_doc using ids_per_baseline offset lookup. Mutates in place."""
    baseline_ids = ids_per_baseline.get(str(baseline), {})
    baseline_path = baselines_dir / f"pubmed_baseline_{baseline}.jsonl"
    if not baseline_path.exists():
        for doc in neg_docs:
            doc["text"] = ""
        return
    with baseline_path.open("rb") as f:
        for doc in neg_docs:
            pmid = str(doc["id"])
            offset = baseline_ids.get(pmid)
            if offset is None:
                doc["text"] = ""
                continue
            f.seek(offset)
            line = f.readline()
            if not line:
                doc["text"] = ""
                continue
            pub = orjson.loads(line)
            title = pub.get("title") or ""
            abstract = pub.get("abstract") or ""
            doc["text"] = title + "  " + abstract


@app.command()
def main(
    training_file: Path = typer.Argument(..., help="The training file to use."),
    indexes_dir: Path = typer.Option(
        Path("../data/indexes"),
        "-i",
        "--indexes-dir",
        help="The directory where the indexes are stored.",
    ),
    output_file: Path = typer.Option(
        Path("../data/negatives.jsonl"),
        "-o",
        "--output-file",
        help="The file to save the results to.",
    ),
    baselines_dir: Path = typer.Option(
        Path("../data/baselines"),
        "--baselines-dir",
        help="Directory containing pubmed_baseline_*.jsonl files for doc text lookup.",
    ),
    ids_per_baseline_file: Path = typer.Option(
        Path("../data/ids_per_baseline.json"),
        "-p",
        "--ids-per-baseline",
        help="JSON file mapping baseline year -> pmid -> byte offset (from data/get_pmid.py).",
    ),
    k1: float = typer.Option(0.4, "-k", "--k1", help="The k1 parameter for BM25."),
    b: float = typer.Option(0.3, "-b", "--b", help="The b parameter for BM25."),
    num_results: int = typer.Option(
        100,
        "-n",
        "--num-results",
        help="The number of negative documents to return per question.",
    ),
):
    if not training_file.exists():
        raise FileNotFoundError(f"Training file '{training_file}' doesn't exist.")

    if not indexes_dir.exists():
        raise FileNotFoundError(f"Indexes directory '{indexes_dir}' doesn't exist.")

    if not baselines_dir.exists():
        raise FileNotFoundError(f"Baselines directory '{baselines_dir}' doesn't exist.")

    if not ids_per_baseline_file.exists():
        raise FileNotFoundError(
            f"IDs per baseline file '{ids_per_baseline_file}' doesn't exist. "
            "Run: uv run python -m data.get_pmid <training_json> -o ids_per_baseline.json"
        )

    print(f"Loading ids_per_baseline from '{ids_per_baseline_file}'...")
    ids_per_baseline: dict[str, dict[str, int]] = orjson.loads(
        ids_per_baseline_file.read_bytes()
    )

    print(f"Loading training data from '{training_file}'...")
    with training_file.open("rb") as f:
        training_data: list[dict[str, Any]] = [orjson.loads(line) for line in f]

    print(f"Loading indexes from '{indexes_dir}'...")
    # Request extra results to have enough negatives after filtering out positives
    fetch_size = num_results + 100
    index_pools: dict[Year, PisaRetrieve] = {
        index_dir.name.split("_")[-1]: load_index(index_dir).bm25(
            k1=k1, b=b, num_results=fetch_size, threads=32
        )
        for index_dir in tqdm(
            indexes_dir.iterdir(), desc="Loading indexes", unit="index"
        )
        if index_dir.is_dir()
    }

    with output_file.open("wb+") as f:
        for question in tqdm(
            training_data, desc="Processing questions", unit="question"
        ):
            baseline = question.get("baseline")
            if not baseline:
                raise ValueError(
                    f"Question '{question['id']}' has no 'baseline' field. "
                    "Negatives must be fetched from the baseline of that query."
                )

            bm25 = index_pools.get(cast(Year, str(baseline)))
            if bm25 is None:
                raise FileNotFoundError(
                    f"No index found for baseline '{baseline}' (question '{question['id']}'). "
                    f"Available baselines: {sorted(index_pools.keys())}."
                )

            output: QuestionWithNegatives = {
                "id": question["id"],
                "body": question["body"],
                "pos_docs": question["documents"],
                "neg_docs": [],
            }

            pos_docs_ids = {str(doc["id"]) for doc in output["pos_docs"]}

            query_df = pd.DataFrame(
                [{"qid": question["id"], "query": question["body"]}]
            )
            results: pd.DataFrame = bm25.transform(query_df)

            neg_docs = [
                {"id": str(row["docno"]), "text": "", "score": row["score"]}
                for _, row in results.iterrows()
                if str(row["docno"]) not in pos_docs_ids
            ]
            neg_docs = sorted(neg_docs, key=lambda x: -x["score"])[:num_results]

            _fill_neg_docs_text(baselines_dir, ids_per_baseline, str(baseline), neg_docs)
            output["neg_docs"] = neg_docs

            f.write(orjson.dumps(output) + b"\n")

    print(f"Saved results to '{output_file}'.")


if __name__ == "__main__":
    app()
