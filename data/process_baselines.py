"""
Process the training data with the given baselines and save the results to a jsonl file.

- Injects the content of the baseline documents
- Removes snippets and concepts (unused in our approach)

The output is a jsonl file that can be used for training.
"""

from tqdm import tqdm

from typing import Any

from pathlib import Path

import orjson
import typer
from . import PMID, IDsPerBaseline, Year

app = typer.Typer()


def _find_baseline_and_line_idx(
    ids_per_baseline: IDsPerBaseline, pmid: PMID
) -> tuple[Year, int] | None:
    for year, ids in ids_per_baseline.items():
        if pmid in ids:
            return year, ids[pmid]
    return None


@app.command()
def main(
    file: Path = typer.Argument(
        ..., help="The original training data file from BioASQ to process."
    ),
    baselines_dir: Path = typer.Option(
        Path("./baselines"),
        "-b",
        "--baselines",
        help="The directory containing the baseline files.",
    ),
    ids_per_baseline_file: Path = typer.Option(
        Path("./ids_per_baseline.json"),
        "-i",
        "--ids-per-baseline",
        help="The file containing the IDs per baseline.",
    ),
    out_file: Path | None = typer.Option(
        None,
        "-o",
        "--out",
        help="The output file to save the results to. By default, it is the original file with '_processed' appended to the name.",
    ),
):
    if not baselines_dir.exists():
        raise FileNotFoundError(f"Baselines directory '{baselines_dir}' doesn't exist.")

    if not ids_per_baseline_file.exists():
        raise FileNotFoundError(
            f"IDs per baseline file '{ids_per_baseline_file}' doesn't exist."
        )

    ids_per_baseline: IDsPerBaseline = orjson.loads(ids_per_baseline_file.read_bytes())

    if out_file is None:
        out_file = file.parent / file.name.replace(".json", "_processed.jsonl")

    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing file '{file}' and saving results to '{out_file}'...")

    not_found: list[tuple[str, PMID]] = []

    with file.open("r") as f, out_file.open("wb") as out:
        questions: list[dict[str, Any]] = orjson.loads(f.read())["questions"]
        for question in tqdm(questions, desc="Processing questions", position=0):
            for key in ["snippets", "concepts", "triples"]:
                if key in question:
                    del question[key]

            document_ids: list[dict[str, str]] = [
                {"id": url.split("/")[-2 if url.endswith("/") else -1]}
                for url in question["documents"]
            ]
            for idx, document_id in enumerate(document_ids):
                year, offset = _find_baseline_and_line_idx(
                    ids_per_baseline, document_id["id"]
                ) or (None, None)

                if year is None or offset is None:
                    not_found.append((question["id"], document_id["id"]))
                    continue

                with open(baselines_dir / f"pubmed_baseline_{year}.jsonl", "r") as f:
                    f.seek(offset)
                    doc = orjson.loads(f.readline())
                    document_ids[idx]["text"] = doc["title"] + "  " + doc["abstract"]

            question["documents"] = document_ids
            out.write(orjson.dumps(question) + b"\n")

    print(f"Found {len(not_found)} documents not found in any baseline.")
    with open("not_found.jsonl", "wb") as f:
        for question_id, pmid in not_found:
            f.write(orjson.dumps({"question_id": question_id, "pmid": pmid}) + b"\n")


if __name__ == "__main__":
    app()
