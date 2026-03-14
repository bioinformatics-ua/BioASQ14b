"""
Process the training data with the given baselines and save the results to a jsonl file.

- Injects the content of the baseline documents
- Removes snippets and concepts (unused in our approach)

The output is a jsonl file that can be used for training.
"""
from pyterrier_pisa import PisaRetrieve

from tqdm import tqdm

from typing import Any

from pathlib import Path

import orjson
import typer
from index import load_index

app = typer.Typer()


@app.command()
def main(
    file: Path = typer.Argument(
        ..., help="The original training data file from BioASQ to process."
    ),
    indexes_dir: Path = typer.Option(
        Path("../data/indexes"),
        "-i",
        "--indexes",
        help="The directory containing the index directories.",
    ),
    out_file: Path | None = typer.Option(
        None,
        "-o",
        "--out",
        help="The output file to save the results to. By default, it is the original file with '_processed' appended to the name.",
    ),
):
    if not indexes_dir.exists():
        raise FileNotFoundError(f"Indexes directory '{indexes_dir}' doesn't exist.")

    if out_file is None:
        out_file = file.parent / file.name.replace(".json", "_processed.jsonl")

    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing file '{file}' and saving results to '{out_file}'...")

    indexes: list[PisaRetrieve] = [load_index(index).bm25(k1=0.4, b=0.3) for index in indexes_dir.iterdir()]

    with file.open("r") as f, out_file.open("wb") as out:
        questions: list[dict[str, Any]] = orjson.loads(f.read())["questions"]
        for question in tqdm(questions, desc="Processing questions", position=0):
            del question["snippets"]
            del question["concepts"]
            document_ids: list[dict[str, str]] = [
                {"id": url.split("/")[-1]} for url in question["documents"]
            ]
            for idx, document_id in tqdm(
                enumerate(document_ids),
                desc="Searching documents",
                position=1,
                total=len(document_ids),
            ):
                for index in indexes:
                    results = index.search(document_id["id"])
                    if results:
                        print("pixa", results)
                        document_ids[idx]["text"] = results[0]["text"]
                        break
            question["documents"] = document_ids
            out.write(orjson.dumps(question) + b"\n")


if __name__ == "__main__":
    app()
