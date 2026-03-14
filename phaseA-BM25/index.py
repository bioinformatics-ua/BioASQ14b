from typing import Any, Generator
import orjson
from tqdm import tqdm
from pyterrier_pisa import PisaIndex
from pathlib import Path

import typer

app = typer.Typer()


def load_index(index_dir: Path) -> PisaIndex:
    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory '{index_dir}' doesn't exist.")

    return PisaIndex(str(index_dir), text_field="text", threads=32)


def create_index(baseline: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    index = PisaIndex(str(output_dir), text_field="text")
    index.index(_load_collection(baseline))


def _load_collection(baseline: Path) -> Generator[dict[str, Any], None, None]:
    with open(baseline, "r") as f:
        for line in tqdm(f, desc="Loading collection"):
            pub: dict[str, Any] = orjson.loads(line)
            yield {
                "docno": pub["pmid"],
                "text": " ".join([pub["title"], pub["abstract"]]),
            }


@app.command()
def _main(
    baseline: Path = typer.Option(..., help="The baseline to index."),
    output_dir: Path | None = typer.Option(
        ..., "-o", "--out", help="The directory to save the index to."
    ),
):
    if output_dir is None:
        output_dir = baseline.parent / "indexes" / baseline.name.split(".")[0]

    create_index(baseline, output_dir)
    print(f"Collection '{baseline}' has been indexed successfully at '{output_dir}'.")


if __name__ == "__main__":
    app()
