import orjson
import typer
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from . import IDsPerBaseline

app = typer.Typer()


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
    out_file: Path = typer.Option(
        Path("ids_per_baseline.json"),
        "-o",
        "--out",
        help="The output file to save the results to.",
    ),
):
    if not file.exists():
        raise FileNotFoundError(f"File '{file}' doesn't exist.")

    ids_per_baseline: IDsPerBaseline = defaultdict(dict)  # year -> pmid -> line_idx
    baselines: list[Path] = [
        baseline
        for baseline in baselines_dir.iterdir()
        if baseline.is_file()
        and baseline.name.startswith("pubmed_baseline_")
        and baseline.name.endswith(".jsonl")
    ]

    for baseline in tqdm(baselines, desc="Processing baselines", unit="baseline"):
        year = baseline.name.split("_")[-1].rstrip(".jsonl")
        with baseline.open("rb") as f:
            for doc in f:
                pmid: bytes = doc.lstrip(b'{"pmid": "').split(b'"', 1)[0]
                ids_per_baseline[year][pmid.decode("utf-8")] = f.tell() - len(doc)

    with out_file.open("wb") as fo:
        fo.write(orjson.dumps(ids_per_baseline))
    print(f"Saved results to '{out_file}'.")


if __name__ == "__main__":
    app()
