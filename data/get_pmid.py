import orjson
import typer
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

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

    ids_per_baseline: defaultdict[str, dict[str, int]] = defaultdict(
        dict
    )  # year -> pmid -> line_idx
    baselines: list[Path] = [
        baseline
        for baseline in baselines_dir.iterdir()
        if baseline.is_file()
        and baseline.name.startswith("pubmed_baseline_")
        and baseline.name.endswith(".jsonl")
    ]

    for baseline in tqdm(baselines, desc="Processing baselines", position=0):
        year = baseline.name.split("_")[-1].rstrip(".jsonl")
        with baseline.open("r") as f:
            for idx, doc in tqdm(enumerate(f), desc="Processing baseline", position=1):
                pmid = doc.lstrip('{"pmid": "').split('"', 1)[0]
                ids_per_baseline[year][pmid] = idx

    with out_file.open("wb") as fo:
        fo.write(orjson.dumps(ids_per_baseline))
    print(f"Saved results to '{out_file}'.")


if __name__ == "__main__":
    app()
