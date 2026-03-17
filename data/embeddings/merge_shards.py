import orjson
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import typer

type PMID = str
app = typer.Typer()


@app.command()
def main(
    shards_dir: Path = typer.Argument(..., help="Path to shards directory."),
    baseline: Path = typer.Argument(..., help="Path to JSONL."),
    output_file: Path = typer.Option(
        Path("../similarity_results/lookup.json"),
        "-o",
        "--output",
        help="Path to output file.",
    ),
):
    if not shards_dir.exists():
        raise FileNotFoundError(f"Shards directory '{shards_dir}' doesn't exist.")
    if not baseline.exists():
        raise FileNotFoundError(f"Baseline file '{baseline}' doesn't exist.")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading collection from '{baseline}'...")
    with baseline.open("rb") as f:
        collection: dict[int, PMID] = {
            i: line.lstrip(b'{"pmid": "').split(b'"', 1)[0].decode("utf-8")
            for i, line in enumerate(f)
            if len(line) < 1000
        }

    shard_files = list(shards_dir.glob("*.npy"))

    shard_files.sort(key=lambda x: int(x.stem.split("_")[1]))
    lookup: defaultdict[PMID, list[tuple[PMID, float]]] = defaultdict(list)
    for shard_file in tqdm(shard_files, desc="Processing shards", unit="shard"):
        with shard_file.open("rb") as f:
            indexes0, indexes1, scores = np.load(f, allow_pickle=False).T
            for idx0, idx1, score in zip(indexes0, indexes1, scores):
                if idx0 in collection and idx1 in collection:
                    pmid0 = collection[idx0]
                    pmid1 = collection[idx1]
                    lookup[pmid0].append((pmid1, score))
                    lookup[pmid1].append((pmid0, score))

    with output_file.open("wb") as f:
        f.write(orjson.dumps(lookup))


if __name__ == "__main__":
    app()
