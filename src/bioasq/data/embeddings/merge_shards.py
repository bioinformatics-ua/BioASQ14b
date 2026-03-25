"""Merge distributed similarity shards into a lookup dictionary.

Combines multiple .npy shard files containing similarity scores
between document pairs into a JSON lookup table mapping PMIDs to similar PMIDs.
"""

from collections import defaultdict
from pathlib import Path
from typing import Annotated

import numpy as np
import orjson
import typer
from tqdm import tqdm

from bioasq.common.io import load_collection

type PMID = str
app = typer.Typer()


@app.command()
def main(
    shards_dir: Annotated[Path, typer.Argument(help="Path to shards directory.")],
    baseline: Annotated[Path, typer.Argument(help="Path to JSONL.")],
    output_file: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            help="Path to output file.",
        ),
    ] = Path("../similarity_results/lookup.json"),
    threshold: Annotated[
        float, typer.Option("--threshold", "-t", help="Similarity threshold.")
    ] = 0.9,
) -> None:
    if not shards_dir.exists():
        raise FileNotFoundError(f"Shards directory '{shards_dir}' doesn't exist.")
    if not baseline.exists():
        raise FileNotFoundError(f"Baseline file '{baseline}' doesn't exist.")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading collection from '{baseline}'...")
    collection_iter = load_collection(
        baseline, constraints=[lambda x: len(f"{x.title} {x.abstract}") > 100], id_present=True
    )
    # TODO here we will consume the colletction
    # Maps the line id to the pmid
    # Needs duck duck db
    collection: dict[int, PMID] = {i: doc[0] for i, doc in enumerate(collection_iter)}

    shard_files = list(shards_dir.glob("*.npy"))

    shard_files.sort(key=lambda x: int(x.stem.split("_")[1]))
    lookup: defaultdict[PMID, list[tuple[PMID, float]]] = defaultdict(list)
    for shard_file in tqdm(shard_files, desc="Processing shards", unit="shard"):
        with shard_file.open("rb") as f:
            indexes0, indexes1, scores = np.load(f, allow_pickle=False).T
            for idx0, idx1, score in zip(indexes0, indexes1, scores, strict=False):
                if score < threshold:
                    continue
                if idx0 in collection and idx1 in collection:
                    pmid0 = collection[idx0]
                    pmid1 = collection[idx1]
                    lookup[pmid0].append((pmid1, score))
                    lookup[pmid1].append((pmid0, score))

    with output_file.open("wb") as f:
        f.write(orjson.dumps(lookup))


if __name__ == "__main__":
    app()
