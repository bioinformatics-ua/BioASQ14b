import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import orjson
from tqdm import tqdm
import numpy as np
import typer
import gc

app = typer.Typer()


def load_collection(path: Path):
    with open(path) as f:
        for article in map(orjson.loads, f):
            yield article["title"] + " " + article["abstract"]


def chunked_load_collection(path: Path, chunk_size: int):
    chunk = []  # Initialize an empty list to hold a chunk of articles
    for article in load_collection(path):
        chunk.append(article)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []  # Reset chunk
    if chunk:  # If there are remaining articles in the chunk, yield them
        yield chunk


@app.command()
def main(
    baseline: Path = typer.Argument(..., help="The path to the JSONL baseline."),
    model_name: str = typer.Option(
        "unsloth/bge-m3",
        "-m",
        "--model",
        help="The name of the model to use.",
    ),
    output_dir: Path = typer.Option(
        Path("../dense_vectors"),
        "-o",
        "--output-dir",
        help="The directory to save the dense vectors to.",
    ),
    chunk_size: int = typer.Option(100_000, help="The size of the chunk."),
    batch_size: int = typer.Option(
        512, help="Encode batch size. Lower = less GPU memory."
    ),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name, trust_remote_code=True)
    pool = model.start_multi_process_pool(target_devices=["cuda:0", "cuda:1"])

    year = baseline.name.split("_")[-1].rstrip(".jsonl")

    existing_files = {f.name for f in output_dir.glob(f"{year}_*.npy")}
    for i, batch_text in enumerate(
        tqdm(
            chunked_load_collection(baseline, chunk_size),
            desc="Processing chunk",
            unit="chunk",
        )
    ):
        output_file = output_dir / f"{year}_{i * chunk_size}_{(i + 1) * chunk_size}.npy"

        # Skip if file already exists
        if output_file.name in existing_files:
            print(f"Skipping {output_file}: already exists.")
            continue

        embeddings = model.encode(
            batch_text,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            pool=pool,
        )

        # clean the grad in cache
        torch.cuda.empty_cache()
        gc.collect()

        np.save(output_file, embeddings)
        print(f"Saved embeddings to {output_file}")


if __name__ == "__main__":
    app()
