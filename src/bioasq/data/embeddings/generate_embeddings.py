"""
docker run --gpus '"device=0"' -p 8080:80 -v $PWD/data/dense_vectors:/data --pull always \
  ghcr.io/huggingface/text-embeddings-inference:86-1.9 \
  --model-id BAAI/bge-m3 --max-batch-tokens 16384 --dtype float16

docker run --gpus '"device=1"' -p 8081:80 -v $PWD/data/dense_vectors:/data --pull always \
  ghcr.io/huggingface/text-embeddings-inference:86-1.9 \
  --model-id BAAI/bge-m3 --max-batch-tokens 16384 --dtype float16

"""

import asyncio
import random
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import httpx
import numpy as np
import typer
from tqdm.asyncio import tqdm

from bioasq.common.io import load_collection

if TYPE_CHECKING:
    from collections.abc import Awaitable

app = typer.Typer()

ENDPOINTS = ["http://localhost:8080/embed"]  # "http://localhost:8081/embed"
MAX_EMBEDDINGS_PER_GPU = 16
N_GPUS = len(ENDPOINTS)


async def call_tei(
    client: httpx.AsyncClient, url: str, texts: list[str], semaphore: asyncio.Semaphore
) -> list[list[float]]:
    """Sends a batch to a specific GPU endpoint."""
    async with semaphore:
        response = await client.post(url, json={"inputs": texts}, timeout=None)
        return response.json()


async def process_chunk_async(
    texts: list[str], output_file: Path, tei_batch_size: int = 16
) -> None:
    semaphore = asyncio.Semaphore(MAX_EMBEDDINGS_PER_GPU * N_GPUS)  # This is the LIMIT of TEI

    async with httpx.AsyncClient() as client:
        tasks: list[Awaitable[list[list[float]]]] = []

        for i in range(0, len(texts), tei_batch_size):
            url = random.choice(ENDPOINTS)
            batch = texts[i : i + tei_batch_size]
            tasks.append(call_tei(client, url, batch, semaphore))

        # Gather results for this 100k chunk
        results = await tqdm.gather(*tasks, desc="Encoding Batches", leave=False)
        print(f"Results length: {len(results)}")
        # Flatten and save
        embeddings = np.concatenate([np.array(r, dtype=np.float16) for r in results])
        print(f"Saving embeddings to {output_file}")
        print(f"Embeddings shape: {embeddings.shape}")
        np.save(output_file, embeddings)
        print(f"Saved embeddings to {output_file}")


@app.command()
def main(
    baseline: Annotated[Path, typer.Argument(help="Path to JSONL.")],
    output_dir: Annotated[Path, typer.Option()] = Path("../dense_vectors_numpy"),
    chunk_size: Annotated[int, typer.Option(help="Size of the .npy files.")] = 100_000,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    year = baseline.name.split("_")[-1].replace(".jsonl", "")  # TODO THIS CAN BE SHIT

    current_chunk: list[str] = []
    chunk_count = 0

    for text in tqdm(load_collection(baseline), desc="Total Progress"):
        current_chunk.append(text)

        if len(current_chunk) == chunk_size:
            out_path = (
                output_dir
                / f"{year}_{chunk_count * chunk_size}_{(chunk_count + 1) * chunk_size}.npy"
            )
            if not out_path.exists():
                asyncio.run(process_chunk_async(current_chunk, out_path))

            current_chunk = []
            chunk_count += 1

    # Handle final remainder
    if current_chunk:
        out_path = (
            output_dir
            / f"{year}_{chunk_count * chunk_size}_{(chunk_count * chunk_size) + len(current_chunk)}"
            ".npy"
        )
        asyncio.run(process_chunk_async(current_chunk, out_path))


if __name__ == "__main__":
    app()
