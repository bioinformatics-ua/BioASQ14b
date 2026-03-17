"""
docker run --gpus '"device=0"' -p 8080:80 -v $PWD/data/dense_vectors:/data --pull always \
  ghcr.io/huggingface/text-embeddings-inference:86-1.9 \
  --model-id BAAI/bge-m3 --max-batch-tokens 16384 --dtype float16

docker run --gpus '"device=1"' -p 8081:80 -v $PWD/data/dense_vectors:/data --pull always \
  ghcr.io/huggingface/text-embeddings-inference:86-1.9 \
  --model-id BAAI/bge-m3 --max-batch-tokens 16384 --dtype float16

"""

import random

import asyncio
import httpx
import orjson
import numpy as np
import typer
from pathlib import Path
from tqdm.asyncio import tqdm

app = typer.Typer()

# TEI Endpoints for your 2x A40s
ENDPOINTS = ["http://localhost:8080/embed", "http://localhost:8081/embed"]


async def call_tei(client, url, texts, semaphore):
    """Sends a batch to a specific GPU endpoint."""
    async with semaphore:
        response = await client.post(url, json={"inputs": texts}, timeout=None)
        return response.json()


def load_collection(path: Path):
    with open(path, "rb") as f:
        for line in f:
            article = orjson.loads(line)
            yield f"{article['title']} {article['abstract']}"


async def process_chunk_async(texts, output_file, tei_batch_size=16):
    semaphore = asyncio.Semaphore(32)

    async with httpx.AsyncClient() as client:
        tasks = []

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
    baseline: Path = typer.Argument(..., help="Path to JSONL."),
    output_dir: Path = typer.Option(Path("../dense_vectors_numpy")),
    chunk_size: int = typer.Option(100_000, help="Size of the .npy files."),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    year = baseline.name.split("_")[-1].replace(".jsonl", "")

    current_chunk = []
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
        out_path = output_dir / f"{year}_final.npy"
        asyncio.run(process_chunk_async(current_chunk, out_path))


if __name__ == "__main__":
    app()
