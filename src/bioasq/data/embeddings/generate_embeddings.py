"""Embedding generation using Text Embeddings Inference (TEI).

This script connects to a TEI endpoint to embed a collection of PubMed
documents iteratively and saves the dense vectors as numpy arrays.

Each shard writes ``<stem>.npy`` (float16 matrix, one row per document in order)
and ``<stem>.ids.json`` (same-length list of PMIDs / document ids as strings).
TEI returns embeddings in the same order as ``inputs``, so rows align with ids.

Docker usage examples:
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

from bioasq.common.aliases import DocumentId
from bioasq.common.io import load_collection, save_json
from bioasq.common.utils import typer_async
from bioasq.data.database import create_indexes, update_article_embedding

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
    texts: list[tuple[DocumentId, str]], output_file: Path, tei_batch_size: int = 16
) -> None:
    semaphore = asyncio.Semaphore(MAX_EMBEDDINGS_PER_GPU * N_GPUS)  # This is the LIMIT of TEI
    ordered_ids: list[int] = [int(doc_id) for doc_id, _ in texts]

    async with httpx.AsyncClient() as client:
        tasks: list[Awaitable[list[list[float]]]] = []

        for i in range(0, len(texts), tei_batch_size):
            url = random.choice(ENDPOINTS)
            batch = texts[i : i + tei_batch_size]
            text_inputs = [t for _, t in batch]
            tasks.append(call_tei(client, url, text_inputs, semaphore))

        # Gather results for this 100k chunk (order matches batch submission order)
        results = await tqdm.gather(*tasks, desc="Encoding Batches", leave=False)
        print(f"Results length: {len(results)}")
        # Flatten and save (row i <-> ordered_ids[i], same as TEI input order)
        embeddings = np.concatenate([np.array(r, dtype=np.float16) for r in results])
        print(f"Embeddings shape: {embeddings.shape}")
        if embeddings.shape[0] != len(ordered_ids):
            raise RuntimeError(
                f"Embedding rows ({embeddings.shape[0]}) != ids ({len(ordered_ids)})"
            )

        print(f"Saving embeddings to {output_file}")
        np.save(output_file, embeddings)
        save_json(ordered_ids, output_file.with_name(f"{output_file.stem}.ids.json"))

        print("Saving embeddings in database")
        for id_, embedding in zip(ordered_ids, embeddings, strict=True):
            await update_article_embedding(id_, embedding)


@app.command()
@typer_async
async def main(
    baseline: Annotated[Path, typer.Argument(help="Path to JSONL.")],
    output_dir: Annotated[Path, typer.Option()] = Path("../dense_vectors_numpy"),
    chunk_size: Annotated[int, typer.Option(help="Rows per shard (.npy + .ids.json).")] = 100_000,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    year = baseline.name.split("_")[-1].replace(".jsonl", "")  # TODO THIS CAN BE SHIT

    current_chunk: list[tuple[DocumentId, str]] = []
    chunk_count = 0

    for id_, text in tqdm(load_collection(baseline, None, id_present=True), desc="Total Progress"):
        current_chunk.append((id_, text))

        if len(current_chunk) == chunk_size:
            out_path = (
                output_dir
                / f"{year}_{chunk_count * chunk_size}_{(chunk_count + 1) * chunk_size}.npy"
            )
            ids_path = out_path.with_name(f"{out_path.stem}.ids.json")
            if not out_path.exists() or not ids_path.exists():
                await process_chunk_async(current_chunk, out_path)

            current_chunk = []
            chunk_count += 1

    # Handle final remainder
    if current_chunk:
        out_path = (
            output_dir
            / f"{year}_{chunk_count * chunk_size}_{(chunk_count * chunk_size) + len(current_chunk)}"
            ".npy"
        )
        remainder_ids_path = out_path.with_name(f"{out_path.stem}.ids.json")
        if not out_path.exists() or not remainder_ids_path.exists():
            await process_chunk_async(current_chunk, out_path)

    print("Creating indexes")
    await create_indexes()
    print("Indexes created")


if __name__ == "__main__":
    app()
