"""Embedding generation using Text Embeddings Inference (TEI).

This script connects to a TEI endpoint to embed a collection of PubMed
documents iteratively and saves the dense vectors as numpy arrays.

Each shard writes ``<stem>.npy`` (float16 matrix, one row per document in order)
and ``<stem>.ids.json`` (same-length list of PMIDs / document ids as strings).
TEI returns embeddings in the same order as ``inputs``, so rows align with ids.

Docker usage examples:
docker run --device nvidia.com/gpu=0 -p 8080:80 -v $PWD/tei-volume:/data --pull always \
  ghcr.io/huggingface/text-embeddings-inference:86-1.9 \
  --model-id BAAI/bge-m3 --max-batch-tokens 16384 --dtype float16

docker run --device nvidia.com/gpu=1 -p 8081:80 -v $PWD/tei-volume:/data --pull always \
  ghcr.io/huggingface/text-embeddings-inference:86-1.9 \
  --model-id BAAI/bge-m3 --max-batch-tokens 16384 --dtype float16
"""

import asyncio
import random
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import httpx
import numpy as np
import typer
from tqdm.asyncio import tqdm

from bioasq.common.io import load_collection, save_json
from bioasq.common.types import Document
from bioasq.common.utils import typer_async
from bioasq.data.database import (
    add_baseline_ids,
    insert_articles,
    update_article_embedding,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from bioasq.common.aliases import DocumentId


app = typer.Typer()

ENDPOINTS = ["http://localhost:8080/embed", "http://localhost:8081/embed"]
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
    documents: list[Document],
    output_file: Path | None = None,
    insert_into_db: Literal["insert", "update", "none"] = "none",
    tei_batch_size: int = 16,
) -> None:
    semaphore = asyncio.Semaphore(MAX_EMBEDDINGS_PER_GPU * N_GPUS)  # This is the LIMIT of TEI
    ordered_ids: list[int] = [int(doc.pmid) for doc in documents]

    async with httpx.AsyncClient() as client:
        tasks: list[Awaitable[list[list[float]]]] = []

        for i in range(0, len(documents), tei_batch_size):
            url = random.choice(ENDPOINTS)
            batch = documents[i : i + tei_batch_size]
            text_inputs = [d.full_text for d in batch]
            tasks.append(call_tei(client, url, text_inputs, semaphore))

        results = await tqdm.gather(*tasks, desc="Encoding Batches", leave=False)

        embeddings = np.concatenate([np.array(r, dtype=np.float16) for r in results])
        print(f"Embeddings shape: {embeddings.shape}")

        if embeddings.shape[0] != len(documents):
            raise RuntimeError(
                f"Embedding rows ({embeddings.shape[0]}) != documents ({len(documents)})"
            )

        if output_file is not None:
            print(f"Saving embeddings to {output_file}")
            np.save(output_file, embeddings)
            save_json(ordered_ids, output_file.with_name(f"{output_file.stem}.ids.json"))

        try:
            match insert_into_db:
                case "insert":
                    print("Inserting articles and embeddings into database")
                    await insert_articles(documents, embeddings)
                    print("PIXA2 Adding baseline ids")
                    await add_baseline_ids(2026, ordered_ids)
                case "update":
                    print("Updating embeddings in database")
                    for id_, embedding in zip(ordered_ids, embeddings, strict=True):
                        await update_article_embedding(id_, embedding)
                case "none":
                    pass

        except Exception as e:
            print(f"Error inserting articles and embeddings into database: {e}")
            return


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


if __name__ == "__main__":
    app()
