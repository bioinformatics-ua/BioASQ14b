"""Embedding generation using Text Embeddings Inference (TEI).

This script connects to a TEI endpoint to embed a collection of PubMed
documents iteratively and saves the dense vectors as numpy arrays.

Each shard writes ``<stem>.npy`` (float16 matrix, one row per document in order)
and ``<stem>.ids.json`` (same-length list of PMIDs / document ids as strings).
TEI returns embeddings in the same order as ``inputs``, so rows align with ids.

Docker usage examples:
docker run --device nvidia.com/gpu=0 -p 8080:80 -v ~/tei-volume:/data --pull always \
  ghcr.io/huggingface/text-embeddings-inference:86-1.9 \
  --model-id BAAI/bge-m3 --max-batch-tokens 16384 --dtype float16

docker run --device nvidia.com/gpu=1 -p 8081:80 -v ~/tei-volume:/data --pull always \
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

from bioasq.common import PROJECT_DATA_DIR
from bioasq.common.io import load_collection, save_json
from bioasq.common.types import Document
from bioasq.common.utils import typer_async
from bioasq.data.database import (
    add_baseline_ids,
    get_pool,
    insert_articles,
)
from bioasq.data.qdrant_store import upload_embeddings

if TYPE_CHECKING:
    from collections.abc import Awaitable


app = typer.Typer()

ENDPOINTS = [
    "http://localhost:8080/embed",
    "http://localhost:8081/embed",
]  # Add more if you have more TEI instances
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
    insert_into_db: Literal["all", "embeddings", "none"] = "none",
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
                case "all":
                    print("Inserting articles and embeddings into database")
                    await insert_articles(documents, embeddings)
                    await add_baseline_ids(2026, ordered_ids)
                    await upload_embeddings((ordered_ids, embeddings))
                case "embeddings":
                    print("Updating embeddings in database")
                    await upload_embeddings((ordered_ids, embeddings))
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

    current_chunk: list[Document] = []
    chunk_count = 0

    for doc in tqdm(load_collection(baseline, None), desc="Total Progress"):
        current_chunk.append(doc)

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


@app.command()
@typer_async
async def generate_missing(
    qdrant_ids_file: Annotated[
        Path, typer.Argument(help="JSON file with PMIDs already in Qdrant (from dump-ids command).")
    ] = PROJECT_DATA_DIR / Path("qdrant_ids.json"),
    chunk_size: Annotated[int, typer.Option(help="Rows per shard.")] = 50_000,
) -> None:
    """
    Generate embeddings for articles missing from Qdrant.

    Load the Qdrant IDs from *qdrant_ids_file* (produced by the dump-ids command),
    diff against Postgres, and embed+upload only the missing ones.
    """
    import orjson

    existing_ids: set[int] = {int(x) for x in orjson.loads(qdrant_ids_file.read_bytes())}
    print(f"Loaded {len(existing_ids):,} existing Qdrant IDs from {qdrant_ids_file}")

    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        cursor = conn.cursor("SELECT pmid, full_text FROM articles ORDER BY pmid")
        current_chunk: list[Document] = []
        cursor_generator = (row async for row in cursor)

        async for row in tqdm(cursor_generator, desc="Total Progress"):
            if int(row["pmid"]) in existing_ids:
                continue
            current_chunk.append(Document(pmid=int(row["pmid"]), full_text=row["full_text"]))
            if len(current_chunk) == chunk_size:
                await process_chunk_async(current_chunk, insert_into_db="embeddings")
                current_chunk = []

        if current_chunk:
            await process_chunk_async(current_chunk, insert_into_db="embeddings")


if __name__ == "__main__":
    app()
