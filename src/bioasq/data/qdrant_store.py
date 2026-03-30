"""Qdrant vector store helpers for the BioASQ pipeline.

Loads pre-exported embedding shards (``embeddings_{start}_{end}.npy`` +
``embeddings_{start}_{end}.ids.json``) and upserts them into a Qdrant
collection, leveraging GPU-accelerated HNSW indexing when available.

Each point payload includes ``first_year``: the earliest year the article
appears in ``ids_per_baseline``.  This enables year-scoped ANN searches
(``first_year <= query_year``) without fetching millions of IDs from Postgres.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated

import asyncpg
import numpy as np
import typer
from tqdm import tqdm

from bioasq.common import PROJECT_DATA_EMBEDDINGS_DIR_EXPORT
from bioasq.common.io import load_json

_QDRANT_URL_ENV = "BIOASQ_QDRANT_URL"
_DATABASE_URL_ENV = "BIOASQ_DATABASE_URL"
_DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
_DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@127.0.0.1:5432/postgres"
_COLLECTION = "articles"
_UPLOAD_BATCH = 50_000


def _qdrant_url() -> str:
    return os.environ.get(_QDRANT_URL_ENV, _DEFAULT_QDRANT_URL)


def _database_url() -> str:
    return os.environ.get(_DATABASE_URL_ENV, _DEFAULT_DATABASE_URL)


def _sort_key(path: Path) -> tuple[int, int]:
    """Sort shard paths by their numeric start/end indices."""
    parts = path.stem.split("_")  # embeddings_{start}_{end}
    return int(parts[1]), int(parts[2])


def _shard_pairs(embeddings_dir: Path) -> list[tuple[Path, Path]]:
    """Return sorted (npy, ids_json) pairs from the export directory."""
    npy_files = sorted(embeddings_dir.glob("embeddings_*.npy"), key=_sort_key)
    pairs: list[tuple[Path, Path]] = []
    for npy in npy_files:
        ids_file = npy.parent / f"{npy.stem}.ids.json"
        if not ids_file.exists():
            raise FileNotFoundError(f"Missing ids file: {ids_file}")
        pairs.append((npy, ids_file))
    return pairs


async def _fetch_first_years(
    conn: asyncpg.Connection, pmids: list[int]
) -> dict[int, int]:
    """Return {pmid: min_year} for the given PMIDs from ids_per_baseline."""
    rows = await conn.fetch(
        "SELECT pmid, MIN(year) AS first_year FROM ids_per_baseline"
        " WHERE pmid = ANY($1) GROUP BY pmid",
        pmids,
    )
    return {int(r["pmid"]): int(r["first_year"]) for r in rows}


async def upload_embeddings(
    embeddings_dir: Path,
    *,
    collection: str = _COLLECTION,
    url: str | None = None,
    db_url: str | None = None,
    batch_size: int = _UPLOAD_BATCH,
) -> None:
    """Upload all embedding shards from *embeddings_dir* into Qdrant.

    Connects to Postgres (via *db_url*) to enrich each point payload with
    ``first_year`` so that year-scoped ANN searches work without ID filters.
    """
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Batch, Distance, VectorParams

    resolved_url = url or _qdrant_url()
    resolved_db_url = db_url or _database_url()

    client = AsyncQdrantClient(url=resolved_url, timeout=60)
    pg_conn: asyncpg.Connection = await asyncpg.connect(dsn=resolved_db_url)

    try:
        pairs = _shard_pairs(embeddings_dir)
        if not pairs:
            raise FileNotFoundError(f"No embedding shards found in {embeddings_dir}")

        probe: np.ndarray = np.load(pairs[0][0], mmap_mode="r")
        vector_size: int = probe.shape[1]
        print(f"Vector size: {vector_size}  |  Shards: {len(pairs)}  |  Collection: {collection}")

        if not await client.collection_exists(collection):
            await client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

        for npy_path, ids_path in tqdm(pairs, desc="Uploading shards", unit="shard"):
            embeddings: np.ndarray = np.load(npy_path).astype(np.float32)
            pmids: list[int] = load_json(ids_path, type_=list[int])

            for start in tqdm(
                range(0, len(pmids), batch_size),
                desc=npy_path.name,
                leave=False,
                unit="batch",
            ):
                batch_pmids = batch_pmids = pmids[start : start + batch_size]
                batch_vecs = embeddings[start : start + batch_size]

                first_years = await _fetch_first_years(pg_conn, batch_pmids)

                await client.upsert(
                    collection_name=collection,
                    points=Batch(
                        ids=batch_pmids,
                        vectors=batch_vecs.tolist(),
                        payloads=[
                            {"pmid": pid, "first_year": first_years.get(pid)}
                            for pid in batch_pmids
                        ],
                    ),
                )
    finally:
        await pg_conn.close()
        await client.close()


def upload_embeddings_command(
    embeddings_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing .npy + .ids.json shard pairs."),
    ] = PROJECT_DATA_EMBEDDINGS_DIR_EXPORT,
    collection: Annotated[
        str,
        typer.Option(help="Qdrant collection name."),
    ] = _COLLECTION,
    url: Annotated[
        str | None,
        typer.Option(
            help=f"Qdrant base URL (defaults to ${_QDRANT_URL_ENV} or {_DEFAULT_QDRANT_URL})."
        ),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option(
            help=f"Postgres DSN (defaults to ${_DATABASE_URL_ENV} or {_DEFAULT_DATABASE_URL})."
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(help="Number of points per upsert batch."),
    ] = _UPLOAD_BATCH,
) -> None:
    """Upload pre-exported embedding shards to Qdrant with first_year payload."""
    asyncio.run(
        upload_embeddings(
            embeddings_dir, collection=collection, url=url, db_url=db_url, batch_size=batch_size
        )
    )
