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
from typing import Annotated, cast

import asyncpg
import numpy as np
import typer
from tqdm import tqdm

from bioasq.common import PROJECT_DATA_DIR, PROJECT_DATA_EMBEDDINGS_DIR_EXPORT
from bioasq.common.io import load_json

_QDRANT_URL_ENV = "BIOASQ_QDRANT_URL"
_DATABASE_URL_ENV = "BIOASQ_DATABASE_URL"
_DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
_DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@127.0.0.1:5432/postgres"
_COLLECTION = "articles"
_UPLOAD_BATCH = 50_000
# Qdrant REST rejects bodies larger than ~32 MiB; JSON float lists blow up fast.
_MAX_REST_JSON_BYTES = 28 * 1024 * 1024
_GRPC_PORT_ENV = "BIOASQ_QDRANT_GRPC_PORT"
_PREFER_GRPC_ENV = "BIOASQ_QDRANT_PREFER_GRPC"


def _qdrant_url() -> str:
    return os.environ.get(_QDRANT_URL_ENV, _DEFAULT_QDRANT_URL)


def _database_url() -> str:
    return os.environ.get(_DATABASE_URL_ENV, _DEFAULT_DATABASE_URL)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _grpc_port() -> int:
    return int(os.environ.get(_GRPC_PORT_ENV, "6334"))


def _max_points_per_rest_upload(vector_size: int) -> int:
    """Upper bound on points per upsert so JSON stays under Qdrant's REST limit."""
    bytes_per_point = vector_size * 14 + 150
    return max(1, _MAX_REST_JSON_BYTES // bytes_per_point)


def _points_per_upsert(batch_size: int, vector_size: int, *, prefer_grpc: bool) -> int:
    if prefer_grpc:
        return batch_size
    return min(batch_size, _max_points_per_rest_upload(vector_size))


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


async def _fetch_first_years(conn: asyncpg.Connection, pmids: list[int]) -> dict[int, int]:
    """Return {pmid: min_year} for the given PMIDs from ids_per_baseline."""
    rows = await conn.fetch(
        "SELECT pmid, MIN(year) AS first_year FROM ids_per_baseline"
        " WHERE pmid = ANY($1) GROUP BY pmid",
        pmids,
    )
    return {int(r["pmid"]): int(r["first_year"]) for r in rows}


async def upload_embeddings(
    embeddings: Path
    | tuple[list[int], np.ndarray],  # dir |  (ids, vectors) pair for single-batch upload
    *,
    collection: str = _COLLECTION,
    url: str | None = None,
    db_url: str | None = None,
    batch_size: int = _UPLOAD_BATCH,
    prefer_grpc: bool | None = None,
    grpc_port: int | None = None,
    timeout: int = 300,
) -> None:
    """Upload all embedding shards from *embeddings_dir* into Qdrant.

    Connects to Postgres (via *db_url*) to enrich each point payload with
    ``first_year`` so that year-scoped ANN searches work without ID filters.

    Uses gRPC by default (same host as REST, port 6334) so large batches are
    sent as protobuf instead of huge JSON (REST limit ~32 MiB). Set
    ``prefer_grpc=False`` to force HTTP; uploads are then split automatically
    to stay under that limit.
    """
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Batch, Distance, VectorParams

    resolved_url = url or _qdrant_url()
    resolved_db_url = db_url or _database_url()
    use_grpc = _env_bool(_PREFER_GRPC_ENV, True) if prefer_grpc is None else prefer_grpc
    gp = grpc_port if grpc_port is not None else _grpc_port()

    client = AsyncQdrantClient(
        url=resolved_url,
        grpc_port=gp,
        prefer_grpc=use_grpc,
        timeout=timeout,
    )
    pg_conn: asyncpg.Connection = await asyncpg.connect(dsn=resolved_db_url)

    try:
        if isinstance(embeddings, tuple):
            pmids, vecs = embeddings
            vecs = vecs.astype(np.float32)
            vector_size = vecs.shape[1]
            per_upsert = _points_per_upsert(batch_size, vector_size, prefer_grpc=use_grpc)
            print(
                f"Uploading {len(pmids)} embeddings to collection '{collection}' "
                f"via {'gRPC' if use_grpc else 'REST'}",
            )
            if not await client.collection_exists(collection):
                await client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
            for start in tqdm(range(0, len(pmids), per_upsert), desc="Uploading", unit="batch"):
                chunk_pmids = pmids[start : start + per_upsert]
                chunk_vecs = vecs[start : start + per_upsert]
                first_years = await _fetch_first_years(pg_conn, chunk_pmids)
                await client.upsert(
                    collection_name=collection,
                    points=Batch(
                        ids=chunk_pmids,  # pyright: ignore[reportArgumentType]
                        vectors=chunk_vecs.tolist(),
                        payloads=[
                            {"pmid": pid, "first_year": first_years.get(pid)} for pid in chunk_pmids
                        ],
                    ),
                )
        else:
            pairs = _shard_pairs(embeddings)
            if not pairs:
                raise FileNotFoundError(f"No embedding shards found in {embeddings}")
            probe: np.ndarray = np.load(pairs[0][0], mmap_mode="r")
            vector_size = probe.shape[1]
            per_upsert = _points_per_upsert(batch_size, vector_size, prefer_grpc=use_grpc)
            print(
                f"Vector size: {vector_size}  |  Shards: {len(pairs)}"
                f"  |  Collection: {collection}\n"
                f"prefer_grpc={use_grpc}  grpc_port={gp}  points_per_upsert={per_upsert}  "
                f"(logical batch_size={batch_size})",
            )
            if not await client.collection_exists(collection):
                await client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
            for npy, ids in tqdm(pairs, desc="Uploading shards", unit="shard"):
                shard_vecs: np.ndarray = np.load(npy).astype(np.float32)
                shard_pmids = cast("list[int]", load_json(ids))

                for start in tqdm(
                    range(0, len(shard_pmids), batch_size),
                    desc=npy.name,
                    leave=False,
                    unit="batch",
                ):
                    batch_pmids = shard_pmids[start : start + batch_size]
                    batch_vecs = shard_vecs[start : start + batch_size]
                    first_years = await _fetch_first_years(pg_conn, batch_pmids)

                    for sub in range(0, len(batch_pmids), per_upsert):
                        chunk_pmids = batch_pmids[sub : sub + per_upsert]
                        chunk_vecs = batch_vecs[sub : sub + per_upsert]
                        await client.upsert(
                            collection_name=collection,
                            points=Batch(
                                ids=chunk_pmids,  # pyright: ignore[reportArgumentType]
                                vectors=chunk_vecs.tolist(),
                                payloads=[
                                    {"pmid": pid, "first_year": first_years.get(pid)}
                                    for pid in chunk_pmids
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
        typer.Option(help="Target points per upsert; REST is auto-chunked under ~32 MiB."),
    ] = _UPLOAD_BATCH,
    prefer_grpc: Annotated[
        bool,
        typer.Option(
            "--grpc/--no-grpc",
            help="Use gRPC for upserts (recommended; avoids REST ~32 MiB JSON limit).",
        ),
    ] = True,
    grpc_port: Annotated[
        int,
        typer.Option(help=f"gRPC port (default: env {_GRPC_PORT_ENV} or 6334)."),
    ] = 6334,
    timeout: Annotated[
        int,
        typer.Option(help="Client timeout seconds (REST and gRPC)."),
    ] = 300,
) -> None:
    """Upload pre-exported embedding shards to Qdrant with first_year payload."""
    asyncio.run(
        upload_embeddings(
            embeddings_dir,
            collection=collection,
            url=url,
            db_url=db_url,
            batch_size=batch_size,
            prefer_grpc=prefer_grpc,
            grpc_port=grpc_port,
            timeout=timeout,
        )
    )


async def dump_qdrant_ids(
    output: Path,
    *,
    collection: str = _COLLECTION,
    url: str | None = None,
    batch_size: int = 10_000,
) -> None:
    """Scroll all point IDs from Qdrant and write them to *output* as a JSON array."""
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(url=url or _qdrant_url(), timeout=300)
    try:
        total = (await client.get_collection(collection)).points_count
        ids: list[int] = []
        offset = None
        with tqdm(total=total, desc="Scrolling Qdrant IDs", unit="pts") as bar:
            while True:
                result, offset = await client.scroll(
                    collection_name=collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                ids.extend(int(p.id) for p in result)
                bar.update(len(result))
                if offset is None:
                    break
        print(f"Writing {len(ids):,} IDs to {output}")
        from bioasq.common.io import save_json

        save_json(ids, output)
    finally:
        await client.close()


def dump_ids_command(
    output: Annotated[Path, typer.Argument(help="Output .json file path.")] = Path(
        PROJECT_DATA_DIR / "qdrant_ids.json"
    ),
    collection: Annotated[str, typer.Option()] = _COLLECTION,
    url: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Dump all Qdrant point IDs to a JSON file for diffing against Postgres."""
    import asyncio

    asyncio.run(dump_qdrant_ids(output, collection=collection, url=url))


if __name__ == "__main__":
    dump_ids_command()
    # upload_embeddings_command()
