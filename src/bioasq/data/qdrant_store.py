"""Qdrant vector store helpers for the BioASQ pipeline.

Loads pre-exported embedding shards (``embeddings_{start}_{end}.npy`` +
``embeddings_{start}_{end}.ids.json``) and upserts them into a Qdrant
collection, leveraging GPU-accelerated HNSW indexing when available.
"""

from __future__ import annotations

import os
from pathlib import Path  # noqa: TC003
from typing import Annotated

import numpy as np
import typer
from tqdm import tqdm

from bioasq.common import PROJECT_DATA_EMBEDDINGS_DIR_EXPORT
from bioasq.common.io import load_json

_QDRANT_URL_ENV = "BIOASQ_QDRANT_URL"
_DEFAULT_URL = "http://127.0.0.1:6333"
_COLLECTION = "articles"
_UPLOAD_BATCH = 50_000
# Qdrant REST rejects bodies larger than ~32 MiB; JSON float lists blow up fast.
_MAX_REST_JSON_BYTES = 28 * 1024 * 1024
_GRPC_PORT_ENV = "BIOASQ_QDRANT_GRPC_PORT"
_PREFER_GRPC_ENV = "BIOASQ_QDRANT_PREFER_GRPC"


def _qdrant_url() -> str:
    return os.environ.get(_QDRANT_URL_ENV, _DEFAULT_URL)


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


def upload_embeddings(
    embeddings_dir: Path,
    *,
    collection: str = _COLLECTION,
    url: str | None = None,
    batch_size: int = _UPLOAD_BATCH,
    prefer_grpc: bool | None = None,
    grpc_port: int | None = None,
    timeout: int = 300,
) -> None:
    """Upload all embedding shards from *embeddings_dir* into Qdrant.

    Uses gRPC by default (same host as REST, port 6334) so large batches are
    sent as protobuf instead of huge JSON (REST limit ~32 MiB). Set
    ``prefer_grpc=False`` to force HTTP; uploads are then split automatically
    to stay under that limit.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Batch, Distance, VectorParams

    resolved_url = url or _qdrant_url()
    use_grpc = _env_bool(_PREFER_GRPC_ENV, True) if prefer_grpc is None else prefer_grpc
    gp = grpc_port if grpc_port is not None else _grpc_port()

    client = QdrantClient(
        url=resolved_url,
        grpc_port=gp,
        prefer_grpc=use_grpc,
        timeout=timeout,
    )

    pairs = _shard_pairs(embeddings_dir)
    if not pairs:
        raise FileNotFoundError(f"No embedding shards found in {embeddings_dir}")

    # Infer vector dimensionality from the first shard without loading it fully
    probe: np.ndarray = np.load(pairs[0][0], mmap_mode="r")
    vector_size: int = probe.shape[1]
    per_upsert = _points_per_upsert(batch_size, vector_size, prefer_grpc=use_grpc)
    print(
        f"Vector size: {vector_size}  |  Shards: {len(pairs)}  |  Collection: {collection}\n"
        f"prefer_grpc={use_grpc}  grpc_port={gp}  points_per_upsert={per_upsert}  "
        f"(logical batch_size={batch_size})",
    )

    if not client.collection_exists(collection):
        client.create_collection(
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
            batch_pmids = pmids[start : start + batch_size]
            batch_vecs = embeddings[start : start + batch_size]
            for sub in range(0, len(batch_pmids), per_upsert):
                chunk_pmids = batch_pmids[sub : sub + per_upsert]
                chunk_vecs = batch_vecs[sub : sub + per_upsert]
                client.upsert(
                    collection_name=collection,
                    points=Batch(
                        ids=chunk_pmids,
                        vectors=chunk_vecs.tolist(),
                        payloads=[{"pmid": pid} for pid in chunk_pmids],
                    ),
                )


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
            help=f"Qdrant base URL (defaults to ${_QDRANT_URL_ENV} or {_DEFAULT_URL})."
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(help="Target points per upsert; REST is auto-chunked under ~32 MiB."),
    ] = _UPLOAD_BATCH,
    prefer_grpc: Annotated[
        bool,
        typer.Option(
            True,
            "--grpc/--no-grpc",
            help="Use gRPC for upserts (recommended; avoids REST ~32 MiB JSON limit).",
        ),
    ] = True,
    grpc_port: Annotated[
        int,
        typer.Option(6334, help=f"gRPC port (default: env {_GRPC_PORT_ENV} or 6334)."),
    ] = 6334,
    timeout: Annotated[
        int,
        typer.Option(help="Client timeout seconds (REST and gRPC)."),
    ] = 300,
) -> None:
    """Upload pre-exported embedding shards to Qdrant."""
    upload_embeddings(
        embeddings_dir,
        collection=collection,
        url=url,
        batch_size=batch_size,
        prefer_grpc=prefer_grpc,
        grpc_port=grpc_port,
        timeout=timeout,
    )


if __name__ == "__main__":
    upload_embeddings_command()
