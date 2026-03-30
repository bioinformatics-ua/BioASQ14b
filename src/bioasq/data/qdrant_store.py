"""Qdrant vector store helpers for the BioASQ pipeline.

Loads pre-exported embedding shards (``embeddings_{start}_{end}.npy`` +
``embeddings_{start}_{end}.ids.json``) and upserts them into a Qdrant
collection, leveraging GPU-accelerated HNSW indexing when available.
"""

from __future__ import annotations

import os
from pathlib import Path
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


def _qdrant_url() -> str:
    return os.environ.get(_QDRANT_URL_ENV, _DEFAULT_URL)


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
) -> None:
    """Upload all embedding shards from *embeddings_dir* into Qdrant."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Batch, Distance, VectorParams

    resolved_url = url or _qdrant_url()
    client = QdrantClient(url=resolved_url, timeout=60)

    pairs = _shard_pairs(embeddings_dir)
    if not pairs:
        raise FileNotFoundError(f"No embedding shards found in {embeddings_dir}")

    # Infer vector dimensionality from the first shard without loading it fully
    probe: np.ndarray = np.load(pairs[0][0], mmap_mode="r")
    vector_size: int = probe.shape[1]
    print(f"Vector size: {vector_size}  |  Shards: {len(pairs)}  |  Collection: {collection}")

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
            client.upsert(
                collection_name=collection,
                points=Batch(
                    ids=batch_pmids,
                    vectors=batch_vecs.tolist(),
                    payloads=[{"pmid": pid} for pid in batch_pmids],
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
        typer.Option(help="Number of points per upsert batch."),
    ] = _UPLOAD_BATCH,
) -> None:
    """Upload pre-exported embedding shards to Qdrant."""
    upload_embeddings(embeddings_dir, collection=collection, url=url, batch_size=batch_size)
