"""Upload pre-computed SPLARE sparse vector shards to Qdrant.

Reads shards produced by :mod:`bioasq.phase_a.splare.encoder` and inserts
them into a Qdrant sparse-vector collection.

Usage::

    python -m bioasq.phase_a.splare.upload --shards-dir data/splare/export
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from bioasq.common.utils import typer_async
from bioasq.phase_a.splare.encoder import _load_shard, list_shards
from bioasq.phase_a.splare.search import create_splare_collection, upload_splare_vectors

app = typer.Typer(name="splare-upload", help="Upload SPLARE vectors to Qdrant.")


@app.command()
@typer_async
async def upload_command(
    shards_dir: Annotated[str, typer.Option(help="Directory with SPLARE shards.")],
    collection: Annotated[str, typer.Option(help="Qdrant collection name.")] = "articles_splare",
    batch_size: Annotated[int, typer.Option(help="Points per upsert batch.")] = 1_000,
) -> None:
    """Upload all SPLARE shards from disk into a Qdrant sparse-vector collection."""
    shards_path = Path(shards_dir)
    prefixes = list_shards(shards_path)
    if not prefixes:
        print(f"No SPLARE shards found in {shards_path}")
        raise typer.Exit(1)

    print(f"Found {len(prefixes)} shards in {shards_path}")

    await create_splare_collection(collection=collection)

    total_uploaded = 0
    for prefix in prefixes:
        ids, indices, values = _load_shard(shards_path, prefix)
        await upload_splare_vectors(
            ids,
            indices,
            values,
            collection=collection,
            batch_size=batch_size,
        )
        total_uploaded += len(ids)
        print(f"  {prefix}: {len(ids)} vectors (total: {total_uploaded})")

    print(f"Upload complete. {total_uploaded} vectors in collection '{collection}'.")


if __name__ == "__main__":
    app()
