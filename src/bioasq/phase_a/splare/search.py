"""Qdrant sparse-vector search for SPLARE representations.

Mirrors the interface of :func:`bioasq.data.database.semantic_search` but
queries a Qdrant collection using sparse (inverted-index) vectors instead of
dense HNSW vectors.

The collection stores one named sparse vector per point (``"splare"``), plus
the point ID equals the article PMID.  Full text is hydrated from Postgres.
"""

from __future__ import annotations

import os

from qdrant_client.models import (
    Filter,
    HasIdCondition,
    SparseVector,
)

from bioasq.common.aliases import DocumentId
from bioasq.common.types import DocumentWithScore
from bioasq.data.database import get_pool, get_qdrant_client

_SPLARE_COLLECTION = os.environ.get("BIOASQ_SPLARE_COLLECTION", "articles_splare")
_SPARSE_VECTOR_NAME = "splare"


async def splare_search(
    query_indices: list[int],
    query_values: list[float],
    topk: int = 10,
    *,
    year: int | None = None,
    exclude_ids: set[DocumentId] | None = None,
) -> list[DocumentWithScore]:
    """Sparse retrieval via Qdrant, hydrates full_text from Postgres.

    When *year* is provided, fetches extra candidates and then filters by
    year using ``ids_per_baseline`` in Postgres.
    """
    fetch_k = max(topk * 2, 200) if year is not None else topk

    client = await get_qdrant_client()

    must_not = []
    if exclude_ids:
        must_not.append(HasIdCondition(has_id=[int(x) for x in exclude_ids]))
    filter_ = Filter(must_not=must_not) if must_not else None

    hits = await client.query_points(
        collection_name=_SPLARE_COLLECTION,
        query=SparseVector(indices=query_indices, values=query_values),
        using=_SPARSE_VECTOR_NAME,
        limit=fetch_k,
        query_filter=filter_,
        with_payload=False,
        with_vectors=False,
    )

    if not hits.points:
        return []

    ranked = [(str(point.id), float(point.score)) for point in hits.points]
    pmid_ints = [int(pmid) for pmid, _ in ranked]

    pool = await get_pool()
    async with pool.acquire() as conn:
        if year is not None:
            rows = await conn.fetch(
                """
                SELECT a.pmid, a.full_text
                FROM articles a
                JOIN ids_per_baseline b ON a.pmid = b.pmid AND b.year = $1
                WHERE a.pmid = ANY($2)
                """,
                year,
                pmid_ints,
            )
        else:
            rows = await conn.fetch(
                "SELECT pmid, full_text FROM articles WHERE pmid = ANY($1)",
                pmid_ints,
            )

    texts: dict[str, str] = {str(r["pmid"]): r["full_text"] for r in rows}
    return [
        DocumentWithScore(pmid=pmid, full_text=texts[pmid], score=score)
        for pmid, score in ranked
        if pmid in texts
    ][:topk]


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


async def create_splare_collection(
    vector_size: int = 131072,
    *,
    collection: str = _SPLARE_COLLECTION,
) -> None:
    """Create a Qdrant collection with a named sparse vector field."""
    from qdrant_client.models import SparseIndexParams, SparseVectorParams

    client = await get_qdrant_client()

    if await client.collection_exists(collection):
        print(f"Collection '{collection}' already exists, skipping creation.")
        return

    await client.create_collection(
        collection_name=collection,
        vectors_config={},
        sparse_vectors_config={
            _SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    print(f"Created sparse collection '{collection}'.")


async def upload_splare_vectors(
    ids: list[str],
    indices: list[list[int]],
    values: list[list[float]],
    *,
    collection: str = _SPLARE_COLLECTION,
    batch_size: int = 1_000,
) -> None:
    """Upload sparse vectors to the SPLARE Qdrant collection.

    Each point has ID = PMID (int), with a single named sparse vector.
    """
    from qdrant_client.models import PointStruct

    client = await get_qdrant_client()

    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        batch_indices = indices[start : start + batch_size]
        batch_values = values[start : start + batch_size]

        points = [
            PointStruct(
                id=int(pmid),
                vector={
                    _SPARSE_VECTOR_NAME: SparseVector(
                        indices=idx,
                        values=val,
                    )
                },
                payload={},
            )
            for pmid, idx, val in zip(batch_ids, batch_indices, batch_values, strict=True)
        ]
        await client.upsert(collection_name=collection, points=points)

    print(f"Uploaded {len(ids)} sparse vectors to '{collection}'.")
