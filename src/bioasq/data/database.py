"""
Database module for the BioASQ project.

It uses PostgreSQL for the database, with the pgvectorscale extension for the vector store
and the pg_textsearch extension for the search engine.

The articles table has the following columns:
- pmid: the pmid of the article (commonly referred to as PMID)
- title: the title of the article
- abstract: the abstract of the article
- full_text: title + space + coalesced abstract (GENERATED ALWAYS AS ...)
- embedding: the embedding of the article
"""

import asyncio
import os
from typing import Annotated, overload

import asyncpg
import numpy as np
from asyncpg import Pool
from pgvector.asyncpg import register_vector
from tqdm.asyncio import tqdm

from bioasq.common import PROJECT_DATA_BASELINES_DIR
from bioasq.common.aliases import DocumentId
from bioasq.common.io import load_collection_ids
from bioasq.common.types import Document

_POOL: Pool | None = None


def _pool_dsn() -> str:
    """Build DSN from BIOASQ_DATABASE_URL or default local Postgres."""
    return os.environ.get(
        "BIOASQ_DATABASE_URL",
        "postgresql://postgres:postgres@127.0.0.1:5432/postgres",
    )


async def get_pool() -> Pool:
    """Return a shared asyncpg pool with pgvector codecs registered on each connection."""
    global _POOL
    if _POOL is None:
        _POOL = await asyncpg.create_pool(dsn=_pool_dsn())
        async with _POOL.acquire() as conn:
            await register_vector(conn)
    return _POOL


async def close_pool() -> None:
    """Close the shared pool (e.g. on shutdown)."""
    global _POOL
    if _POOL is not None:
        await _POOL.close()
        _POOL = None


async def create_indexes() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        print("Creating embedding index")
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS articles_embedding_idx
            ON articles USING diskann (embedding vector_cosine_ops)
            """
        )

        print("Creating BM25 index")
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS articles_bm25_idx
            ON articles USING bm25(full_text) WITH (text_config = 'english')
            """
        )


def _row_to_document(row: asyncpg.Record) -> Document:
    abstract_val = row["abstract"]
    abstract_str = "" if abstract_val is None else str(abstract_val)
    return Document(
        pmid=str(row["pmid"]),
        title=str(row["title"]),
        abstract=abstract_str,
    )


@overload
async def insert_articles(
    docs: Document, embedding: np.ndarray | None = None, semaphore: asyncio.Semaphore | None = None
) -> None: ...
@overload
async def insert_articles(
    docs: list[Document],
    embeddings: list[np.ndarray] | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> None: ...


async def insert_articles(
    docs: Document | list[Document],
    embeddings: np.ndarray | list[np.ndarray] | None = None,
    semaphore: asyncio.Semaphore | None = None,
    pool: Pool | None = None,
) -> None:
    """Insert a PubMed-style article; duplicate pmid is ignored."""

    if isinstance(docs, Document):
        docs = [docs]
        if embeddings is not None:
            embeddings = [embeddings]

    if embeddings is not None and len(docs) != len(embeddings):
        raise ValueError(
            f"Number of documents and embeddings must be the same. "
            f"Docs: {len(docs)}, Embeddings: {len(embeddings)}"
        )

    async with semaphore or asyncio.Semaphore(1), (pool or await get_pool()).acquire() as conn:
        await register_vector(conn)
        try:
            await conn.copy_records_to_table(
                "articles",
                records=(
                    [
                        (
                            int(doc.pmid),
                            doc.title,
                            doc.abstract or None,
                            np.array(embedding, dtype=np.float32),
                        )
                        for doc, embedding in zip(docs, embeddings, strict=True)
                    ]
                    if embeddings is not None
                    else [(int(doc.pmid), doc.title, doc.abstract or None) for doc in docs]
                ),
                columns=("pmid", "title", "abstract", "embedding"),
            )
        except asyncpg.UniqueViolationError:
            return


async def get_article_by_id(article_id: int) -> Document | None:
    """Fetch one article by PMID."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT pmid, title, abstract FROM articles WHERE pmid = $1",
            article_id,
        )
    return _row_to_document(row) if row else None


async def bm25_search(query: str, topk: int = 10) -> list[Document]:
    """BM25 full-text search over generated full_text (pg_textsearch)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT pmid, title, abstract
            FROM articles
            ORDER BY full_text <@> $1
            LIMIT $2
            """,
            query,
            topk,
        )
    return [_row_to_document(r) for r in rows]


async def update_article_embedding(article_id: int, embedding: np.ndarray) -> None:
    """Set the embedding vector for an article."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE articles SET embedding = $1 WHERE pmid = $2",
            embedding,
            article_id,
        )


async def vss_search(
    embedding: np.ndarray,
    topk: int | None = 10,
    *,
    exclude_id: DocumentId | None = None,
) -> list[tuple[DocumentId, float]]:
    """
    Vector similarity search (cosine via pgvector / diskann).
    Returns (pmid, similarity) where similarity = 1 - cosine_distance.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        exclude_clause = "" if exclude_id is None else f"AND pmid <> {exclude_id}"
        rows = await conn.fetch(
            f"""
                SELECT pmid, 1.0 - (embedding <=> $1) AS distance
                FROM articles
                WHERE embedding IS NOT NULL {exclude_clause}
                ORDER BY embedding <=> $1
                LIMIT $2
            """,
            embedding,
            topk,
            exclude_id,
        )
    return [(DocumentId(r["pmid"]), float(r["distance"])) for r in rows]


async def lookup_by_pmid(article_id: DocumentId) -> list[tuple[DocumentId, float]]:
    """
    Neighbours of an article by VSS: load embedding for article_id, then top-k similar.
    The source article is excluded from results.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT embedding FROM articles WHERE pmid = $1 AND embedding IS NOT NULL",
            article_id,
        )
    if row is None or row["embedding"] is None:
        return []
    emb = row["embedding"]
    return await vss_search(emb, topk=10000, exclude_id=article_id)


async def add_baseline_ids(year: int, pmids: list[DocumentId]) -> None:
    """Add the PMIDs to the ids_per_baseline table."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO ids_per_baseline (year, pmid) VALUES ($1, $2) ON CONFLICT DO NOTHING",
            [(year, int(pmid)) for pmid in pmids],
        )


async def get_pmids_per_baseline(year: int) -> list[DocumentId]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT pmid FROM ids_per_baseline WHERE year = $1",
            year,
        )
    return [DocumentId(row["pmid"]) for row in rows]


async def get_baselines_per_pmid(pmid: DocumentId) -> list[int]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT year FROM ids_per_baseline WHERE pmid = $1",
            pmid,
        )
    return [row["year"] for row in rows]


if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    import typer

    from bioasq.common.io import load_collection
    from bioasq.common.utils import typer_async

    app = typer.Typer()

    @app.command(name="populate")
    @typer_async
    async def populate(
        jsonl_path: Annotated[
            Path, typer.Argument(help="The path to the JSONL file containing the articles.")
        ] = Path("../../../data/baselines/pubmed_baseline_2026.jsonl"),
    ) -> None:
        import time

        print("Loading PubMed articles from JSONL")
        for docs in tqdm(
            load_collection(jsonl_path, chunk_size=10000), desc="Inserting articles", unit="article"
        ):
            start_time = time.perf_counter()
            await insert_articles(docs)
            end_time = time.perf_counter()
            print(f"Inserted article in {end_time - start_time} seconds")

    @app.command(name="populate-with-embeddings")
    @typer_async
    async def populate_with_embeddings(
        jsonl_path: Annotated[
            Path, typer.Argument(help="The path to the JSONL file containing the articles.")
        ] = Path("../../../data/baselines/pubmed_baseline_2026.jsonl"),
        embeddings_dir: Annotated[
            Path, typer.Argument(help="The path to the embeddings file.")
        ] = Path("../../../data/embeddings_2026"),
    ) -> None:

        import numpy as np

        embeddings_files = sorted(
            embeddings_dir.glob("*.npy"),
            key=lambda x: (int(x.stem.split("_")[1]), int(x.stem.split("_")[2])),
        )
        semaphore = asyncio.Semaphore(8)
        pool = await get_pool()
        tasks = [
            asyncio.create_task(insert_articles(docs, embeddings, semaphore=semaphore, pool=pool))
            for docs, embeddings in tqdm(
                zip(
                    load_collection(jsonl_path, chunk_size=50000),
                    (np.load(x) for x in embeddings_files),
                    strict=True,
                ),
                desc="Loading collection",
                unit="article",
                total=len(embeddings_files),
                unit_scale=50000,
            )
        ]
        await tqdm.gather(
            *tasks,
            desc="Inserting articles",
            unit="article",
            total=len(embeddings_files),
            unit_scale=50000,
        )

    @app.command(name="populate-baseline-ids")
    @typer_async
    async def populate_baseline_ids(
        baselines_dir: Annotated[
            Path,
            typer.Argument(
                help="The path to the directory containing the baseline files.",
            ),
        ] = PROJECT_DATA_BASELINES_DIR,
    ) -> None:
        baselines: list[Path] = list(baselines_dir.glob("pubmed_baseline_20[0-9][0-9].jsonl"))

        for baseline in tqdm(baselines, desc="Processing baselines", unit="baseline"):
            year = int(baseline.name.removeprefix("pubmed_baseline_").removesuffix(".jsonl"))
            pmids = list(load_collection_ids(baseline))
            await add_baseline_ids(year, pmids)

    @app.command(name="create-indexes")
    @typer_async
    async def indexes() -> None:
        await create_indexes()

    asyncio.run(app())
