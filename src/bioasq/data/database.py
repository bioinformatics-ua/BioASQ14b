"""
Database module for the BioASQ project.

It uses PostgreSQL for the database, with the pgvectorscale extension for the vector store
and the pg_textsearch extension for the search engine.

The articles table has the following columns:
- pmid: the pmid of the article (commonly referred to as PMID)
- full_text: the full text of the article
- embedding: the embedding of the article
"""

from pathlib import Path

import asyncio
import os
from typing import Annotated, overload

import asyncpg
import numpy as np
from asyncpg import Pool
from pgvector.asyncpg import register_vector
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, HasIdCondition, Range
from tqdm.asyncio import tqdm

from bioasq.common import PROJECT_DATA_BASELINES_DIR, PROJECT_DATA_EMBEDDINGS_DIR, PROJECT_DATA_EMBEDDINGS_DIR_EXPORT
from bioasq.common.aliases import DocumentId
from bioasq.common.io import load_collection_ids, save_json
from bioasq.common.types import Document, DocumentWithScore

_POOL: Pool | None = None
_QDRANT_CLIENT: AsyncQdrantClient | None = None
_QDRANT_COLLECTION = "articles"


def _qdrant_url() -> str:
    return os.environ.get("BIOASQ_QDRANT_URL", "http://127.0.0.1:6333")


async def get_qdrant_client() -> AsyncQdrantClient:
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        _QDRANT_CLIENT = AsyncQdrantClient(url=_qdrant_url(), timeout=60)
    return _QDRANT_CLIENT


async def close_qdrant_client() -> None:
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is not None:
        await _QDRANT_CLIENT.close()
        _QDRANT_CLIENT = None


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
        print("Setting up configs")

        await conn.execute("SET maintenance_work_mem = '512MB'")
        await conn.execute("SET diskann.min_vectors_for_parallel_build = 10000")
        await conn.execute("SET diskann.force_parallel_workers = 16")
        await conn.execute("SET diskann.parallel_flush_interval = 0.05;")

        print("Creating embedding index")
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS articles_embedding_idx
            ON articles USING diskann (embedding vector_cosine_ops)
            """
        )

        # print("Creating BM25 index")
        # await conn.execute(
        #     """
        #     CREATE INDEX IF NOT EXISTS articles_bm25_idx ON articles USING bm25(full_text) WITH (text_config='english', k1=0.4, b=0.3);
        #     """
        # )


def _row_to_document(row: asyncpg.Record) -> Document:
    return Document(
        pmid=str(row["pmid"]),
        full_text=str(row["full_text"]),
    )


def _row_to_bm25_result(row: asyncpg.Record) -> DocumentWithScore:
    return DocumentWithScore(
        pmid=str(row["pmid"]),
        full_text=str(row["full_text"]),
        score=-float(row["score"]),
    )


def _row_to_semantic_result(row: asyncpg.Record) -> DocumentWithScore:
    return DocumentWithScore(
        pmid=str(row["pmid"]),
        full_text=str(row["full_text"]),
        score=float(row["score"]),
    )


def _exclude_ids_array(exclude_ids: set[DocumentId] | None) -> list[int] | None:
    if not exclude_ids:
        return None
    return [int(x) for x in exclude_ids]


async def insert_articles(
    docs: Document | list[Document],
    semaphore: asyncio.Semaphore | None = None,
    pool: Pool | None = None,
) -> None:
    """Insert PubMed-style articles in bulk; duplicate pmids are silently ignored."""

    if isinstance(docs, Document):
        docs = [docs]
    if not docs:
        return

    records = [(int(doc.pmid), doc.full_text) for doc in docs]

    async with semaphore or asyncio.Semaphore(1), (pool or await get_pool()).acquire() as conn:
        await conn.execute("SET temp_buffers = '256MB'")
        async with conn.transaction():
            await conn.execute(
                "CREATE TEMP TABLE _tmp_articles (pmid bigint, full_text text) ON COMMIT DROP"
            )
            await conn.copy_records_to_table(
                "_tmp_articles",
                records=records,
                columns=("pmid", "full_text"),
            )
            await conn.execute(
                "INSERT INTO articles (pmid, full_text) "
                "SELECT pmid, full_text FROM _tmp_articles "
                "ON CONFLICT (pmid) DO NOTHING"
            )


async def get_article_by_id(article_id: int) -> Document | None:
    """Fetch one article by PMID."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT pmid, full_text FROM articles WHERE pmid = $1",
            article_id,
        )
    return _row_to_document(row) if row else None


async def get_if_articles_exist(pmids: list[DocumentId]) -> set[DocumentId]:
    """Check if the articles exist in the database."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT pmid FROM articles WHERE pmid = ANY($1)",
            pmids,
        )
    return {str(row["pmid"]) for row in rows}


async def bm25_search(
    query: str,
    topk: int = 10,
    *,
    year: int | None = None,
    exclude_ids: set[DocumentId] | None = None,
) -> list[DocumentWithScore]:
    """BM25 full-text search over generated full_text (pg_textsearch).

    When *year* is provided the search is restricted to articles present in
    that year's PubMed baseline (via ``ids_per_baseline``).
    """
    pool = await get_pool()
    excl = _exclude_ids_array(exclude_ids)
    async with pool.acquire() as conn:
        if year is not None:
            rows = await conn.fetch(
                """
                SELECT a.pmid, a.full_text, a.full_text <@> $1 AS score
                FROM articles a
                JOIN ids_per_baseline b ON a.pmid = b.pmid AND b.year = $4
                WHERE ($3::bigint[] IS NULL OR cardinality($3::bigint[]) = 0
                       OR a.pmid != ALL($3::bigint[]))
                ORDER BY score
                LIMIT $2
                """,
                query,
                topk,
                excl,
                year,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT pmid, full_text, full_text <@> $1 AS score
                FROM articles
                WHERE ($3::bigint[] IS NULL OR cardinality($3::bigint[]) = 0
                       OR pmid != ALL($3::bigint[]))
                ORDER BY score
                LIMIT $2
                """,
                query,
                topk,
                excl,
            )

    return [_row_to_bm25_result(r) for r in rows]


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
    year: int | None = None,
    exclude_ids: set[DocumentId] | None = None,
) -> list[tuple[DocumentId, float]]:
    """Vector similarity search (cosine via Qdrant). Returns (pmid, similarity).

    When *year* is provided, only articles whose ``first_year`` payload field
    is <= *year* are considered (i.e. articles present in that year's baseline).
    """
    client = await get_qdrant_client()
    lim = topk if topk is not None else 10

    must: list[FieldCondition] = []
    must_not = []
    if year is not None:
        must.append(FieldCondition(key="first_year", range=Range(lte=year)))
    if exclude_ids:
        must_not.append(HasIdCondition(has_id=[int(x) for x in exclude_ids]))

    filter_ = Filter(must=must or None, must_not=must_not or None) if (must or must_not) else None

    hits = await client.search(
        collection_name=_QDRANT_COLLECTION,
        query_vector=embedding.astype(np.float32).tolist(),
        limit=lim,
        query_filter=filter_,
        with_payload=False,
        with_vectors=False,
    )
    return [(DocumentId(str(hit.id)), float(hit.score)) for hit in hits]


async def semantic_search(
    query_embedding: np.ndarray,
    topk: int = 10,
    *,
    year: int | None = None,
    exclude_ids: set[DocumentId] | None = None,
) -> list[DocumentWithScore]:
    """Dense retrieval via Qdrant, hydrates full_text from Postgres."""
    ranked = await vss_search(query_embedding, topk, year=year, exclude_ids=exclude_ids)
    if not ranked:
        return []
    pmid_ints = [int(pmid) for pmid, _ in ranked]
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT pmid, full_text FROM articles WHERE pmid = ANY($1)",
            pmid_ints,
        )
    texts: dict[str, str] = {str(r["pmid"]): r["full_text"] for r in rows}
    return [
        DocumentWithScore(pmid=pmid, full_text=texts[pmid], score=score)
        for pmid, score in ranked
        if pmid in texts
    ]


async def lookup_by_pmid(article_id: DocumentId) -> list[tuple[DocumentId, float]]:
    """
    Neighbours of an article by VSS: fetch the stored vector from Qdrant, then search.
    The source article is excluded from results.
    """
    client = await get_qdrant_client()
    points = await client.retrieve(
        collection_name=_QDRANT_COLLECTION,
        ids=[int(article_id)],
        with_vectors=True,
    )
    if not points:
        return []
    embedding = np.array(points[0].vector, dtype=np.float32)
    return await vss_search(embedding, topk=10000, exclude_ids={article_id})


"""
SELECT articles.* FROM articles
JOIN ids_per_baseline ON articles.pmid = ids_per_baseline.pmid AND ids_per_baseline.year = 2026
ORDER BY embedding <@> $1::vector(1024)
LIMIT 100;
"""


async def add_baseline_ids(year: int, pmids: list[DocumentId] | list[int]) -> None:
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


async def _export_embeddings(output_dir: Path) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM articles")
        for i in tqdm(range(0, count, 50000), desc="Exporting embeddings", unit="chunk"):
            rows = await conn.fetch(
                "SELECT pmid, embedding FROM articles LIMIT 50000 OFFSET $1",
                i,
            )
            np.save(
                output_dir / f"embeddings_{i}_{i + 50000}.npy",
                np.array([row["embedding"] for row in rows]),
            )
            save_json(
                [row["pmid"] for row in rows],
                output_dir / f"embeddings_{i}_{i + 50000}.ids.json",
            )


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
            Path,
            typer.Argument(help="The path to the JSONL file containing the articles.", exists=True),
        ] = PROJECT_DATA_BASELINES_DIR / "pubmed_baseline_2026.jsonl",
        year: Annotated[int, typer.Option(help="The year of the baseline.")] = 2026,
    ) -> None:
        print("Loading PubMed articles from JSONL")
        for docs in tqdm(
            load_collection(jsonl_path, chunk_size=50000), desc="Inserting articles", unit="article"
        ):
            await insert_articles(docs)
            await add_baseline_ids(year, [doc.pmid for doc in docs])

    @app.command(name="populate-with-embeddings")
    @typer_async
    async def populate_with_embeddings(
        jsonl_path: Annotated[
            Path,
            typer.Argument(help="The path to the JSONL file containing the articles.", exists=True),
        ] = PROJECT_DATA_BASELINES_DIR / "pubmed_baseline_2026.jsonl",
        embeddings_dir: Annotated[
            Path, typer.Argument(help="The path to the embeddings file.")
        ] = PROJECT_DATA_EMBEDDINGS_DIR,
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


    @app.command(name="export-embeddings")
    @typer_async
    async def export_embeddings(
        output_dir: Annotated[
            Path, typer.Argument(help="The path to the directory to export the embeddings to.")
        ] = PROJECT_DATA_EMBEDDINGS_DIR_EXPORT,
    ) -> None:
        await _export_embeddings(output_dir)

    asyncio.run(app())
