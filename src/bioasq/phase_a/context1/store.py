"""Article-level retrieval primitives for Context-1."""

import asyncio
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, HasIdCondition

from bioasq.data.database import get_article_by_id, get_pool
from bioasq.phase_a.context1.types import CorpusDocument
from bioasq.phase_a.retrieval.query_encoder import embed_queries_tei

_ARTICLES_BM25_INDEX = "articles_bm25_idx"
_DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"


def _qdrant_url() -> str:
    return os.environ.get("BIOASQ_QDRANT_URL", _DEFAULT_QDRANT_URL)


def fuse_rrf_documents(
    named_lists: Sequence[tuple[str, Sequence[CorpusDocument]]],
    *,
    rrf_k: int = 60,
) -> list[CorpusDocument]:
    """Fuse ranked PMID lists with reciprocal rank fusion."""

    by_pmid: dict[str, CorpusDocument] = {}
    fused_scores: dict[str, float] = {}
    for _, documents in named_lists:
        for rank, document in enumerate(documents, start=1):
            by_pmid.setdefault(document.pmid, document)
            fused_scores[document.pmid] = fused_scores.get(document.pmid, 0.0) + (
                1.0 / (rrf_k + rank)
            )

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [replace(by_pmid[pmid], score=score) for pmid, score in ranked]


class Context1CorpusStore:
    """PMID-level corpus store backed by existing Postgres and Qdrant data."""

    def __init__(
        self,
        *,
        token_counter: Callable[[str], int],
        text_truncator: Callable[[str, int], str],
        tei_embed_url: str | None = None,
        collection_name: str = "articles",
        qdrant_url: str | None = None,
        qdrant_grpc_port: int = 6334,
        qdrant_prefer_grpc: bool = True,
    ) -> None:
        self.token_counter = token_counter
        self.text_truncator = text_truncator
        self.tei_embed_url = tei_embed_url
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url or _qdrant_url()
        self.qdrant_grpc_port = qdrant_grpc_port
        self.qdrant_prefer_grpc = qdrant_prefer_grpc
        self._qdrant_client: AsyncQdrantClient | None = None

    async def get_qdrant_client(self) -> AsyncQdrantClient:
        """Get or create the Qdrant client for the article collection."""

        if self._qdrant_client is None:
            self._qdrant_client = AsyncQdrantClient(
                url=self.qdrant_url,
                grpc_port=self.qdrant_grpc_port,
                prefer_grpc=self.qdrant_prefer_grpc,
                timeout=300,
            )
        return self._qdrant_client

    async def close(self) -> None:
        """Close the Qdrant client if it was opened."""

        if self._qdrant_client is not None:
            await self._qdrant_client.close()
            self._qdrant_client = None

    async def prepare_existing_corpus(
        self,
        *,
        year: int | None = None,
        ensure_bm25: bool = False,
    ) -> dict[str, int | bool | None]:
        """Validate the existing article corpus and optionally ensure BM25 exists."""

        pool = await get_pool()
        async with pool.acquire() as conn:
            if ensure_bm25:
                await conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {_ARTICLES_BM25_INDEX}
                    ON articles USING bm25 (full_text)
                    WITH (text_config = 'english', k1 = 0.4, b = 0.3)
                    """
                )

            if year is None:
                article_count = await conn.fetchval("SELECT COUNT(*) FROM articles")
            else:
                article_count = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM articles a
                    JOIN ids_per_baseline b ON a.pmid = b.pmid AND b.year = $1
                    """,
                    year,
                )

        client = await self.get_qdrant_client()
        collection_exists = await client.collection_exists(self.collection_name)
        qdrant_count = None
        if collection_exists:
            qdrant_count = int((await client.count(self.collection_name, exact=False)).count)

        return {
            "article_count": int(article_count or 0),
            "qdrant_collection_exists": collection_exists,
            "qdrant_point_count": qdrant_count,
        }

    async def bm25_search(
        self,
        query: str,
        *,
        topk: int,
        year: int | None = None,
        exclude_pmids: set[str] | None = None,
    ) -> list[CorpusDocument]:
        """BM25 full-text search over the existing articles table."""

        exclude_ids = sorted(int(pmid) for pmid in exclude_pmids) if exclude_pmids else None
        pool = await get_pool()
        async with pool.acquire() as conn:
            if year is not None:
                rows = await conn.fetch(
                    f"""
                    SELECT
                        a.pmid,
                        a.full_text,
                        a.full_text <@> to_bm25query($1, '{_ARTICLES_BM25_INDEX}') AS score
                    FROM articles a
                    JOIN ids_per_baseline b ON a.pmid = b.pmid AND b.year = $4
                    WHERE ($3::bigint[] IS NULL OR cardinality($3::bigint[]) = 0
                           OR a.pmid != ALL($3::bigint[]))
                    ORDER BY score
                    LIMIT $2
                    """,
                    query,
                    topk,
                    exclude_ids,
                    year,
                )
            else:
                rows = await conn.fetch(
                    f"""
                    SELECT
                        pmid,
                        full_text,
                        full_text <@> to_bm25query($1, '{_ARTICLES_BM25_INDEX}') AS score
                    FROM articles
                    WHERE ($3::bigint[] IS NULL OR cardinality($3::bigint[]) = 0
                           OR pmid != ALL($3::bigint[]))
                    ORDER BY score
                    LIMIT $2
                    """,
                    query,
                    topk,
                    exclude_ids,
                )

        return [
            CorpusDocument(
                pmid=str(row["pmid"]),
                text=str(row["full_text"]),
                token_count=0,
                score=-float(row["score"]),
            )
            for row in rows
        ]

    async def semantic_search(
        self,
        query_embedding: np.ndarray,
        *,
        topk: int,
        year: int | None = None,
        exclude_pmids: set[str] | None = None,
    ) -> list[CorpusDocument]:
        """Dense retrieval over the existing Qdrant article embeddings."""

        fetch_k = max(topk * 2, 200) if year is not None else topk
        client = await self.get_qdrant_client()

        query_filter = None
        if exclude_pmids:
            query_filter = Filter(
                must_not=[HasIdCondition(has_id=sorted(int(pmid) for pmid in exclude_pmids))]
            )

        result = await client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.astype(np.float32).tolist(),
            limit=fetch_k,
            query_filter=query_filter,
            with_payload=False,
            with_vectors=False,
        )
        ranked_pmids = [(str(point.id), float(point.score)) for point in result.points]
        if not ranked_pmids:
            return []

        hydrated = await self._fetch_documents_by_pmids(
            [pmid for pmid, _ in ranked_pmids],
            year=year,
        )
        by_pmid = {document.pmid: document for document in hydrated}
        return [
            replace(by_pmid[pmid], score=score) for pmid, score in ranked_pmids if pmid in by_pmid
        ][:topk]

    async def grep_search(
        self,
        pattern: str,
        *,
        topk: int,
        year: int | None = None,
        exclude_pmids: set[str] | None = None,
        preview_tokens: int,
    ) -> list[CorpusDocument]:
        """Regex search over full documents, returned as PMID-level previews."""

        exclude_ids = sorted(int(pmid) for pmid in exclude_pmids) if exclude_pmids else None
        pool = await get_pool()
        async with pool.acquire() as conn:
            if year is not None:
                rows = await conn.fetch(
                    """
                    SELECT a.pmid, a.full_text
                    FROM articles a
                    JOIN ids_per_baseline b ON a.pmid = b.pmid AND b.year = $4
                    WHERE a.full_text ~* $1
                      AND ($3::bigint[] IS NULL OR cardinality($3::bigint[]) = 0
                           OR a.pmid != ALL($3::bigint[]))
                    ORDER BY a.pmid
                    LIMIT $2
                    """,
                    pattern,
                    topk,
                    exclude_ids,
                    year,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT pmid, full_text
                    FROM articles
                    WHERE full_text ~* $1
                      AND ($3::bigint[] IS NULL OR cardinality($3::bigint[]) = 0
                           OR pmid != ALL($3::bigint[]))
                    ORDER BY pmid
                    LIMIT $2
                    """,
                    pattern,
                    topk,
                    exclude_ids,
                )

        documents: list[CorpusDocument] = []
        for row in rows:
            preview_text = self._match_preview(str(row["full_text"]), pattern, preview_tokens)
            documents.append(
                CorpusDocument(
                    pmid=str(row["pmid"]),
                    text=preview_text,
                    token_count=self.token_counter(preview_text),
                    is_expanded=False,
                )
            )
        return documents

    async def read_document(self, pmid: str, *, max_tokens: int) -> CorpusDocument | None:
        """Load one PMID as a visible document entry for the agent."""

        full_text = await self.get_document_text(pmid)
        if not full_text:
            return None
        visible_text = self.text_truncator(full_text, max_tokens)
        return CorpusDocument(
            pmid=pmid,
            text=visible_text,
            token_count=self.token_counter(visible_text),
            is_expanded=True,
        )

    async def hybrid_search_candidates(
        self,
        query: str,
        *,
        bm25_topk: int,
        dense_topk: int,
        year: int | None = None,
        exclude_pmids: set[str] | None = None,
        rrf_k: int = 60,
    ) -> list[CorpusDocument]:
        """Run BM25 and dense PMID retrieval in parallel and fuse with RRF."""

        query_embedding = (await embed_queries_tei([query], embed_url=self.tei_embed_url))[0]
        bm25_task = asyncio.create_task(
            self.bm25_search(
                query,
                topk=bm25_topk,
                year=year,
                exclude_pmids=exclude_pmids,
            )
        )
        dense_task = asyncio.create_task(
            self.semantic_search(
                query_embedding,
                topk=dense_topk,
                year=year,
                exclude_pmids=exclude_pmids,
            )
        )
        bm25_result, dense_result = await asyncio.gather(
            bm25_task,
            dense_task,
            return_exceptions=True,
        )

        bm25_documents = [] if isinstance(bm25_result, Exception) else bm25_result
        dense_documents = [] if isinstance(dense_result, Exception) else dense_result

        if not bm25_documents and not dense_documents:
            errors = [
                str(result)
                for result in (bm25_result, dense_result)
                if isinstance(result, Exception)
            ]
            message = "Hybrid retrieval failed"
            if errors:
                message = f"{message}: {'; '.join(errors)}"
            raise RuntimeError(message)

        if not bm25_documents:
            return dense_documents
        if not dense_documents:
            return bm25_documents
        return fuse_rrf_documents(
            (("bm25", bm25_documents), ("dense", dense_documents)),
            rrf_k=rrf_k,
        )

    def preview_documents(
        self,
        documents: Sequence[CorpusDocument],
        *,
        max_tokens: int,
    ) -> list[CorpusDocument]:
        """Convert full documents into short previews suitable for the search context."""

        previews: list[CorpusDocument] = []
        for document in documents:
            preview_text = self.text_truncator(document.text, max_tokens)
            previews.append(
                replace(
                    document,
                    text=preview_text,
                    token_count=self.token_counter(preview_text),
                    is_expanded=False,
                )
            )
        return previews

    async def get_document_text(self, pmid: str) -> str:
        """Hydrate the full article text for a PMID."""

        document = await get_article_by_id(int(pmid))
        return "" if document is None else document.full_text

    async def _fetch_documents_by_pmids(
        self,
        pmids: Sequence[str],
        *,
        year: int | None = None,
    ) -> list[CorpusDocument]:
        if not pmids:
            return []

        pmid_ints = [int(pmid) for pmid in pmids]
        pool = await get_pool()
        async with pool.acquire() as conn:
            if year is None:
                rows = await conn.fetch(
                    "SELECT pmid, full_text FROM articles WHERE pmid = ANY($1)",
                    pmid_ints,
                )
            else:
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

        by_pmid = {
            str(row["pmid"]): CorpusDocument(
                pmid=str(row["pmid"]),
                text=str(row["full_text"]),
                token_count=0,
            )
            for row in rows
        }
        return [by_pmid[pmid] for pmid in pmids if pmid in by_pmid]

    def _match_preview(self, text: str, pattern: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        try:
            match = re.search(pattern, text, flags=re.IGNORECASE)
        except re.error:
            match = None

        if match is None:
            return self.text_truncator(text, max_tokens)

        window = 1_500
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        excerpt = text[start:end]
        if start > 0:
            excerpt = f"...{excerpt}"
        if end < len(text):
            excerpt = f"{excerpt}..."
        return self.text_truncator(excerpt, max_tokens)
