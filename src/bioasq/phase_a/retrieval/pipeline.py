"""End-to-end hybrid retrieval and optional multi-reranker fusion."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np

from bioasq.common.aliases import DocumentId, QuestionId, RunDict
from bioasq.common.types import DocumentWithScore
from bioasq.data.database import bm25_search, semantic_search
from bioasq.phase_a.retrieval import fuse_retrieval_lists_wsum
from bioasq.phase_a.retrieval.fusion import fuse_rerank_run_dicts, fuse_retrieval_lists_rrf
from bioasq.phase_a.retrieval.query_encoder import embed_queries_tei
from bioasq.phase_a.retrieval.query_expansion import generate_hyde_document
from bioasq.phase_a.splare.search import splare_search
from bioasq.phase_b.backends.base import BaseModelBackend

type RerankerFn = Callable[
    [QuestionId, str, list[DocumentWithScore]],
    Awaitable[dict[str, float]],
]


@dataclass
class SplitRetrievalResult:
    """Holds BM25 and dense results separately before fusion."""

    bm25_docs: list[DocumentWithScore]
    dense_docs: list[DocumentWithScore]


async def hybrid_retrieve(
    qid: QuestionId,
    query_text: str,
    *,
    year: int | None = None,
    bm25_topk: int = 200,
    semantic_topk: int = 200,
    query_embedding: np.ndarray | None = None,
    embed_url: str | None = None,
    exclude_ids: set[DocumentId] | None = None,
    rrf_k: int = 60,
    hyde_backend: BaseModelBackend | None = None,
) -> tuple[list[DocumentWithScore], list[DocumentWithScore], list[DocumentWithScore]]:
    """
    Parallel BM25 + dense DB search, then RRF fusion.

    Returns a tuple of (RRF-fused, WSum-fused, BM25-only) results.

    If ``query_embedding`` is omitted, encodes ``query_text`` via TEI
    (:func:`embed_queries_tei`).  Pass *year* to restrict both searches to
    the corresponding PubMed baseline.
    """
    if query_embedding is None:
        dense_query_text = query_text
        if hyde_backend is not None:
            dense_query_text = generate_hyde_document(query_text, hyde_backend)
        mat = await embed_queries_tei([dense_query_text], embed_url=embed_url)
        query_embedding = mat[0]

    bm25_task = bm25_search(query_text, topk=bm25_topk, year=year, exclude_ids=exclude_ids)
    dense_task = semantic_search(
        query_embedding, topk=semantic_topk, year=year, exclude_ids=exclude_ids
    )
    bm25_docs, dense_docs = await asyncio.gather(bm25_task, dense_task)
    bm25_docs = [d for d in bm25_docs if len(d.full_text) > 200]
    dense_docs = [d for d in dense_docs if len(d.full_text) > 200]

    bm25_docs = bm25_docs[:100]
    dense_docs = dense_docs[:100]

    rrf_docs = fuse_retrieval_lists_rrf(
        qid,
        [("bm25", bm25_docs), ("dense", dense_docs)],
        rrf_k=rrf_k,
    )

    wsum_docs = fuse_retrieval_lists_wsum(
        qid,
        [("bm25", bm25_docs), ("dense", dense_docs)],
        weights=[0.6, 0.4],
    )
    return rrf_docs, wsum_docs, bm25_docs


async def hybrid_retrieve_with_splare(
    qid: QuestionId,
    query_text: str,
    *,
    year: int | None = None,
    bm25_topk: int = 200,
    semantic_topk: int = 200,
    splare_topk: int = 200,
    query_embedding: np.ndarray | None = None,
    query_sparse: tuple[list[int], list[float]] | None = None,
    embed_url: str | None = None,
    exclude_ids: set[DocumentId] | None = None,
    rrf_k: int = 60,
) -> tuple[list[DocumentWithScore], list[DocumentWithScore], list[DocumentWithScore]]:
    """
    Parallel BM25 + dense + SPLARE search, then 3-way RRF fusion.

    Returns a tuple of (RRF-3way-fused, RRF-2way-fused, BM25-only) results.

    If ``query_sparse`` is omitted, encodes ``query_text`` via the local
    SPLARE model (expensive — requires GPU). Pass pre-computed sparse vectors
    when possible.
    """
    # Dense embedding
    if query_embedding is None:
        mat = await embed_queries_tei([query_text], embed_url=embed_url)
        query_embedding = mat[0]

    # SPLARE sparse embedding
    if query_sparse is None:
        from bioasq.phase_a.splare.query_encoder import encode_queries_splare

        sparse_vecs = encode_queries_splare([query_text])
        query_sparse = sparse_vecs[0]

    bm25_task = bm25_search(query_text, topk=bm25_topk, year=year, exclude_ids=exclude_ids)
    dense_task = semantic_search(
        query_embedding, topk=semantic_topk, year=year, exclude_ids=exclude_ids
    )
    splare_task = splare_search(
        query_sparse[0], query_sparse[1], topk=splare_topk, year=year, exclude_ids=exclude_ids
    )

    bm25_docs, dense_docs, splare_docs = await asyncio.gather(bm25_task, dense_task, splare_task)

    bm25_docs = [d for d in bm25_docs if len(d.full_text) > 200]
    dense_docs = [d for d in dense_docs if len(d.full_text) > 200]
    splare_docs = [d for d in splare_docs if len(d.full_text) > 200]

    bm25_docs = bm25_docs[:100]
    dense_docs = dense_docs[:100]
    splare_docs = splare_docs[:100]

    # 3-way RRF fusion (BM25 + Dense + SPLARE)
    rrf_3way_docs = fuse_retrieval_lists_rrf(
        qid,
        [("bm25", bm25_docs), ("dense", dense_docs), ("splare", splare_docs)],
        rrf_k=rrf_k,
    )

    # 2-way RRF as fallback comparison
    rrf_2way_docs = fuse_retrieval_lists_rrf(
        qid,
        [("bm25", bm25_docs), ("dense", dense_docs)],
        rrf_k=rrf_k,
    )

    return rrf_3way_docs, rrf_2way_docs, bm25_docs


async def apply_rerankers_and_fuse(
    qid: QuestionId,
    query_text: str,
    candidates: Sequence[DocumentWithScore],
    rerankers: Sequence[RerankerFn],
    *,
    rrf_k: int = 60,
) -> list[DocumentWithScore]:
    """
    Score ``candidates`` with each async reranker, then fuse rankings with RRF.

    Each reranker returns ``{pmid: score}`` (higher = more relevant). Documents
    omitted by a reranker are excluded from that run only.
    """
    cand_list = list(candidates)
    if not rerankers:
        return list(cand_list)

    score_maps: list[dict[str, float]] = []
    for rank_fn in rerankers:
        score_maps.append(await rank_fn(qid, query_text, cand_list))

    runs: list[RunDict] = []
    for sm in score_maps:
        runs.append({qid: {doc_id: float(s) for doc_id, s in sm.items()}})

    fused_scores = fuse_rerank_run_dicts(runs, rrf_k=rrf_k).get(qid, {})
    text_by_pmid = {d.pmid: d.full_text for d in cand_list}
    ordered = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        DocumentWithScore(pmid=doc_id, full_text=text_by_pmid.get(doc_id, ""), score=fs)
        for doc_id, fs in ordered
        if doc_id in text_by_pmid
    ]


async def hybrid_retrieve_split(
    qid: QuestionId,
    query_text: str,
    *,
    year: int | None = None,
    bm25_topk: int = 100,
    semantic_topk: int = 100,
    query_embedding: np.ndarray | None = None,
    embed_url: str | None = None,
    exclude_ids: set[DocumentId] | None = None,
) -> SplitRetrievalResult:
    """Parallel BM25 + dense DB search, returning results separately (not fused).

    Use this when the reranker was trained on BM25 data only — rerank the
    BM25 candidates, then fuse with raw dense via :func:`rerank_bm25_then_fuse`.
    """
    if query_embedding is None:
        mat = await embed_queries_tei([query_text], embed_url=embed_url)
        query_embedding = mat[0]

    bm25_task = bm25_search(query_text, topk=bm25_topk, year=year, exclude_ids=exclude_ids)
    dense_task = semantic_search(
        query_embedding, topk=semantic_topk, year=year, exclude_ids=exclude_ids
    )
    bm25_docs, dense_docs = await asyncio.gather(bm25_task, dense_task)

    return SplitRetrievalResult(
        bm25_docs=sorted(bm25_docs, key=lambda d: d.score, reverse=True),
        dense_docs=sorted(dense_docs, key=lambda d: d.score, reverse=True),
    )


async def rerank_bm25_then_fuse(
    qid: QuestionId,
    query_text: str,
    split: SplitRetrievalResult,
    rerankers: Sequence[RerankerFn],
    *,
    rrf_k: int = 60,
) -> list[DocumentWithScore]:
    """Rerank only BM25 candidates, then RRF-fuse with raw dense scores.

    Avoids feeding dense-retrieved documents to a reranker that was trained
    exclusively on BM25 inputs, which would produce unreliable scores and
    bury potentially relevant dense-only results.

    Strategy:
      1. Apply reranker(s) to BM25 candidates only → reranked BM25 run
      2. Build a raw dense run from original dense retrieval scores
      3. RRF-fuse the two runs → final hybrid ranking
    """
    bm25_docs = split.bm25_docs
    dense_docs = split.dense_docs

    # Collect text for final output
    text_by_pmid: dict[str, str] = {}
    for d in bm25_docs:
        text_by_pmid.setdefault(d.pmid, d.full_text)
    for d in dense_docs:
        text_by_pmid.setdefault(d.pmid, d.full_text)

    # Rerank BM25 candidates
    if rerankers:
        reranked_runs: list[RunDict] = []
        for rank_fn in rerankers:
            scores = await rank_fn(qid, query_text, bm25_docs)
            reranked_runs.append({qid: {doc_id: float(s) for doc_id, s in scores.items()}})
        reranked_bm25 = fuse_rerank_run_dicts(reranked_runs, rrf_k=rrf_k).get(qid, {})
    else:
        reranked_bm25 = {d.pmid: float(d.score) for d in bm25_docs}

    # Raw dense scores
    dense_scores = {d.pmid: float(d.score) for d in dense_docs}

    # RRF-fuse reranked BM25 with raw dense
    fused = fuse_retrieval_lists_rrf(
        qid,
        [
            (
                "reranked_bm25",
                [
                    DocumentWithScore(pmid=pid, full_text=text_by_pmid.get(pid, ""), score=s)
                    for pid, s in reranked_bm25.items()
                ],
            ),
            (
                "dense",
                [
                    DocumentWithScore(pmid=pid, full_text=text_by_pmid.get(pid, ""), score=s)
                    for pid, s in dense_scores.items()
                ],
            ),
        ],
        rrf_k=rrf_k,
    )
    return fused
