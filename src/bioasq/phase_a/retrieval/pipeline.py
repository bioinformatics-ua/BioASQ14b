"""End-to-end hybrid retrieval and optional multi-reranker fusion."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

import numpy as np  # noqa: TC002

from bioasq.common.aliases import DocumentId, QuestionId, RunDict
from bioasq.common.types import DocumentWithScore
from bioasq.data.database import bm25_search, semantic_search
from bioasq.phase_a.retrieval.fusion import fuse_rerank_run_dicts, fuse_retrieval_lists_rrf
from bioasq.phase_a.retrieval.query_encoder import embed_queries_tei

type RerankerFn = Callable[
    [QuestionId, str, list[DocumentWithScore]],
    Awaitable[dict[str, float]],
]


async def hybrid_retrieve_rrf(
    qid: QuestionId,
    query_text: str,
    *,
    year: int | None = None,
    bm25_topk: int = 100,
    semantic_topk: int = 100,
    query_embedding: np.ndarray | None = None,
    embed_url: str | None = None,
    exclude_ids: set[DocumentId] | None = None,
    rrf_k: int = 60,
) -> list[DocumentWithScore]:
    """
    Parallel BM25 + dense DB search, then RRF fusion.

    If ``query_embedding`` is omitted, encodes ``query_text`` via TEI
    (:func:`embed_queries_tei`).  Pass *year* to restrict both searches to
    the corresponding PubMed baseline.
    """
    if query_embedding is None:
        mat = await embed_queries_tei([query_text], embed_url=embed_url)
        query_embedding = mat[0]

    bm25_task = bm25_search(query_text, topk=bm25_topk, year=year, exclude_ids=exclude_ids)
    dense_task = semantic_search(query_embedding, topk=semantic_topk, year=year, exclude_ids=exclude_ids)
    bm25_docs, dense_docs = await asyncio.gather(bm25_task, dense_task)

    return fuse_retrieval_lists_rrf(
        qid,
        [("bm25", bm25_docs), ("dense", dense_docs)],
        rrf_k=rrf_k,
    )


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
