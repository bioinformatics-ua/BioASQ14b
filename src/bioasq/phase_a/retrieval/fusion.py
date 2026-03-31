"""Reciprocal Rank Fusion (RRF) via ranx for retrieval and reranker outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ranx import Run
from ranx.fusion import rrf, wsum

from bioasq.common.types import DocumentWithScore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bioasq.common.aliases import DocumentId, QuestionId, RunDict


def fuse_retrieval_lists_rrf(
    qid: QuestionId,
    named_lists: Sequence[tuple[str, Sequence[DocumentWithScore]]],
    *,
    rrf_k: int = 60,
) -> list[DocumentWithScore]:
    """
    Fuse multiple ranked document lists with RRF (Cormack et al., SIGIR 2009).

    Each list keeps its own retrieval scores for ordering inside ranx; RRF
    produces the final fused scores on the union of PMIDs.
    """
    text_by_pmid: dict[DocumentId, str] = {}
    runs: list[Run] = []
    for run_name, docs in named_lists:
        for d in docs:
            text_by_pmid.setdefault(d.pmid, d.full_text)
        if docs:
            runs.append(
                Run({qid: {d.pmid: float(d.score) for d in docs}}, name=run_name),
            )

    if not runs:
        return []
    if len(runs) == 1:
        for _, docs in named_lists:
            if docs:
                return sorted(docs, key=lambda d: d.score, reverse=True)
        return []

    fused = rrf(runs, k=rrf_k, name="rrf_bm25_dense")
    fused.sort()
    scores: dict[str, float] = dict(fused.to_dict()[qid])
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        DocumentWithScore(pmid=doc_id, full_text=text_by_pmid[doc_id], score=fused_score)
        for doc_id, fused_score in ordered
    ]


def fuse_retrieval_lists_wsum(
    qid: QuestionId,
    named_lists: Sequence[tuple[str, Sequence[DocumentWithScore]]],
    *,
    weights: Sequence[float],
) -> list[DocumentWithScore]:
    """Fuse ranked lists with weighted sum (ranx ``wsum``)."""
    text_by_pmid: dict[DocumentId, str] = {}
    runs: list[Run] = []
    for run_name, docs in named_lists:
        for d in docs:
            text_by_pmid.setdefault(d.pmid, d.full_text)
        if docs:
            runs.append(
                Run({qid: {d.pmid: float(d.score) for d in docs}}, name=run_name),
            )

    if not runs:
        return []
    if len(runs) == 1:
        for _, docs in named_lists:
            if docs:
                return sorted(docs, key=lambda d: d.score, reverse=True)
        return []

    fused = wsum(runs, weights=list(weights), name="wsum_bm25_dense")
    fused.sort()
    scores: dict[str, float] = dict(fused.to_dict()[qid])
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        DocumentWithScore(pmid=doc_id, full_text=text_by_pmid[doc_id], score=fused_score)
        for doc_id, fused_score in ordered
    ]


def fuse_rerank_run_dicts(run_dicts: Sequence[RunDict], *, rrf_k: int = 60) -> RunDict:
    """
    Fuse N reranker outputs (each ``{qid: {doc_id: score}}``) with RRF.

    All runs must cover the same query ids for sensible fusion; missing
    queries in a run are treated as absent from that run.
    """
    if not run_dicts:
        return {}
    if len(run_dicts) == 1:
        return {q: dict(docs) for q, docs in run_dicts[0].items()}

    runs = [
        Run({q: dict(docs) for q, docs in rd.items()}, name=f"reranker_{i}")
        for i, rd in enumerate(run_dicts)
    ]
    fused = rrf(runs, k=rrf_k, name="rrf_rerankers")
    fused.sort()
    return dict(fused.to_dict())
