"""Phase A hybrid retrieval: BM25 + dense DB search, RRF fusion, multi-reranker RRF."""

from bioasq.phase_a.retrieval.fusion import fuse_rerank_run_dicts, fuse_retrieval_lists_rrf
from bioasq.phase_a.retrieval.pipeline import apply_rerankers_and_fuse, hybrid_retrieve_rrf
from bioasq.phase_a.retrieval.query_encoder import embed_queries_tei

__all__ = [
    "apply_rerankers_and_fuse",
    "embed_queries_tei",
    "fuse_rerank_run_dicts",
    "fuse_retrieval_lists_rrf",
    "hybrid_retrieve_rrf",
]
