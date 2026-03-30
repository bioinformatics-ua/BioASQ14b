"""Phase A document retrieval and reranking for BioASQ.

Initial retrieval combines BM25 and dense DB search with RRF (ranx); neural
rerankers can be fused the same way (:mod:`bioasq.phase_a.retrieval`).
"""
