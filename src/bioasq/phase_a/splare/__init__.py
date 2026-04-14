"""SPLARE — SParse LAtent REtrieval via Sparse Autoencoders.

Implements SAE-based learned sparse retrieval as a third retrieval signal
alongside BM25 and dense embeddings, following the SPLARE paper
(Formal et al., ICLR 2026).
"""

from bioasq.phase_a.splare.model import SplareModel
from bioasq.phase_a.splare.search import splare_search

__all__ = [
    "SplareModel",
    "splare_search",
]
