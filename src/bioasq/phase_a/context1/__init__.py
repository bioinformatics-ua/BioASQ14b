"""Context-1 style agentic retrieval for BioASQ Phase A.

This subpackage implements an article-level hybrid search stack and a
tool-calling retrieval harness inspired by Chroma Context-1. It is
inference-focused: the agent runs against the local BioASQ corpus using a
vLLM-served ``chromadb/context-1`` model, while retrieval tools are backed by
the existing Postgres article table, the existing Qdrant article embeddings,
and the TEI query encoder used elsewhere in this repository.
"""

from bioasq.phase_a.context1.harness import Context1Agent
from bioasq.phase_a.context1.store import Context1CorpusStore
from bioasq.phase_a.context1.vllm_backend import Context1VLLMOpenAIBackend

__all__ = [
    "Context1Agent",
    "Context1CorpusStore",
    "Context1VLLMOpenAIBackend",
]
