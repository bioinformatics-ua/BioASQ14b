"""Query expansion via LLM generation for retrieval augmentation.

Implements two strategies:

- **Query2Doc**: Generate a pseudo-document from the query, concatenate
  with the original query for BM25 lexical retrieval.
- **HyDE** (Hypothetical Document Embeddings): Generate a hypothetical
  PubMed abstract, embed *that* instead of the short query for dense
  retrieval.

Both use the existing :mod:`bioasq.phase_b.backends` (OpenRouter or vLLM).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bioasq.phase_b.backends.base import BaseModelBackend

# ── Prompts ──────────────────────────────────────────────────────────────────

_QUERY2DOC_PROMPT = (
    "You are a biomedical search expert. Given the following biomedical "
    "question, write a short passage (3-5 sentences) that would appear in "
    "a relevant PubMed article answering this question. Use precise "
    "biomedical terminology and include key terms, synonyms, and related "
    "concepts that would help find relevant documents.\n\n"
    "Question: {query}\n\n"
    "Relevant passage:"
)

_HYDE_PROMPT = (
    "You are a biomedical researcher. Given the following question, write "
    "a hypothetical PubMed abstract (title + abstract, ~150 words) for a "
    "paper that directly answers this question. Be specific, use standard "
    "biomedical nomenclature, include relevant gene names, protein names, "
    "drug names, disease names, and mechanisms.\n\n"
    "Question: {query}\n\n"
    "Hypothetical abstract:"
)


# ── Core functions ───────────────────────────────────────────────────────────


def expand_query2doc(
    queries: Sequence[str],
    backend: BaseModelBackend,
) -> list[str]:
    """Expand queries for BM25 by appending an LLM-generated pseudo-document.

    Returns ``original_query + " " + generated_passage`` for each query.
    """
    prompts = [_QUERY2DOC_PROMPT.format(query=q) for q in queries]
    expansions = backend.generate_batch(prompts)
    return [f"{q} {exp.strip()}" for q, exp in zip(queries, expansions, strict=True)]


def generate_hyde_documents(
    queries: Sequence[str],
    backend: BaseModelBackend,
) -> list[str]:
    """Generate hypothetical documents for HyDE dense retrieval.

    Returns the generated hypothetical abstract for each query (to be
    embedded instead of the original query).
    """
    prompts = [_HYDE_PROMPT.format(query=q) for q in queries]
    return [doc.strip() for doc in backend.generate_batch(prompts)]
