"""Type aliases for the BioASQ pipeline.

Uses Python 3.12+ ``type`` statement syntax for clarity and forward-reference
support.  All aliases used across the codebase are centralised here.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Scalar identity aliases
# ---------------------------------------------------------------------------

type DocumentId = str
"""PubMed document identifier (PMID)."""

type QuestionId = str
"""BioASQ question identifier."""

type Score = float
"""Numeric relevance or confidence score."""


# ---------------------------------------------------------------------------
# Phase A - reranker dataset aliases
# ---------------------------------------------------------------------------

type SliceDataset = dict[str, dict[str | int, list[dict[str, str]] | str]]
"""Training dataset sliced by question ID.

Structure::

    {
        "<question_id>": {
            "question": "<body text>",
            0: [{"id": "<pmid>", "text": "..."}],   # negatives
            1: [{"id": "<pmid>", "text": "..."}],   # positives
            ...                                       # expanded levels
        }
    }
"""

type Collection = dict[str, str]
"""Mapping of document ID → full document text (title + abstract)."""

type QrelsDict = dict[str, dict[str, int]]
"""Relevance judgements: ``{question_id: {doc_id: relevance_int}}``."""

type Sample = dict[str, str | int]
"""Raw training / inference sample before tokenisation.

Training::

    {"id": "...", "query_text": "...", "doc_text": "...", "label": 0|1}

Inference::

    {"id": "...", "doc_id": "...", "query_text": "...", "doc_text": "..."}
"""

type ProcessedSample = dict[str, Sample | list[Sample] | str | int]
"""Post-tokenisation sample.

Pointwise::

    {"input_ids": [...], "attention_mask": [...], "labels": 1, ...}

Pairwise::

    {"pos_inputs": {<Sample>}, "neg_inputs": {<Sample>}}

Multi-negative::

    {"pos_inputs": {<Sample>}, "neg_inputs": [{<Sample>}, ...]}
"""


# ---------------------------------------------------------------------------
# Phase A - retrieval / evaluation
# ---------------------------------------------------------------------------

type RunDict = dict[QuestionId, dict[DocumentId, Score]]
"""Retrieval run: ``{question_id: {doc_id: score}}``."""
