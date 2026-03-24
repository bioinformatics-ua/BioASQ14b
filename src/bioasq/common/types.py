"""Structured data types for the BioASQ pipeline.

Every domain object is a :class:`msgspec.Struct` subclass — never a raw
``dict``.  Types are kept ``frozen=True`` where mutability is unnecessary.
"""

from __future__ import annotations

from enum import StrEnum

import msgspec

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class QuestionType(StrEnum):
    """BioASQ question categories."""

    YESNO = "yesno"
    FACTOID = "factoid"
    LIST = "list"
    SUMMARY = "summary"


# ---------------------------------------------------------------------------
# Core domain objects
# ---------------------------------------------------------------------------


class Document(msgspec.Struct, frozen=True):
    """A PubMed document (article)."""

    id: str
    title: str = ""
    abstract: str = ""
    text: str = ""
    url: str = ""


class NegDoc(msgspec.Struct, frozen=True):
    """A negative document with its BM25 retrieval score."""

    id: str
    text: str = ""
    score: float = 0.0


class Snippet(msgspec.Struct, frozen=True):
    """A text snippet extracted from a document."""

    text: str
    document: str = ""
    offset_in_begin_section: str = ""
    offset_in_end_section: str = ""
    begin_section: str = ""
    end_section: str = ""


class Question(msgspec.Struct, frozen=True):
    """A BioASQ question with all associated data."""

    id: str
    body: str
    type: QuestionType = QuestionType.SUMMARY
    documents: list[str] = []
    snippets: list[Snippet] = []
    ideal_answer: list[str] | str | None = None
    exact_answer: list[str] | list[list[str]] | str | None = None


# ---------------------------------------------------------------------------
# Phase A - reranker types
# ---------------------------------------------------------------------------


class QuestionWithNegatives(msgspec.Struct):
    """Training question with positive and negative document candidates."""

    id: str
    body: str
    pos_docs: list[Document] = []
    neg_docs: list[NegDoc] = []


class RerankerPrediction(msgspec.Struct, frozen=True):
    """A single reranker score for a (question, document) pair."""

    question_id: str
    document_id: str
    score: float


# ---------------------------------------------------------------------------
# Phase B - answer generation types
# ---------------------------------------------------------------------------


class GeneratedAnswer(msgspec.Struct):
    """Model-generated answer for a BioASQ question."""

    text: str
    valid: bool = False
    raw: str = ""


class SynthesisResult(msgspec.Struct):
    """Result of synthesis step combining multiple generated answers."""

    ideal_answer: str
    exact_answer: list[str] | list[list[str]] | str | None = None
    valid: bool = False
    raw: str = ""


# ---------------------------------------------------------------------------
# Submission format
# ---------------------------------------------------------------------------


class BioASQSubmissionQuestion(msgspec.Struct):
    """Single question entry in a BioASQ submission file."""

    id: str
    type: QuestionType = QuestionType.SUMMARY
    body: str = ""
    documents: list[str] = []
    snippets: list[Snippet] = []
    ideal_answer: str = ""
    exact_answer: list[str] | list[list[str]] | str | None = None


class BioASQSubmission(msgspec.Struct):
    """Top-level BioASQ submission format."""

    questions: list[BioASQSubmissionQuestion] = []
