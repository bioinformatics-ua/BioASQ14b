"""Sentence-aware chunking helpers for Context-1 retrieval."""

from __future__ import annotations

import re
from collections.abc import Callable

_SPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace without changing textual content."""
    return _SPACE_RE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split biomedical text into coarse sentence units.

    This intentionally avoids external sentence models or NLTK downloads so the
    indexing pipeline stays self-contained inside the repository environment.
    """

    clean = normalize_text(text)
    if not clean:
        return []
    pieces = _SENTENCE_SPLIT_RE.split(clean)
    sentences = [piece.strip() for piece in pieces if piece.strip()]
    return sentences if sentences else [clean]


def chunk_document(
    full_text: str,
    *,
    token_counter: Callable[[str], int],
    max_tokens: int,
    overlap_sentences: int,
) -> list[tuple[int, str, int]]:
    """Create overlapping sentence windows that fit a token budget."""

    sentences = split_sentences(full_text)
    if not sentences:
        return []

    chunks: list[tuple[int, str, int]] = []
    start = 0
    chunk_index = 0

    while start < len(sentences):
        end = start
        current: list[str] = []
        best_text = ""
        best_tokens = 0

        while end < len(sentences):
            candidate = " ".join([*current, sentences[end]])
            candidate_tokens = token_counter(candidate)
            if current and candidate_tokens > max_tokens:
                break
            current.append(sentences[end])
            best_text = candidate
            best_tokens = candidate_tokens
            end += 1
            if candidate_tokens >= max_tokens:
                break

        if not best_text:
            single_sentence = sentences[start]
            best_text = single_sentence
            best_tokens = token_counter(single_sentence)
            end = start + 1

        chunks.append((chunk_index, best_text, best_tokens))
        chunk_index += 1

        if end >= len(sentences):
            break

        start = max(start + 1, end - max(0, overlap_sentences))

    return chunks
