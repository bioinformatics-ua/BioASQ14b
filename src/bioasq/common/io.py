"""Unified I/O utilities for the BioASQ pipeline.

Replaces all scattered ``json.load``, ``orjson.loads``, and raw
``open() + read()`` patterns with typed helpers backed by
:mod:`msgspec.json`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

import msgspec

from bioasq.common.decoders import document_decoder

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from bioasq.common.types import Document


T = TypeVar("T")

_json_encoder: msgspec.json.Encoder = msgspec.json.Encoder()
_json_decoder: msgspec.json.Decoder[object] = msgspec.json.Decoder()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


@overload
def load_json[T](path: Path, *, type: type[T]) -> T: ...
@overload
def load_json(path: Path) -> object: ...


def load_json[T](path: Path, *, type_: type[T] | None = None) -> T | object:
    """Load a JSON file, optionally decoding into a typed *msgspec* struct.

    Parameters
    ----------
    path:
        Filesystem path to the JSON file.
    type_:
        If provided, decode into this ``msgspec.Struct`` (or other type).
        Otherwise return a plain Python object (dict / list).
    """
    raw: bytes = path.read_bytes()
    if type_ is not None:
        return msgspec.json.decode(raw, type=type_)
    return msgspec.json.decode(raw)


def save_json(data: object, path: Path, *, indent: bool = False) -> None:
    """Serialise *data* to a JSON file using msgspec.

    Parameters
    ----------
    data:
        Any msgspec-encodable object (Struct, dict, list, …).
    path:
        Destination file path (parent dirs created automatically).
    indent:
        Pretty-print with 2-space indentation if ``True``.
    """
    p: Path = path
    p.parent.mkdir(parents=True, exist_ok=True)
    if indent:
        encoded: bytes = msgspec.json.format(msgspec.json.encode(data), indent=2)
    else:
        encoded = msgspec.json.encode(data)
    p.write_bytes(encoded)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


@overload
def load_jsonl[T](path: Path, *, type: type[T]) -> list[T]: ...
@overload
def load_jsonl(path: Path) -> list[object]: ...


def load_jsonl[T](path: Path, *, type_: type[T] | None = None) -> list[T] | list[object]:
    """Load a JSONL file (one JSON object per line).

    Parameters
    ----------
    path:
        Filesystem path to the JSONL file.
    type_:
        If provided, decode each line into this type.
    """
    results: list[object] = []
    with path.open("rb") as fh:
        for line in fh:
            stripped: bytes = line.strip()
            if not stripped:
                continue
            if type_ is not None:
                results.append(msgspec.json.decode(stripped, type=type_))
            else:
                results.append(msgspec.json.decode(stripped))
    return results


@overload
def load_collection(
    path: Path, constraints: list[Callable[[Document], bool]] | None = None
) -> Iterator[Document]: ...


@overload
def load_collection(
    path: Path, constraints: list[Callable[[Document], bool]] | None = None, chunk_size: int = 50000
) -> Iterator[list[Document]]: ...


def load_collection(
    path: Path, constraints: list[Callable[[Document], bool]] | None = None, chunk_size: int = 1
) -> Iterator[Document] | Iterator[list[Document]]:
    """Load a collection of articles from a JSONL file.

    Example of call:
    load_collection(Path("data/bioasq_2024_collection.jsonl"), \
        constraints=[lambda x: len(f"{x['title']} {x['abstract']}") > 100])

    Parameters
    ----------
    path:
        Filesystem path to the JSONL file.
    constraints:
        based on this collection: dict[int, PMID] = {
            i: line.lstrip(b'{"pmid": "').split(b'"', 1)[0].decode("utf-8")
            for i, line in enumerate(f)
            if len(line) < 1000
        }
    chunk_size:
        Size of the chunk to load.

    Returns:
        Iterator[Document]: Iterator of documents.
        Iterator[list[Document]]: Iterator of lists of documents.
    """
    with path.open("rb") as f:
        chunk: list[Document] = []
        for line in f:
            article = document_decoder.decode(line)

            if constraints is not None:
                for constraint in constraints:
                    if not constraint(article):
                        continue

            if chunk_size == 1:
                yield article
                continue

            chunk.append(article)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk_size != 1 and chunk:
            yield chunk


def save_jsonl(
    data: Sequence[object],
    path: Path,
) -> None:
    """Serialise a sequence of objects as JSONL.

    Parameters
    ----------
    data:
        Iterable of msgspec-encodable objects.
    path:
        Destination file path (parent dirs created automatically).
    """
    p: Path = path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        for item in data:
            fh.write(msgspec.json.encode(item))
            fh.write(b"\n")
