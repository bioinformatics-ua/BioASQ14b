"""Unified I/O utilities for the BioASQ pipeline.

Replaces all scattered ``json.load``, ``orjson.loads``, and raw
``open() + read()`` patterns with typed helpers backed by
:mod:`msgspec.json`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, overload

import msgspec

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")

_json_encoder: msgspec.json.Encoder = msgspec.json.Encoder()
_json_decoder: msgspec.json.Decoder[object] = msgspec.json.Decoder()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


@overload
def load_json[T](path: str | Path, *, type: type[T]) -> T: ...
@overload
def load_json(path: str | Path) -> object: ...


def load_json[T](path: str | Path, *, type_: type[T] | None = None) -> T | object:
    """Load a JSON file, optionally decoding into a typed *msgspec* struct.

    Parameters
    ----------
    path:
        Filesystem path to the JSON file.
    type_:
        If provided, decode into this ``msgspec.Struct`` (or other type).
        Otherwise return a plain Python object (dict / list).
    """
    raw: bytes = Path(path).read_bytes()
    if type_ is not None:
        return msgspec.json.decode(raw, type=type_)
    return msgspec.json.decode(raw)


def save_json(data: object, path: str | Path, *, indent: bool = False) -> None:
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
    p: Path = Path(path)
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
def load_jsonl[T](path: str | Path, *, type: type[T]) -> list[T]: ...
@overload
def load_jsonl(path: str | Path) -> list[object]: ...


def load_jsonl[T](
    path: str | Path, *, type_: type[T] | None = None
) -> list[T] | list[object]:
    """Load a JSONL file (one JSON object per line).

    Parameters
    ----------
    path:
        Filesystem path to the JSONL file.
    type_:
        If provided, decode each line into this type.
    """
    results: list[object] = []
    with Path(path).open("rb") as fh:
        for line in fh:
            stripped: bytes = line.strip()
            if not stripped:
                continue
            if type_ is not None:
                results.append(msgspec.json.decode(stripped, type=type))
            else:
                results.append(msgspec.json.decode(stripped))
    return results  # type: ignore[return-value]


def save_jsonl(
    data: Sequence[object],
    path: str | Path,
) -> None:
    """Serialise a sequence of objects as JSONL.

    Parameters
    ----------
    data:
        Iterable of msgspec-encodable objects.
    path:
        Destination file path (parent dirs created automatically).
    """
    p: Path = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        for item in data:
            fh.write(msgspec.json.encode(item))
            fh.write(b"\n")
