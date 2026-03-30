"""Encode query text via Text Embeddings Inference (TEI), aligned with collection embeddings."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import httpx
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def default_tei_embed_url() -> str:
    """TEI HTTP embed endpoint from BIOASQ_TEI_URL or localhost default."""
    base = os.environ.get("BIOASQ_TEI_URL", "http://127.0.0.1:8080").rstrip("/")
    return f"{base}/embed"


async def embed_queries_tei(
    texts: Sequence[str],
    *,
    embed_url: str | None = None,
    timeout: float | None = 120.0,
) -> np.ndarray:
    """
    Call TEI ``POST …/embed`` and return float32 matrix ``(len(texts), dim)``.
    Same contract as :mod:`bioasq.data.embeddings.generate_embeddings` (one row per input).
    """
    url = embed_url or default_tei_embed_url()
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json={"inputs": list(texts)}, timeout=timeout)
        response.raise_for_status()
        payload: list[list[float]] = response.json()
    arr = np.asarray(payload, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(texts):
        msg = f"TEI returned shape {arr.shape}, expected ({len(texts)}, dim)"
        raise RuntimeError(msg)
    return arr
