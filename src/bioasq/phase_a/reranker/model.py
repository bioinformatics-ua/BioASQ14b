"""Shared reranker model-loading utilities."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def resolve_inference_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Normalize a dtype alias into a torch dtype."""

    if isinstance(dtype, torch.dtype):
        return dtype

    normalized = dtype.strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported inference dtype: {dtype!r}")
    return mapping[normalized]


def _normalize_length_limit(limit: object) -> int | None:
    if not isinstance(limit, int) or limit <= 0:
        return None
    if limit >= 1_000_000:
        return None
    return limit


def resolve_effective_max_length(
    requested_max_length: int,
    *,
    tokenizer_max_length: int,
    config_max_position_embeddings: int | None,
) -> int:
    """Clamp a requested reranker max length to what the model actually supports."""

    if requested_max_length <= 0:
        raise ValueError("requested_max_length must be positive")

    limits = [requested_max_length]
    normalized_tokenizer_limit = _normalize_length_limit(tokenizer_max_length)
    normalized_config_limit = _normalize_length_limit(config_max_position_embeddings)
    if normalized_tokenizer_limit is not None:
        limits.append(normalized_tokenizer_limit)
    if normalized_config_limit is not None:
        limits.append(normalized_config_limit)
    return min(limits)


def load_reranker_model(
    model_name: str,
    *,
    max_length: int = 1_024,
    dtype: str | torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a cross-encoder reranker model and clamp tokenizer length if needed."""

    resolved_dtype = resolve_inference_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    effective_max_length = resolve_effective_max_length(
        max_length,
        tokenizer_max_length=int(tokenizer.model_max_length),
        config_max_position_embeddings=getattr(config, "max_position_embeddings", None),
    )
    if effective_max_length != max_length:
        warnings.warn(
            (
                f"Requested reranker max_length={max_length} exceeds the model limit; "
                f"clamping to {effective_max_length}."
            ),
            UserWarning,
            stacklevel=2,
        )
    tokenizer.model_max_length = effective_max_length

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        torch_dtype=resolved_dtype,
    )
    return model, tokenizer
