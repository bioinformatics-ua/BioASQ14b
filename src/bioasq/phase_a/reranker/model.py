"""Model loading and utility helpers for reranker training.

Extracted from ``refactored-trainer/main.py`` — these are shared by
both training and inference entry points.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def sanitize_position_ids_buffers(model: torch.nn.Module) -> None:
    """Ensure any ``position_ids`` buffers are monotonic ``0..N-1``.

    Some remote-code checkpoints may load a corrupted ``position_ids``
    buffer, causing out-of-bounds indexing in positional embeddings/RoPE.
    """
    for module in model.modules():
        position_ids: torch.Tensor | None = getattr(module, "position_ids", None)
        if not isinstance(position_ids, torch.Tensor):
            continue
        if position_ids.dtype not in (torch.int32, torch.int64):
            continue
        if position_ids.ndim == 1:
            expected: torch.Tensor = torch.arange(
                position_ids.shape[0],
                device=position_ids.device,
                dtype=position_ids.dtype,
            )
        elif position_ids.ndim == 2 and position_ids.shape[0] == 1:
            expected = torch.arange(
                position_ids.shape[1],
                device=position_ids.device,
                dtype=position_ids.dtype,
            ).unsqueeze(0)
        else:
            continue

        if not torch.equal(position_ids, expected):
            position_ids.copy_(expected)


def resolve_model_dtype(*, bf16: bool, fp16: bool) -> torch.dtype:
    """Resolve model dtype from flags.

    Raises
    ------
    ValueError
        If both ``bf16`` and ``fp16`` are ``True``.
    """
    if bf16 and fp16:
        msg: str = "bf16 and fp16 cannot both be enabled"
        raise ValueError(msg)
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return torch.float32


def resolve_inference_dtype(inference_dtype: str) -> torch.dtype:
    """Map a string dtype name to a :class:`torch.dtype`.

    Accepted names: ``float32``, ``fp32``, ``bfloat16``, ``bf16``,
    ``float16``, ``fp16``.
    """
    normalised: str = inference_dtype.strip().lower()
    mapping: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if normalised not in mapping:
        msg: str = "inference_dtype must be one of: float32, bfloat16, float16"
        raise ValueError(msg)
    return mapping[normalised]


def load_reranker_model(
    model_name: str,
    *,
    revision: str | None = None,
    max_length: int = 512,
    dtype: torch.dtype = torch.bfloat16,
    num_labels: int = 1,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a sequence classification model and tokeniser for reranking.

    Parameters
    ----------
    model_name:
        HuggingFace model name or local path.
    revision:
        Optional git revision / branch.
    max_length:
        Maximum token length for the tokeniser.
    dtype:
        Model tensor dtype.
    num_labels:
        Number of output labels (1 for regression scoring).

    Returns
    -------
    ``(model, tokenizer)`` tuple.
    """
    extra_kwargs: dict[str, str] = {}
    if revision:
        extra_kwargs["revision"] = revision

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, **extra_kwargs
    )
    tokenizer.model_max_length = max_length

    config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, **extra_kwargs
    )
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        **extra_kwargs,
    )
    sanitize_position_ids_buffers(model)

    if getattr(model.config, "num_labels", 1) != num_labels:
        model.config.num_labels = num_labels
        model.config.id2label = {0: "SCORE"}
        model.config.label2id = {"SCORE": 0}

    return model, tokenizer


def short_model_name(model_name: str) -> str:
    """Return the last path component for HF model names."""
    if "://" in model_name or "/" not in model_name:
        return model_name
    return Path(model_name).name


def build_output_dir_name(
    *,
    model_name: str,
    seed: int,
    epoch: int,
    sampler_name: str,
    sample_preprocessing_name: str,
    val: str = "val",
    data: str = "data",
    callback: bool = False,
    num_neg_samples: int = 4,
    gradient_accumulation_steps: int = 2,
    use_expanded_pos: bool = False,
    warmup_ratio: bool = True,
    loss_mode: str,
) -> str:
    """Build output directory name following the established pattern."""
    _model_identifier: str = short_model_name(model_name).replace("/", "-")
    return (
        f"{_model_identifier}-{seed}-E{epoch}"
        f"-S{sampler_name}-SP{sample_preprocessing_name}"
        f"-{val}-{data}_data-CB{callback}-KN{num_neg_samples}"
        f"-GA{gradient_accumulation_steps}-ExPOS{use_expanded_pos}-warmup{warmup_ratio}"
        f"-{loss_mode}"
    )


def split_chunks[T](a: list[T], n: int) -> Generator[list[T], None, None]:
    """Split list *a* into *n* roughly equal chunks."""
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))
