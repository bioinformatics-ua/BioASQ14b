"""SPLARE online query encoder — encodes queries at inference time.

Manages a singleton SPLARE model instance for efficient repeated encoding.
The model is loaded lazily on first use.
"""

from __future__ import annotations

import os

from bioasq.phase_a.splare.model import SplareConfig, SplareModel

_MODEL: SplareModel | None = None


def _default_config() -> SplareConfig:
    """Build config from environment variables or defaults."""
    return SplareConfig(
        backbone_name_or_path=os.environ.get("BIOASQ_SPLARE_BACKBONE", "meta-llama/Llama-3.1-8B"),
        sae_name_or_path=os.environ.get("BIOASQ_SPLARE_SAE_PATH", ""),
        sae_layer=int(os.environ.get("BIOASQ_SPLARE_SAE_LAYER", "26")),
        query_topk=int(os.environ.get("BIOASQ_SPLARE_QUERY_TOPK", "40")),
        doc_topk=int(os.environ.get("BIOASQ_SPLARE_DOC_TOPK", "400")),
    )


def get_splare_model() -> SplareModel:
    """Return the singleton SPLARE model, loading it on first call."""
    global _MODEL
    if _MODEL is None:
        device = os.environ.get("BIOASQ_SPLARE_DEVICE", "cuda")
        config = _default_config()

        # Check for LoRA adapter checkpoint
        lora_path = os.environ.get("BIOASQ_SPLARE_LORA_PATH")
        _MODEL = SplareModel(config).load(device)

        if lora_path:
            from peft import PeftModel

            _MODEL._backbone = PeftModel.from_pretrained(_MODEL._backbone, lora_path)
            _MODEL._backbone.eval()
            print(f"Loaded SPLARE LoRA adapter from {lora_path}")

    return _MODEL


def encode_queries_splare(
    texts: list[str],
) -> list[tuple[list[int], list[float]]]:
    """Encode query texts into SPLARE sparse vectors.

    Returns a list of ``(indices, values)`` tuples, one per query.
    """
    model = get_splare_model()
    return model.encode_queries(texts)


def unload_splare_model() -> None:
    """Release the SPLARE model to free GPU memory."""
    global _MODEL
    if _MODEL is not None:
        import torch

        del _MODEL
        _MODEL = None
        torch.cuda.empty_cache()
