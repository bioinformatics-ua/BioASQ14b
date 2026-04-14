"""SPLARE model: Llama backbone + SAE encoder + SPLADE-style pooling.

Produces sparse representations in the SAE latent space (131k dims) from
input text sequences. Uses a frozen SAE encoder attached at an intermediate
layer of the Llama backbone, with LoRA adapters for fine-tuning.

Key components:
  - Llama-3.1-8B backbone with bidirectional attention
  - Llama Scope SAE encoder (frozen) at layer ``sae_layer``
  - SPLADE-pool: ``u_j = max_i log(1 + ReLU(w_ij))``
  - Top-K pooling at inference for efficiency control
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerBase

# ---------------------------------------------------------------------------
# SAE encoder (lightweight wrapper around pre-trained weights)
# ---------------------------------------------------------------------------


class SAEEncoder(nn.Module):
    """Sparse Autoencoder encoder: z = f(W_enc @ x + b_enc).

    Only uses the encoder half — we do not reconstruct; we just extract
    sparse latent features from hidden states.

    Supports both plain ReLU and JumpReLU (Llama Scope default) activation.
    JumpReLU applies ``ReLU(z - threshold)`` for additional sparsity.
    """

    def __init__(
        self,
        d_model: int,
        n_features: int,
        *,
        act_fn: str = "jumprelu",
        jump_relu_threshold: float = 0.0,
        norm_activation: str | None = None,
        activation_norm: float | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.act_fn = act_fn
        self.jump_relu_threshold = jump_relu_threshold
        self.norm_activation = norm_activation
        self.activation_norm = activation_norm
        self.W_enc = nn.Linear(d_model, n_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode hidden states → sparse latent activations."""
        # Optional input normalisation (Llama Scope uses dataset-wise norm)
        if self.norm_activation == "dataset-wise" and self.activation_norm:
            x = x * (self.activation_norm / x.norm(dim=-1, keepdim=True).clamp(min=1e-8))

        z = self.W_enc(x)

        if self.act_fn == "jumprelu":
            return F.relu(z - self.jump_relu_threshold)
        return F.relu(z)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> SAEEncoder:
        """Load a Llama Scope SAE checkpoint.

        Supports the ``OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x`` format with
        ``checkpoints/final.safetensors`` + ``hyperparams.json`` as well as
        raw ``state_dict`` checkpoints.  Only encoder weights are loaded.
        """
        import json

        import safetensors.torch

        path = Path(path) if not isinstance(path, Path) else path

        # Resolve file paths — accept either a directory or a .safetensors file
        if path.is_dir():
            # Llama Scope layout: <layer_dir>/checkpoints/final.safetensors
            st_file = path / "checkpoints" / "final.safetensors"
            if not st_file.exists():
                # Try flat layout
                candidates = list(path.glob("*.safetensors"))
                if not candidates:
                    msg = f"No .safetensors files found in {path}"
                    raise FileNotFoundError(msg)
                st_file = candidates[0]
            hp_file = path / "hyperparams.json"
        else:
            st_file = path
            hp_file = path.parent / "hyperparams.json"

        state = safetensors.torch.load_file(str(st_file))

        # Load hyperparams if available
        act_fn = "jumprelu"
        threshold = 0.0
        norm_activation = None
        activation_norm = None

        if hp_file.exists():
            hp = json.loads(hp_file.read_text())
            act_fn = hp.get("act_fn", "jumprelu")
            threshold = hp.get("jump_relu_threshold", 0.0)
            norm_activation = hp.get("norm_activation")
            avg_norms = hp.get("dataset_average_activation_norm", {})
            activation_norm = avg_norms.get("in")

        # Try various key naming conventions
        enc_weight = (
            state.get("encoder.weight")
            or state.get("W_enc.weight")
            or state.get("W_enc")
        )
        enc_bias = (
            state.get("encoder.bias")
            or state.get("W_enc.bias")
            or state.get("b_enc")
        )

        if enc_weight is None:
            msg = f"Cannot find encoder weight in {st_file}. Keys: {list(state.keys())}"
            raise ValueError(msg)

        n_features, d_model = enc_weight.shape
        encoder = cls(
            d_model,
            n_features,
            act_fn=act_fn,
            jump_relu_threshold=threshold,
            norm_activation=norm_activation,
            activation_norm=activation_norm,
        )
        encoder.W_enc.weight = nn.Parameter(enc_weight)
        if enc_bias is not None:
            encoder.W_enc.bias = nn.Parameter(enc_bias)
        return encoder


# ---------------------------------------------------------------------------
# SPLARE config
# ---------------------------------------------------------------------------


@dataclass
class SplareConfig:
    """Configuration for the SPLARE model."""

    backbone_name_or_path: str = "meta-llama/Llama-3.1-8B"
    sae_name_or_path: str = ""
    sae_layer: int = 26
    sae_width: int = 131072  # 131k features
    query_topk: int = 40
    doc_topk: int = 400
    max_length: int = 512
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    bidirectional: bool = True
    dtype: str = "bfloat16"


# ---------------------------------------------------------------------------
# Bidirectional attention patch for Llama
# ---------------------------------------------------------------------------


def _patch_llama_bidirectional(model: nn.Module) -> None:
    """Replace causal attention masks with full bidirectional attention.

    Following LLM2Vec / SPLARE: simply disabling the causal mask in the
    attention layers allows every token to attend to all others, which is
    critical for pooling over the full sequence.
    """
    for module in model.modules():
        if hasattr(module, "is_causal"):
            module.is_causal = False


# ---------------------------------------------------------------------------
# SPLARE model
# ---------------------------------------------------------------------------


class SplareModel(nn.Module):
    """SPLARE: Llama backbone (up to ``sae_layer``) + frozen SAE encoder + SPLADE-pool."""

    def __init__(self, config: SplareConfig) -> None:
        super().__init__()
        self.config = config
        self._backbone: nn.Module | None = None
        self._sae: SAEEncoder | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    @property
    def backbone(self) -> nn.Module:
        if self._backbone is None:
            raise RuntimeError("Call .load() first")
        return self._backbone

    @property
    def sae(self) -> SAEEncoder:
        if self._sae is None:
            raise RuntimeError("Call .load() first")
        return self._sae

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            raise RuntimeError("Call .load() first")
        return self._tokenizer

    def load(self, device: torch.device | str = "cuda") -> SplareModel:
        """Load backbone, SAE, and tokenizer. Returns self for chaining."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.config.dtype)

        # Load backbone
        self._backbone = AutoModelForCausalLM.from_pretrained(
            self.config.backbone_name_or_path,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
        )

        if self.config.bidirectional:
            _patch_llama_bidirectional(self._backbone)

        self._backbone.to(device)
        self._backbone.eval()

        # Load SAE encoder (frozen)
        if self.config.sae_name_or_path:
            self._sae = SAEEncoder.from_pretrained(self.config.sae_name_or_path)
        else:
            # Try loading from HuggingFace hub
            from huggingface_hub import hf_hub_download

            sae_repo = "fnlp/Llama-Scope-8B-131k"
            sae_file = f"layer{self.config.sae_layer}/model.safetensors"
            local_path = hf_hub_download(repo_id=sae_repo, filename=sae_file)
            self._sae = SAEEncoder.from_pretrained(local_path)

        self._sae.to(device).to(dtype)
        self._sae.eval()
        for p in self._sae.parameters():
            p.requires_grad = False

        # Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.backbone_name_or_path,
            padding_side="right",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        return self

    def apply_lora(self) -> SplareModel:
        """Attach LoRA adapters to the backbone for fine-tuning.

        SAE parameters remain frozen. Only backbone adapter weights are
        trainable.
        """
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._backbone = get_peft_model(self._backbone, lora_config)
        self._backbone.print_trainable_parameters()
        return self

    def _extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run backbone up to ``sae_layer`` and return hidden states ``(B, L, d_model)``."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # hidden_states is a tuple of (n_layers+1) tensors; index sae_layer
        return outputs.hidden_states[self.config.sae_layer]

    def _splade_pool(
        self,
        sae_output: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """SPLADE pooling: u_j = max_i log(1 + ReLU(w_ij)), masked by attention.

        Args:
            sae_output: Sparse activations ``(B, L, W)`` from the SAE encoder.
            attention_mask: ``(B, L)`` with 1 for real tokens, 0 for padding.

        Returns:
            Pooled sparse representation ``(B, W)``.
        """
        # log(1 + ReLU(x)) — ReLU is already applied by SAE encoder
        saturated = torch.log1p(sae_output)  # (B, L, W)

        # Mask padding tokens by setting them to -inf before max-pool
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        saturated = saturated * mask  # zeros out padding positions

        # Max-pool over sequence dimension
        pooled, _ = saturated.max(dim=1)  # (B, W)
        return pooled

    def _topk_pool(self, pooled: torch.Tensor, k: int) -> torch.Tensor:
        """Keep only top-k values per vector, zero the rest."""
        if k <= 0 or k >= pooled.shape[-1]:
            return pooled
        values, indices = pooled.topk(k, dim=-1)
        sparse = torch.zeros_like(pooled)
        sparse.scatter_(-1, indices, values)
        return sparse

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        *,
        topk: int | None = None,
        batch_size: int = 8,
        show_progress: bool = False,
    ) -> list[tuple[list[int], list[float]]]:
        """Encode texts into sparse representations.

        Returns a list of ``(indices, values)`` tuples — one per input text.
        This is the format expected by Qdrant's ``SparseVector``.
        """
        from tqdm import tqdm as tqdm_bar

        all_results: list[tuple[list[int], list[float]]] = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm_bar(iterator, desc="SPLARE encode", total=len(texts) // batch_size + 1)

        for start in iterator:
            batch_texts = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.backbone.device)

            hidden = self._extract_hidden_states(inputs["input_ids"], inputs["attention_mask"])
            sae_out = self.sae(hidden)
            pooled = self._splade_pool(sae_out, inputs["attention_mask"])

            if topk is not None:
                pooled = self._topk_pool(pooled, topk)

            # Convert to sparse format
            for vec in pooled:
                nonzero = vec.nonzero(as_tuple=True)[0]
                indices = nonzero.cpu().tolist()
                values = vec[nonzero].cpu().float().tolist()
                all_results.append((indices, values))

        return all_results

    def encode_queries(
        self, texts: list[str], *, batch_size: int = 8, show_progress: bool = False,
    ) -> list[tuple[list[int], list[float]]]:
        """Encode queries with Top-K pooling (default k=40)."""
        return self.encode(
            texts, topk=self.config.query_topk, batch_size=batch_size, show_progress=show_progress,
        )

    def encode_documents(
        self, texts: list[str], *, batch_size: int = 4, show_progress: bool = False,
    ) -> list[tuple[list[int], list[float]]]:
        """Encode documents with Top-K pooling (default k=400)."""
        return self.encode(
            texts, topk=self.config.doc_topk, batch_size=batch_size, show_progress=show_progress,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        topk: int | None = None,
    ) -> torch.Tensor:
        """Full forward pass returning pooled sparse vectors ``(B, W)``."""
        hidden = self._extract_hidden_states(input_ids, attention_mask)
        sae_out = self.sae(hidden)
        pooled = self._splade_pool(sae_out, attention_mask)
        if topk is not None:
            pooled = self._topk_pool(pooled, topk)
        return pooled
