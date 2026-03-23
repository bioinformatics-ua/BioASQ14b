"""Custom Trainer classes for reranker training.

- :class:`PointwiseRerankerTrainer`: BCE loss, batch from RankingCollator
- :class:`PairwiseRerankerTrainer`: MarginRanking loss, pos + neg inputs
- :class:`MultiNegativePairwiseRerankerTrainer`: Multi-neg margin or InfoNCE loss

Refactored from ``refactored-trainer/trainer.py``.
"""

from __future__ import annotations

import torch
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from bioasq.phase_a.reranker.losses import (
    bce_loss,
    extract_scores_from_logits,
    margin_ranking_loss,
    multi_negative_infonce_loss,
    multi_negative_margin_loss,
)

# Internal type aliases for batch dicts coming from collators
type ModelInputs = dict[str, torch.Tensor]
type RankingBatch = dict[str, ModelInputs | list[int] | list[str]]
type PairwiseBatch = dict[str, ModelInputs]
type MultiNegBatch = dict[str, ModelInputs | list[ModelInputs]]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class EarlyStoppingOnGradNorm(TrainerCallback):
    """Stop training when grad_norm is below threshold for several logs.

    Useful for margin loss when model saturates (loss=0, grad=0) and no
    further learning occurs.
    """

    def __init__(
        self,
        grad_norm_threshold: float = 1e-6,
        patience: int = 5,
    ) -> None:
        self.grad_norm_threshold: float = grad_norm_threshold
        self.patience: int = patience
        self._low_grad_count: int = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: object,
    ) -> None:
        if logs is None or "grad_norm" not in logs:
            return

        grad_norm: float | None = logs.get("grad_norm")
        if grad_norm is not None and grad_norm <= self.grad_norm_threshold:
            self._low_grad_count += 1
            if self._low_grad_count >= self.patience:
                control.should_training_stop = True
                print(
                    f"[EarlyStoppingOnGradNorm] Stopping: grad_norm < "
                    f"{self.grad_norm_threshold} for {self._low_grad_count} "
                    f"consecutive logs"
                )
        else:
            self._low_grad_count = 0


# ---------------------------------------------------------------------------
# Trainer subclasses
# ---------------------------------------------------------------------------


class PointwiseRerankerTrainer(Trainer):
    """Trainer for pointwise (query, doc, label) with BCE loss."""

    def __init__(self, margin: float = 1.0, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.margin: float = margin

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: RankingBatch,
        return_outputs: bool = False,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, SequenceClassifierOutput]:
        labels: list[int] | list[str] | ModelInputs = inputs.pop("labels")  # type: ignore[assignment]
        inputs.pop("id", None)
        inputs.pop("doc_id", None)
        model_inputs: ModelInputs = inputs.pop("inputs")  # type: ignore[assignment]
        outputs: SequenceClassifierOutput = model(**model_inputs)
        if not isinstance(labels, torch.Tensor):
            labels_tensor: torch.Tensor = torch.tensor(
                labels, dtype=torch.float32, device=outputs.logits.device
            )
        else:
            labels_tensor = labels
        loss: torch.Tensor = bce_loss(outputs.logits, labels_tensor)
        return (loss, outputs) if return_outputs else loss


class PairwiseRerankerTrainer(Trainer):
    """Trainer for pairwise (pos, neg) with MarginRanking loss."""

    def __init__(self, margin: float = 1.0, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.margin: float = margin

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: PairwiseBatch,
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, None, None]:
        inputs = self._prepare_inputs(inputs)  # type: ignore[assignment]
        with torch.no_grad():
            loss, _ = self.compute_loss(model, inputs, return_outputs=True)  # type: ignore[misc]
        loss = loss.detach().mean()
        return (loss, None, None)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: PairwiseBatch,
        return_outputs: bool = False,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, SequenceClassifierOutput]:
        pos_inputs: ModelInputs = inputs["pos_inputs"]
        neg_inputs: ModelInputs = inputs["neg_inputs"]

        pos_outputs: SequenceClassifierOutput = model(**pos_inputs)
        neg_outputs: SequenceClassifierOutput = model(**neg_inputs)

        pos_scores: torch.Tensor = extract_scores_from_logits(pos_outputs.logits)
        neg_scores: torch.Tensor = extract_scores_from_logits(neg_outputs.logits)

        loss: torch.Tensor = margin_ranking_loss(
            pos_scores, neg_scores, margin=self.margin
        )
        return (loss, pos_outputs) if return_outputs else loss


class MultiNegativePairwiseRerankerTrainer(Trainer):
    """Trainer for multi-negative pairwise (1 pos + N negs).

    Supports ``loss_type``: ``"margin"`` (default) or ``"infonce"``.
    """

    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = "sum",
        loss_type: str = "margin",
        infonce_temperature: float = 0.05,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.margin: float = margin
        self.reduction: str = reduction
        self.loss_type: str = loss_type.lower()
        self.infonce_temperature: float = infonce_temperature

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: MultiNegBatch,
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, None, None]:
        inputs = self._prepare_inputs(inputs)  # type: ignore[assignment]
        with torch.no_grad():
            loss, _ = self.compute_loss(model, inputs, return_outputs=True)  # type: ignore[misc]
        loss = loss.detach().mean()
        return (loss, None, None)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: MultiNegBatch,
        return_outputs: bool = False,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, SequenceClassifierOutput]:
        pos_inputs: ModelInputs = inputs["pos_inputs"]  # type: ignore[assignment]
        neg_inputs: list[ModelInputs] = inputs["neg_inputs"]  # type: ignore[assignment]

        pos_outputs: SequenceClassifierOutput = model(**pos_inputs)
        pos_scores: torch.Tensor = extract_scores_from_logits(pos_outputs.logits)

        neg_scores_list: list[torch.Tensor] = []
        for neg_batch in neg_inputs:
            neg_outputs: SequenceClassifierOutput = model(**neg_batch)
            neg_scores_list.append(extract_scores_from_logits(neg_outputs.logits))

        loss: torch.Tensor
        if self.loss_type == "infonce":
            loss = multi_negative_infonce_loss(
                pos_scores,
                neg_scores_list,
                temperature=self.infonce_temperature,
            )
        else:
            loss = multi_negative_margin_loss(
                pos_scores,
                neg_scores_list,
                margin=self.margin,
                reduction=self.reduction,
            )
        return (loss, pos_outputs) if return_outputs else loss
