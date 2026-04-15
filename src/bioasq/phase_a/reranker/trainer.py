"""
Custom Trainer classes for reranker training.

- PointwiseRerankerTrainer: BCE loss, batch from RankingCollator (inputs + labels)
- PairwiseRerankerTrainer: MarginRanking loss, pos_inputs + neg_inputs
- MultiNegativePairwiseRerankerTrainer: Multi-negative margin loss, pos_inputs + neg_inputs (list)
"""

import torch
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState

from bioasq.phase_a.reranker.losses import (
    bce_loss,
    extract_scores_from_logits,
    margin_ranking_loss,
    multi_negative_infonce_loss,
    multi_negative_margin_loss,
)


class EarlyStoppingOnGradNorm(TrainerCallback):
    """Stop training when grad_norm is below threshold for several consecutive logs.

    Useful for margin loss when model saturates (loss=0, grad=0) and no further
    learning occurs.
    """

    def __init__(
        self,
        grad_norm_threshold: float = 1e-6,
        patience: int = 5,
    ):
        self.grad_norm_threshold = grad_norm_threshold
        self.patience = patience
        self._low_grad_count = 0

    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ):
        if logs is None or "grad_norm" not in logs:
            return

        grad_norm = logs.get("grad_norm")
        if grad_norm is not None and grad_norm <= self.grad_norm_threshold:
            self._low_grad_count += 1
            if self._low_grad_count >= self.patience:
                control.should_training_stop = True
                print(
                    f"[EarlyStoppingOnGradNorm] Stopping: grad_norm < {self.grad_norm_threshold} "
                    f"for {self._low_grad_count} consecutive logs"
                )
        else:
            self._low_grad_count = 0


class PointwiseRerankerTrainer(Trainer):
    """Trainer for pointwise (query, doc, label) with BCE loss."""

    def __init__(self, margin: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        inputs.pop("id", None)
        inputs.pop("doc_id", None)
        model_inputs = inputs.pop("inputs")
        outputs = model(**model_inputs)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32, device=outputs.logits.device)
        loss = bce_loss(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


class PairwiseRerankerTrainer(Trainer):
    """Trainer for pairwise (pos, neg) with MarginRanking loss."""

    def __init__(self, margin: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Use compute_loss for eval; base Trainer would pass inputs to model() which fails for pairwise."""
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, _ = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach().mean()
        return (loss, None, None)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_inputs = inputs["pos_inputs"]
        neg_inputs = inputs["neg_inputs"]

        pos_outputs = model(**pos_inputs)
        neg_outputs = model(**neg_inputs)

        pos_scores = extract_scores_from_logits(pos_outputs.logits)
        neg_scores = extract_scores_from_logits(neg_outputs.logits)

        loss = margin_ranking_loss(pos_scores, neg_scores, margin=self.margin)
        return (loss, pos_outputs) if return_outputs else loss


class MultiNegativePairwiseRerankerTrainer(Trainer):
    """Trainer for multi-negative pairwise (1 pos + N negs).

    Supports loss_type: "margin" (default) or "infonce".
    """

    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = "sum",
        loss_type: str = "margin",
        infonce_temperature: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.reduction = reduction
        self.loss_type = loss_type.lower()
        self.infonce_temperature = infonce_temperature

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Use compute_loss for eval; base Trainer would pass inputs to model() which fails for pairwise."""
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, _ = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach().mean()
        return (loss, None, None)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_inputs = inputs["pos_inputs"]
        neg_inputs: list[dict] = inputs["neg_inputs"]

        pos_outputs = model(**pos_inputs)
        pos_scores = extract_scores_from_logits(pos_outputs.logits)

        neg_scores_list: list[torch.Tensor] = []
        for neg_batch in neg_inputs:
            neg_outputs = model(**neg_batch)
            neg_scores_list.append(extract_scores_from_logits(neg_outputs.logits))

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
