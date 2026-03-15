"""
Custom Trainer classes for reranker training.

- PointwiseRerankerTrainer: BCE loss, batch from RankingCollator (inputs + labels)
- PairwiseRerankerTrainer: MarginRanking loss, pos_inputs + neg_inputs
- MultiNegativePairwiseRerankerTrainer: Multi-negative margin loss, pos_inputs + neg_inputs (list)
"""

from __future__ import annotations

import torch
from transformers import Trainer

from losses import bce_loss, margin_ranking_loss, multi_negative_margin_loss
from losses import extract_scores_from_logits


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
            labels = torch.tensor(
                labels, dtype=torch.float32, device=outputs.logits.device
            )
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
    """Trainer for multi-negative pairwise (1 pos + N negs) with sum of margin losses."""

    def __init__(self, margin: float = 1.0, reduction: str = "sum", **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.reduction = reduction

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

        loss = multi_negative_margin_loss(
            pos_scores,
            neg_scores_list,
            margin=self.margin,
            reduction=self.reduction,
        )
        return (loss, pos_outputs) if return_outputs else loss
