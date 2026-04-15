"""
Loss functions for reranker training.

- Pointwise BCE: (query, doc, label) with BCEWithLogitsLoss
- Pairwise MarginRanking: score_pos > score_neg with configurable margin
- Multi-Negative Pairwise: 1 pos + N negs; sum/average margin loss over all pos-neg pairs
- InfoNCE: contrastive loss over 1 pos + N negs; always has gradient (no saturation)
"""

import torch
import torch.nn.functional as F


def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy for pointwise relevance (0/1 labels)."""
    loss_fct = torch.nn.BCEWithLogitsLoss()
    return loss_fct(logits.view(-1), labels.float().view(-1))


def margin_ranking_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Margin ranking loss: wants pos_scores > neg_scores by at least margin."""
    loss_fct = torch.nn.MarginRankingLoss(margin=margin)
    target = torch.ones_like(pos_scores)
    return loss_fct(pos_scores, neg_scores, target)


def multi_negative_margin_loss(
    pos_scores: torch.Tensor,
    neg_scores_list: list[torch.Tensor],
    margin: float = 1.0,
    reduction: str = "sum",
) -> torch.Tensor:
    """
    Multi-negative pairwise margin loss.

    For each (pos, neg_i) pair, compute max(0, margin - (pos_score - neg_score)).
    Then sum (or average) over all negative pairs.

    Args:
        pos_scores: [batch_size]
        neg_scores_list: list of [batch_size] tensors, one per negative
        margin: margin for MarginRankingLoss
        reduction: "sum" or "mean" over negative pairs
    """
    total_loss = pos_scores.new_zeros(1)
    count = 0
    for neg_scores in neg_scores_list:
        total_loss = total_loss + margin_ranking_loss(pos_scores, neg_scores, margin)
        count += 1
    if count == 0:
        return total_loss
    if reduction == "mean":
        total_loss = total_loss / count
    return total_loss


def multi_negative_infonce_loss(
    pos_scores: torch.Tensor,
    neg_scores_list: list[torch.Tensor],
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    InfoNCE / multiple negatives cross-entropy loss.

    Treats the positive as the correct class among (1 pos + N negs). Always has
    gradient (no saturation like hinge), encouraging larger score separation.

    loss = -log(exp(pos/T) / (exp(pos/T) + sum(exp(neg_i/T))))

    Args:
        pos_scores: [batch_size]
        neg_scores_list: list of [batch_size] tensors, one per negative
        temperature: scaling (smaller = sharper, larger gradient magnitude)
    """
    if not neg_scores_list:
        return pos_scores.new_zeros(1)

    # Stack: [batch_size, 1 + num_negs]; positive is index 0
    scores = torch.stack([pos_scores] + neg_scores_list, dim=1) / temperature
    target = torch.zeros(
        pos_scores.shape[0],
        dtype=torch.long,
        device=pos_scores.device,
    )
    return F.cross_entropy(scores, target)


def extract_scores_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Extract scalar score from logits for ranking.
    Handles both num_labels==1 (logits.view(-1)) and num_labels==2 (logits[:, 1]).
    """
    if logits.shape[-1] == 1:
        return logits.view(-1)
    return logits[:, 1]
