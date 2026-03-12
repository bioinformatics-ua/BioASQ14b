from transformers import Trainer
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    nested_truncate,
    nested_numpify,
    nested_concat,
    find_batch_size,
)
from transformers.trainer_utils import (
    seed_worker,
    EvalLoopOutput,
    has_length,
    EvalPrediction,
    denumpify_detensorize,
    PredictionOutput,
    speed_metrics,
)
from transformers.utils import logging
import torch
import time
import math
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np


class PairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):

        # forward pass
        pos_doc_logits = model(
            **inputs.get("pos_inputs")
        ).logits  # before .get("logits")
        neg_doc_logits = model(
            **inputs.get("neg_inputs")
        ).logits  # before .get("logits")

        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.MarginRankingLoss(
            margin=1.0, size_average=None, reduce=None, reduction="mean"
        )
        loss = loss_fct(pos_doc_logits, neg_doc_logits, torch.ones_like(pos_doc_logits))
        return (loss, (pos_doc_logits, neg_doc_logits)) if return_outputs else loss
