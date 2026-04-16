"""Lightweight reranker wrapper used by Context-1 search tools."""

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import torch

from bioasq.phase_a.reranker.evaluation import extract_scores
from bioasq.phase_a.reranker.model import load_reranker_model, resolve_inference_dtype

from .types import CorpusDocument

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class Context1Reranker:
    """Batch reranking wrapper for PMID-level document candidates."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 16,
        max_length: int = 1_024,
        device: str = "cuda",
        dtype: str = "bfloat16",
        invert_scores: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.effective_max_length = max_length
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.dtype = resolve_inference_dtype(dtype)
        self.invert_scores = invert_scores
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    def load(self) -> None:
        """Load the reranker model if it is not already loaded."""

        if self._model is not None and self._tokenizer is not None:
            return
        self._model, self._tokenizer = load_reranker_model(
            self.model_name,
            max_length=self.max_length,
            dtype=self.dtype,
        )
        self.effective_max_length = int(self._tokenizer.model_max_length)
        model = cast("torch.nn.Module", self._model)
        self._model = cast("PreTrainedModel", model.to(self.device))
        self._model.eval()

    def score(self, query: str, documents: Sequence[CorpusDocument]) -> list[CorpusDocument]:
        """Rerank article candidates for one query."""

        if not documents:
            return []
        self.load()
        assert self._model is not None
        assert self._tokenizer is not None

        scored: list[CorpusDocument] = []
        for start in range(0, len(documents), self.batch_size):
            batch = list(documents[start : start + self.batch_size])
            encoded = self._tokenizer(
                [query for _ in batch],
                [document.text for document in batch],
                padding=True,
                truncation="only_second",
                max_length=self.effective_max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
                if isinstance(value, torch.Tensor)
            }
            with torch.inference_mode():
                if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        logits = self._model(**encoded).logits
                else:
                    logits = self._model(**encoded).logits
            scores = extract_scores(logits).detach().float().cpu().tolist()
            for document, raw_score in zip(batch, scores, strict=True):
                relevance = -float(raw_score) if self.invert_scores else float(raw_score)
                scored.append(replace(document, score=relevance))

        return sorted(scored, key=lambda document: document.score, reverse=True)


def _normalize_ensemble_scores(documents: Sequence[CorpusDocument]) -> dict[str, float]:
    if not documents:
        return {}

    raw_scores = [document.score for document in documents]
    score_min = min(raw_scores)
    score_max = max(raw_scores)
    if score_max <= score_min:
        return {document.pmid: 0.0 for document in documents}

    scale = score_max - score_min
    return {document.pmid: (document.score - score_min) / scale for document in documents}


def ensemble_rerank_documents(
    query: str,
    documents: Sequence[CorpusDocument],
    rerankers: Sequence[Context1Reranker],
) -> list[CorpusDocument]:
    """Average min-max normalized scores from one or more rerankers."""

    if not rerankers:
        raise ValueError("At least one reranker is required.")
    if not documents:
        return []
    if len(rerankers) == 1:
        return rerankers[0].score(query, documents)

    ensembled_scores = {document.pmid: 0.0 for document in documents}

    for reranker in rerankers:
        scored = reranker.score(query, documents)
        normalized_scores = _normalize_ensemble_scores(scored)
        for pmid, normalized_score in normalized_scores.items():
            ensembled_scores[pmid] = ensembled_scores.get(pmid, 0.0) + normalized_score

    reranker_count = float(len(rerankers))
    return sorted(
        [
            replace(document, score=ensembled_scores[document.pmid] / reranker_count)
            for document in documents
        ],
        key=lambda document: (-document.score, document.pmid),
    )
