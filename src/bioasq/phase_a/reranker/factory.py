"""Factory functions for building reranker components from config names.

Provides registry-pattern factories for samplers, preprocessors,
collators, iterators, and trainers.

Refactored from ``refactored-trainer/factory.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bioasq.phase_a.reranker.collator import (
    MultiNegativePairwiseCollator,
    PairwiseCollator,
    RankingCollator,
)
from bioasq.phase_a.reranker.data import (
    BioASQMultiNegativePairwiseIterator,
    BioASQPairwiseIterator,
    BioASQPointwiseIterator,
)
from bioasq.phase_a.reranker.preprocessing import (
    BasicSamplePreprocessing,
    NemotronSamplePreprocessing,
)
from bioasq.phase_a.reranker.sampler import (
    BasicSampler,
    BasicV2Sampler,
    ExponentialWeightSampler,
    ShifterSampler,
)
from bioasq.phase_a.reranker.trainer import (
    MultiNegativePairwiseRerankerTrainer,
    PairwiseRerankerTrainer,
    PointwiseRerankerTrainer,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, Trainer


def get_sampler(name: str) -> type[BasicSampler]:
    """Return sampler *class* by name.

    Names: ``basic``, ``basicv2``, ``exponential``, ``shifter``.
    """
    match name.lower():
        case "basic":
            return BasicSampler
        case "basicv2":
            return BasicV2Sampler
        case "exponential":
            return ExponentialWeightSampler
        case "shifter":
            return ShifterSampler
        case _:
            msg: str = f"Unknown sampler: {name!r}"
            raise ValueError(msg)


def get_preprocessor(
    name: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> BasicSamplePreprocessing | NemotronSamplePreprocessing:
    """Return preprocessor *instance* by name.

    Names: ``basic``, ``nemotron``.
    """
    match name.lower():
        case "basic":
            return BasicSamplePreprocessing(tokenizer, model_max_length=max_length)
        case "nemotron":
            return NemotronSamplePreprocessing(tokenizer, model_max_length=max_length)
        case _:
            msg: str = f"Unknown preprocessor: {name!r}"
            raise ValueError(msg)


def get_collator(
    mode: str,
    tokenizer: PreTrainedTokenizerBase,
    **kwargs: object,
) -> RankingCollator | PairwiseCollator | MultiNegativePairwiseCollator:
    """Return data collator *instance* by training mode.

    Modes: ``pointwise``, ``pairwise``, ``multi_neg_pairwise``.
    """
    match mode.lower():
        case "pointwise":
            return RankingCollator(tokenizer=tokenizer, **kwargs)
        case "pairwise":
            return PairwiseCollator(tokenizer=tokenizer)
        case "multi_neg_pairwise":
            return MultiNegativePairwiseCollator(tokenizer=tokenizer)
        case _:
            msg: str = f"Unknown mode: {mode!r}"
            raise ValueError(msg)


def get_iterator(
    mode: str,
    sample_preprocessing: BasicSamplePreprocessing | NemotronSamplePreprocessing,
    sampler_cls: type[BasicSampler],
    num_neg_samples: int = 1,
    sampler_kwargs: dict[str, object] | None = None,
    **kwargs: object,
) -> (
    BioASQPointwiseIterator
    | BioASQPairwiseIterator
    | BioASQMultiNegativePairwiseIterator
):
    """Return iterator *instance* by training mode.

    Modes: ``pointwise``, ``pairwise``, ``multi_neg_pairwise``.
    """
    sampler_kwargs = sampler_kwargs or {}

    match mode.lower():
        case "pointwise":
            return BioASQPointwiseIterator(
                sample_preprocessing=sample_preprocessing,
                sampler_class_type=sampler_cls,
                num_neg_samples=num_neg_samples,
                sampler_kwargs=sampler_kwargs,
                **kwargs,
            )
        case "pairwise":
            return BioASQPairwiseIterator(
                sample_preprocessing=sample_preprocessing,
                sampler_class_type=sampler_cls,
                num_neg_samples=num_neg_samples,
                sampler_kwargs=sampler_kwargs,
                **kwargs,
            )
        case "multi_neg_pairwise":
            return BioASQMultiNegativePairwiseIterator(
                sample_preprocessing=sample_preprocessing,
                sampler_class=sampler_cls,
                num_neg_samples=num_neg_samples,
                sampler_kwargs=sampler_kwargs,
                **kwargs,
            )
        case _:
            msg: str = f"Unknown mode: {mode!r}"
            raise ValueError(msg)


def get_trainer_cls(mode: str) -> type[Trainer]:
    """Return trainer *class* by training mode.

    Modes: ``pointwise``, ``pairwise``, ``multi_neg_pairwise``.
    """
    match mode.lower():
        case "pointwise":
            return PointwiseRerankerTrainer  # type: ignore[return-value]
        case "pairwise":
            return PairwiseRerankerTrainer  # type: ignore[return-value]
        case "multi_neg_pairwise":
            return MultiNegativePairwiseRerankerTrainer  # type: ignore[return-value]
        case _:
            msg: str = f"Unknown mode: {mode!r}"
            raise ValueError(msg)
