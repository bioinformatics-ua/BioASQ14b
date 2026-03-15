"""
Factory functions to build samplers, preprocessors, collators, iterators, and trainers
from CLI/config names.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from collator import (
    MultiNegativePairwiseCollator,
    PairwiseCollator,
    RankingCollator,
)
from data import (
    BioASQMultiNegativePairwiseIterator,
    BioASQPairwiseIterator,
    BioASQPointwiseIterator,
)
from sample_preprocessing import BasicSamplePreprocessing
from sampler import (
    BasicSampler,
    BasicV2Sampler,
    ExponentialWeightSampler,
    ShifterSampler,
)
from trainer import (
    MultiNegativePairwiseRerankerTrainer,
    PairwiseRerankerTrainer,
    PointwiseRerankerTrainer,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, TokenizersBackend
    from transformers import Trainer


def get_sampler(
    name: str,
    **kwargs: object,
) -> type[BasicSampler]:
    """
    Return sampler class by name.

    - basic: BasicSampler
    - basicv2: BasicV2Sampler (falls back to negs from other questions if empty)
    - exponential: ExponentialWeightSampler (requires use_expanded_pos)
    - shifter: ShifterSampler (requires max_epoch in kwargs)
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
            raise ValueError(f"Unknown sampler: {name!r}")


def get_preprocessor(
    name: str,
    tokenizer: PreTrainedTokenizerBase | TokenizersBackend,
    max_length: int = 512,
    **kwargs: object,
) -> BasicSamplePreprocessing:
    """
    Return preprocessor instance by name.

    - basic: BasicSamplePreprocessing (query+doc concatenation)
    """
    match name.lower():
        case "basic":
            return BasicSamplePreprocessing(tokenizer, model_max_length=max_length)
        case _:
            raise ValueError(f"Unknown preprocessor: {name!r}")


def get_collator(
    mode: str,
    tokenizer: PreTrainedTokenizerBase | TokenizersBackend,
    **kwargs: object,
) -> RankingCollator | PairwiseCollator | MultiNegativePairwiseCollator:
    """
    Return data collator by training mode.

    - pointwise: RankingCollator
    - pairwise: PairwiseCollator
    - multi_neg_pairwise: MultiNegativePairwiseCollator
    """
    match mode.lower():
        case "pointwise":
            return RankingCollator(tokenizer=tokenizer, **kwargs)
        case "pairwise":
            return PairwiseCollator(tokenizer=tokenizer)
        case "multi_neg_pairwise":
            return MultiNegativePairwiseCollator(tokenizer=tokenizer)
        case _:
            raise ValueError(f"Unknown mode: {mode!r}")


def get_iterator(
    mode: str,
    sample_preprocessing: BasicSamplePreprocessing,
    sampler_cls: type[BasicSampler],
    num_neg_samples: int = 1,
    sampler_kwargs: dict[str, object] | None = None,
    **kwargs: object,
) -> (
    BioASQPointwiseIterator
    | BioASQPairwiseIterator
    | BioASQMultiNegativePairwiseIterator
):
    """
    Return iterator instance by training mode.

    - pointwise: BioASQPointwiseIterator
    - pairwise: BioASQPairwiseIterator
    - multi_neg_pairwise: BioASQMultiNegativePairwiseIterator

    sampler_kwargs: passed to sampler when created (e.g. max_epoch for ShifterSampler)
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
            raise ValueError(f"Unknown mode: {mode!r}")


def get_trainer_cls(mode: str) -> type[Trainer]:
    """
    Return trainer class by training mode.

    - pointwise: PointwiseRerankerTrainer
    - pairwise: PairwiseRerankerTrainer
    - multi_neg_pairwise: MultiNegativePairwiseRerankerTrainer
    """
    match mode.lower():
        case "pointwise":
            return PointwiseRerankerTrainer
        case "pairwise":
            return PairwiseRerankerTrainer
        case "multi_neg_pairwise":
            return MultiNegativePairwiseRerankerTrainer
        case _:
            raise ValueError(f"Unknown mode: {mode!r}")
