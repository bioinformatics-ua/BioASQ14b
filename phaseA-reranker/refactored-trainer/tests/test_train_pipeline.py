"""
Smoke and integration tests for the train pipeline.

- Factory: get_sampler, get_preprocessor, get_collator, get_iterator, get_trainer_cls
- sampler_kwargs flow: main -> factory -> iterator -> sampler (ShifterSampler needs max_epoch)
- Minimal training step with mock data
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from collator import PairwiseCollator, RankingCollator
from data import (
    BioASQDataset,
    BioASQMultiNegativePairwiseIterator,
    BioASQPairwiseIterator,
    BioASQPointwiseIterator,
    create_bioASQ_datasets,
)
from factory import (
    get_collator,
    get_iterator,
    get_preprocessor,
    get_sampler,
    get_trainer_cls,
)
from sample_preprocessing import BasicSamplePreprocessing
from sampler import BasicSampler, ShifterSampler
from trainer import PairwiseRerankerTrainer, PointwiseRerankerTrainer
from utils import create_training_config, set_seed
from aliases import SliceDataset

BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "config" / "train_config.yaml"


def create_mock_slice_dataset() -> SliceDataset:
    """Mock SliceDataset: {q_id: {0: [neg_docs], 1: [pos_docs], "question": str}}."""
    return {
        "q1": {
            "question": "What is diabetes?",
            0: [
                {"id": "n1", "text": "Diabetes is rare."},
                {"id": "n2", "text": "Cats are furry."},
            ],
            1: [{"id": "p1", "text": "Diabetes is a metabolic disease."}],
        },
        "q2": {
            "question": "What is hypertension?",
            0: [
                {"id": "n3", "text": "Blood pressure varies."},
                {"id": "n4", "text": "Exercise helps."},
            ],
            1: [{"id": "p2", "text": "Hypertension is high blood pressure."}],
        },
    }


def create_mock_jsonl_files() -> tuple[Path, Path]:
    """Create temporary JSONL files for create_bioASQ_datasets."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as pos_f:
        pos_f.write(
            '{"id": "q1", "body": "What is diabetes?", "documents": [{"id": "p1", "text": "Diabetes is a metabolic disease."}]}\n'
        )
        pos_f.write(
            '{"id": "q2", "body": "What is hypertension?", "documents": [{"id": "p2", "text": "Hypertension is high blood pressure."}]}\n'
        )
        pos_path = Path(pos_f.name)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as all_f:
        all_f.write(
            '{"id": "q1", "neg_docs": [{"id": "n1", "text": "Diabetes is rare.", "score": 1}, {"id": "n2", "text": "Cats are furry.", "score": 0.5}]}\n'
        )
        all_f.write(
            '{"id": "q2", "neg_docs": [{"id": "n3", "text": "Blood pressure varies.", "score": 1}, {"id": "n4", "text": "Exercise helps.", "score": 0.5}]}\n'
        )
        all_path = Path(all_f.name)

    return pos_path, all_path


class TestFactory:
    """Test factory functions."""

    def test_get_sampler(self) -> None:
        assert get_sampler("basic") is BasicSampler
        assert get_sampler("basicv2") is not BasicSampler
        assert get_sampler("shifter") is ShifterSampler

    def test_get_preprocessor(self) -> None:
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        prep = get_preprocessor("basic", tok, max_length=128)
        assert isinstance(prep, BasicSamplePreprocessing)
        assert prep.model_max_length == 128

    def test_get_collator(self) -> None:
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        assert isinstance(get_collator("pointwise", tok), RankingCollator)
        assert isinstance(get_collator("pairwise", tok), PairwiseCollator)

    def test_get_iterator(self) -> None:
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        prep = get_preprocessor("basic", tok, max_length=64)
        it = get_iterator(
            "pairwise", prep, BasicSampler, num_neg_samples=2, sampler_kwargs={}
        )
        assert isinstance(it, BioASQPairwiseIterator)

    def test_get_iterator_shifter_sampler_kwargs(self) -> None:
        """Verify sampler_kwargs is passed through to ShifterSampler."""
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        prep = get_preprocessor("basic", tok, max_length=64)
        it = get_iterator(
            "multi_neg_pairwise",
            prep,
            ShifterSampler,
            num_neg_samples=2,
            sampler_kwargs={"max_epoch": 7},
        )
        assert isinstance(it, BioASQMultiNegativePairwiseIterator)
        assert it.sampler_kwargs == {"max_epoch": 7}

        # Call iterator to create sampler - ShifterSampler should receive max_epoch
        ds = create_mock_slice_dataset()
        it(dataset=ds, epoch=0, collection=None)
        assert isinstance(it.sampler, ShifterSampler)
        assert it.sampler.max_epoch == 7

    def test_get_trainer_cls(self) -> None:
        assert get_trainer_cls("pointwise") is PointwiseRerankerTrainer
        assert get_trainer_cls("pairwise") is PairwiseRerankerTrainer


class TestSamplerKwargsFlow:
    """Integration: sampler_kwargs from iterator to ShifterSampler."""

    def test_shifter_sampler_receives_max_epoch(self) -> None:
        ds = create_mock_slice_dataset()
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        prep = get_preprocessor("basic", tok, max_length=64)

        it = get_iterator(
            "pairwise",
            prep,
            ShifterSampler,
            num_neg_samples=1,
            sampler_kwargs={"max_epoch": 12},
        )
        it(dataset=ds, epoch=0, collection=None)

        assert it.sampler.max_epoch == 12

    def test_basic_sampler_ignores_sampler_kwargs(self) -> None:
        """BasicSampler does not use max_epoch; should not crash."""
        ds = create_mock_slice_dataset()
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        prep = get_preprocessor("basic", tok, max_length=64)

        it = get_iterator(
            "pairwise",
            prep,
            BasicSampler,
            num_neg_samples=1,
            sampler_kwargs={"max_epoch": 99},  # BasicSampler ignores this
        )
        it(dataset=ds, epoch=0, collection=None)

        # Should work - BasicSampler doesn't have max_epoch
        assert isinstance(it.sampler, BasicSampler)


class TestCreateBioASQDatasets:
    """Integration: create_bioASQ_datasets with real format."""

    def test_create_datasets_smoke(self) -> None:
        pos_path, all_path = create_mock_jsonl_files()
        try:
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            prep = get_preprocessor("basic", tok, max_length=64)
            it = get_iterator(
                "pairwise", prep, BasicSampler, num_neg_samples=1, sampler_kwargs={}
            )

            train_ds, test_ds, _, _, _ = create_bioASQ_datasets(
                positive_data_path=str(pos_path),
                all_data_path=str(all_path),
                iterator=it,
                test_sample_preprocessing=prep,
                val_files=None,
                relevance_mapping={"documents": 1},
            )

            assert len(train_ds) > 0
        finally:
            pos_path.unlink(missing_ok=True)
            all_path.unlink(missing_ok=True)


class TestSmokeTrainStep:
    """Minimal 1-step training smoke test."""

    def test_pointwise_one_step(self) -> None:
        set_seed(42)
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=1
        )
        prep = get_preprocessor("basic", tok, max_length=64)
        collator = get_collator("pointwise", tok)

        ds = create_mock_slice_dataset()
        it = get_iterator(
            "pointwise", prep, BasicSampler, num_neg_samples=1, sampler_kwargs={}
        )
        train_ds = BioASQDataset(dataset=ds, iterator=it, collection=None)

        args = create_training_config(
            CONFIG_PATH,
            output_dir=str(Path(tempfile.mkdtemp())),
            per_device_train_batch_size=2,
            max_steps=1,
            eval_strategy="no",
            report_to="none",  # HF expects str "none", not Python None
        )
        trainer = PointwiseRerankerTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            data_collator=collator,
            processing_class=tok,
        )
        trainer.train()
        assert trainer.state.global_step >= 1

    def test_pairwise_one_step(self) -> None:
        set_seed(42)
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=1
        )
        prep = get_preprocessor("basic", tok, max_length=64)
        collator = get_collator("pairwise", tok)

        ds = create_mock_slice_dataset()
        it = get_iterator(
            "pairwise", prep, BasicSampler, num_neg_samples=1, sampler_kwargs={}
        )
        train_ds = BioASQDataset(dataset=ds, iterator=it, collection=None)

        args = create_training_config(
            CONFIG_PATH,
            output_dir=str(Path(tempfile.mkdtemp())),
            per_device_train_batch_size=2,
            max_steps=1,
            eval_strategy="no",
            report_to="none",
            remove_unused_columns=False,
        )
        trainer = PairwiseRerankerTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            data_collator=collator,
            processing_class=tok,
        )
        trainer.train()
        assert trainer.state.global_step >= 1

    def test_multi_neg_pairwise_one_step(self) -> None:
        set_seed(42)
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=1
        )
        prep = get_preprocessor("basic", tok, max_length=64)
        collator = get_collator("multi_neg_pairwise", tok)

        ds = create_mock_slice_dataset()
        it = get_iterator(
            "multi_neg_pairwise",
            prep,
            BasicSampler,
            num_neg_samples=2,
            sampler_kwargs={},
        )
        train_ds = BioASQDataset(dataset=ds, iterator=it, collection=None)

        args = create_training_config(
            CONFIG_PATH,
            output_dir=str(Path(tempfile.mkdtemp())),
            per_device_train_batch_size=2,
            max_steps=1,
            eval_strategy="no",
            report_to="none",
            remove_unused_columns=False,
        )
        trainer_cls = get_trainer_cls("multi_neg_pairwise")
        trainer = trainer_cls(
            model=model,
            args=args,
            train_dataset=train_ds,
            data_collator=collator,
            processing_class=tok,
        )
        trainer.train()
        assert trainer.state.global_step >= 1
