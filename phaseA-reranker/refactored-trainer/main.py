"""
Refactored reranker trainer CLI.

Usage:
    python -m main train --model-name bert-base-uncased \\
        --positive-data-path data/positives.jsonl \\
        --all-data-path data/all.jsonl

    python -m main inference --model-name outputs/checkpoint \\
        --val-files data/13B1_golden.json,data/13B2_golden.json \\
        --positive-data-path data/positives.jsonl \\
        --all-data-path data/all.jsonl
"""

from __future__ import annotations

import json
import os


from pathlib import Path
from typing import Annotated, Optional

import typer
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from collator import RankingCollator
from data import create_bioASQ_datasets
from evaluation import (
    DEFAULT_METRICS,
    evaluate_run,
    run_inference,
    save_predictions,
)
from factory import (
    get_collator,
    get_iterator,
    get_preprocessor,
    get_sampler,
    get_trainer_cls,
)
from utils import build_output_dir_name, create_training_config, set_seed

app = typer.Typer()
BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG = BASE_DIR / "config" / "train_config.yaml"


def _sanitize_position_ids_buffers(model: torch.nn.Module) -> None:
    """Ensure any `position_ids` buffers are monotonic 0..N-1.

    Some remote-code checkpoints may load a corrupted `position_ids` buffer,
    which later causes out-of-bounds indexing in positional embeddings/RoPE.
    """
    for module in model.modules():
        position_ids = getattr(module, "position_ids", None)
        if not isinstance(position_ids, torch.Tensor):
            continue
        if position_ids.dtype not in (torch.int32, torch.int64):
            continue
        if position_ids.ndim == 1:
            expected = torch.arange(
                position_ids.shape[0],
                device=position_ids.device,
                dtype=position_ids.dtype,
            )
        elif position_ids.ndim == 2 and position_ids.shape[0] == 1:
            expected = torch.arange(
                position_ids.shape[1],
                device=position_ids.device,
                dtype=position_ids.dtype,
            ).unsqueeze(0)
        else:
            continue

        if not torch.equal(position_ids, expected):
            position_ids.copy_(expected)


def _build_sampler_kwargs(
    sampler: str,
    sampler_max_epoch: int = 10,
    **extra: object,
) -> dict:
    """
    Build kwargs passed to the sampler when instantiated.
    Maps CLI args to sampler-specific parameters.
    """
    kwargs: dict = dict(extra)
    if sampler.lower() == "shifter":
        kwargs["max_epoch"] = sampler_max_epoch
    return kwargs


def _resolve_model_dtype(*, bf16: bool, fp16: bool) -> torch.dtype:
    if bf16 and fp16:
        raise typer.BadParameter("bf16 and fp16 cannot both be enabled")
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return torch.float32


def _resolve_inference_dtype(inference_dtype: str) -> torch.dtype:
    normalized = inference_dtype.strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if normalized not in mapping:
        raise typer.BadParameter(
            "inference_dtype must be one of: float32, bfloat16, float16"
        )
    return mapping[normalized]


@app.command()
def train(
    model_name: Annotated[str, typer.Option(help="HuggingFace model name or path")],
    positive_data_path: Annotated[
        str, typer.Option(help="JSONL with id, body, documents")
    ],
    all_data_path: Annotated[str, typer.Option(help="JSONL with id, neg_docs")],
    val_files: Annotated[
        str | None,
        typer.Option(help="Comma-separated paths to golden JSON val files"),
    ] = None,
    output_dir: Annotated[
        str, typer.Option(help="Base directory for model outputs")
    ] = "outputs",
    use_expanded_pos: Annotated[bool, typer.Option()] = False,
    callback: Annotated[bool, typer.Option(help="Enable ResampleByReranker callback")] = False,
    warmup_ratio: Annotated[bool, typer.Option(help="Enable 10%% warmup")] = True,
    data: Annotated[str, typer.Option(help="Data identifier (e.g. quality)")] = "data",
    seed: Annotated[int, typer.Option()] = 42,
    batch_size: Annotated[int, typer.Option()] = 16,
    gradient_accumulation_steps: Annotated[int, typer.Option()] = 2,
    num_epochs: Annotated[int, typer.Option()] = 5,
    learning_rate: Annotated[float, typer.Option()] = 2e-5,
    max_length: Annotated[int, typer.Option()] = 512,
    num_neg_samples: Annotated[int, typer.Option()] = 4,
    sampler: Annotated[
        str, typer.Option(help="basic, basicv2, exponential, shifter")
    ] = "basic",
    mode: Annotated[
        str,
        typer.Option(help="pointwise, pairwise, multi_neg_pairwise"),
    ] = "pairwise",
    config_path: Annotated[
        Optional[Path],
        typer.Option(help="Path to YAML training config"),
    ] = None,
    sampler_max_epoch: Annotated[
        int,
        typer.Option(help="max_epoch for ShifterSampler (if sampler=shifter)"),
    ] = 10,
    margin: Annotated[
        float, typer.Option(help="Margin for pairwise/multi-neg loss")
    ] = 1.0,
    bf16: Annotated[bool, typer.Option()] = True,
    fp16: Annotated[bool, typer.Option()] = False,
) -> None:
    """Train a reranker model."""
    set_seed(seed)

    model_dtype = _resolve_model_dtype(bf16=bf16, fp16=fp16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.model_max_length = max_length

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    _sanitize_position_ids_buffers(model)
    if getattr(model.config, "num_labels", 1) != 1:
        model.config.num_labels = 1
        model.config.id2label = {0: "SCORE"}
        model.config.label2id = {"SCORE": 0}

    preprocessor = get_preprocessor("basic", tokenizer, max_length=max_length)
    sampler_cls = get_sampler(sampler)

    # Build sampler_kwargs from CLI args (extensible for future samplers)
    sampler_kwargs: dict = _build_sampler_kwargs(
        sampler=sampler,
        sampler_max_epoch=sampler_max_epoch,
    )

    iterator = get_iterator(
        mode=mode,
        sample_preprocessing=preprocessor,
        sampler_cls=sampler_cls,
        num_neg_samples=num_neg_samples,
        sampler_kwargs=sampler_kwargs,
    )

    val_files_list: list[str] = []
    if val_files:
        val_files_list = [p.strip() for p in val_files.split(",") if p.strip()]

    train_dataset, test_dataset, eval_pointwise, eval_pairwise, eval_multi_neg = (
        create_bioASQ_datasets(
            positive_data_path=positive_data_path,
            all_data_path=all_data_path,
            iterator=iterator,
            test_sample_preprocessing=preprocessor,
            val_files=val_files_list if val_files_list else None,
            relevance_mapping={"documents": 1},
        )
    )

    collator = get_collator(mode, tokenizer)
    eval_dataset = (
        eval_pointwise
        if mode == "pointwise"
        else eval_multi_neg if mode == "multi_neg_pairwise" else eval_pairwise
    )

    loss_mode = (
        "Pointwise"
        if mode == "pointwise"
        else "MultiNegPairwise" if mode == "multi_neg_pairwise" else "Pairwise"
    )
    val_str = "val" if val_files_list else "full"
    out_dir_name = build_output_dir_name(
        model_name=model_name,
        seed=seed,
        epoch=num_epochs,
        sampler_name=sampler_cls.__name__,
        sample_preprocessing_name=preprocessor.__class__.__name__,
        val=val_str,
        data=data,
        callback=callback,
        num_neg_samples=num_neg_samples,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_expanded_pos=use_expanded_pos,
        warmup_ratio=warmup_ratio,
        loss_mode=loss_mode,
    )
    final_output_dir = Path(output_dir) / out_dir_name
    typer.echo(f"Output dir: {final_output_dir}")

    training_args = create_training_config(
        config_path or DEFAULT_CONFIG,
        output_dir=str(final_output_dir),
        seed=seed,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=bf16,
        fp16=fp16,
        warmup_ratio=0.1 if warmup_ratio else 0.0,
        eval_strategy="no" if eval_dataset is None else "epoch",
        remove_unused_columns=(mode == "pointwise"),
    )
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        training_args._n_gpu = 1

    trainer_cls = get_trainer_cls(mode)
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        margin=margin,
    )

    trainer.train()


def _load_qids_per_val_file(val_files: list[str]) -> dict[str, list[str]]:
    """Return {basename: [qid, ...]} for each val file."""
    per_file: dict[str, list[str]] = {}
    for path in val_files:
        with open(path) as f:
            data = json.load(f)
        qids = [str(q["id"]) for q in data["questions"]]
        per_file[os.path.basename(path)] = qids
    return per_file


@app.command()
def inference(
    model_name: Annotated[str, typer.Option(help="Model or checkpoint path")],
    positive_data_path: Annotated[
        str, typer.Option(help="JSONL with id, body, documents")
    ],
    all_data_path: Annotated[str, typer.Option(help="JSONL with id, neg_docs")],
    val_files: Annotated[
        str,
        typer.Option(help="Comma-separated paths to golden JSON val files"),
    ],
    batch_size: Annotated[int, typer.Option()] = 64,
    max_length: Annotated[int, typer.Option()] = 512,
    inference_dtype: Annotated[
        str,
        typer.Option(help="float32, bfloat16, or float16"),
    ] = "bfloat16",
    num_workers: Annotated[int, typer.Option(help="DataLoader worker processes")] = 2,
    pin_memory: Annotated[
        bool,
        typer.Option(help="Enable pinned CPU memory for faster GPU transfers"),
    ] = True,
    non_blocking: Annotated[
        bool,
        typer.Option(help="Use non-blocking host->device tensor copies"),
    ] = True,
    inspect_samples: Annotated[
        int,
        typer.Option(help="Print first N model input/output samples"),
    ] = 0,
    inspect_max_chars: Annotated[
        int,
        typer.Option(help="Max decoded chars shown per inspected sample"),
    ] = 240,
    show_progress: Annotated[bool, typer.Option(help="Show inference progress bar")] = True,
    results_file: Annotated[
        Optional[str],
        typer.Option(help="Append results to this JSONL file"),
    ] = None,
) -> None:
    """Run inference and evaluate a reranker model on validation data."""
    val_files_list = [p.strip() for p in val_files.split(",") if p.strip()]
    if not val_files_list:
        raise typer.BadParameter("val_files cannot be empty")

    if num_workers < 0:
        raise typer.BadParameter("num_workers must be >= 0")
    if inspect_samples < 0:
        raise typer.BadParameter("inspect_samples must be >= 0")
    if inspect_max_chars < 0:
        raise typer.BadParameter("inspect_max_chars must be >= 0")

    infer_dtype = _resolve_inference_dtype(inference_dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.model_max_length = max_length
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=infer_dtype,
    )
    _sanitize_position_ids_buffers(model)
    if getattr(model.config, "num_labels", 1) != 1:
        model.config.num_labels = 1
        model.config.id2label = {0: "SCORE"}
        model.config.label2id = {"SCORE": 0}

    preprocessor = get_preprocessor("basic", tokenizer, max_length=max_length)
    sampler_cls = get_sampler("basic")
    iterator = get_iterator(
        mode="pointwise",
        sample_preprocessing=preprocessor,
        sampler_cls=sampler_cls,
        num_neg_samples=1,
    )

    _, test_dataset, _, _, _ = create_bioASQ_datasets(
        positive_data_path=positive_data_path,
        all_data_path=all_data_path,
        iterator=iterator,
        test_sample_preprocessing=preprocessor,
        val_files=val_files_list,
        relevance_mapping={"documents": 1},
    )
    collator = RankingCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    run_dict = run_inference(
        model,
        dataloader,
        tokenizer=tokenizer,
        inspect_samples=inspect_samples,
        inspect_max_chars=inspect_max_chars,
        non_blocking=non_blocking,
        amp_dtype=infer_dtype,
        show_progress=show_progress,
    )
    qrels = test_dataset.get_qrels()
    per_file = _load_qids_per_val_file(val_files_list)
    results = evaluate_run(
        run_dict,
        qrels,
        metrics=DEFAULT_METRICS,
        per_file_results=per_file,
    )
    for key, metrics_dict in results.items():
        typer.echo(f"{key}: {metrics_dict}")

    # Save predictions to model folder (model_name/predictions/predictions.json)
    pred_path = save_predictions(run_dict, model_name)
    typer.echo(f"Predictions saved to {pred_path}")

    if results_file:
        metadata = {
            "model": model_name.replace("/", "-"),
            "val_files": val_files_list,
        }
        with open(results_file, "a") as f:
            f.write(json.dumps(metadata | results) + "\n")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
