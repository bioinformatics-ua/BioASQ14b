"""
Reranker trainer CLI commands for Phase A.

Migrated from `phaseA-reranker/refactored-trainer/main.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import orjson
import torch
import typer
from torch.utils.data import DataLoader

from bioasq.common.config import create_training_config, set_seed
from bioasq.common.metrics import DEFAULT_RETRIEVAL_METRICS
from bioasq.phase_a.reranker.collator import RankingCollator
from bioasq.phase_a.reranker.data import (
    create_bioasq_datasets,
    create_inference_dataset_from_bioasq_json,
)
from bioasq.phase_a.reranker.evaluate import evaluate_run, run_inference, save_predictions
from bioasq.phase_a.reranker.factory import (
    get_collator,
    get_iterator,
    get_preprocessor,
    get_sampler,
    get_trainer_cls,
)
from bioasq.phase_a.reranker.model import build_output_dir_name, load_reranker_model

BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG = BASE_DIR / "config" / "train_config.yaml"


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
        raise typer.BadParameter("inference_dtype must be one of: float32, bfloat16, float16")
    return mapping[normalized]


def _load_qids_per_val_file(val_files: list[str]) -> dict[str, list[str]]:
    """Return {basename: [qid, ...]} for each val file."""
    per_file: dict[str, list[str]] = {}
    for path in val_files:
        with Path(path).open() as f:
            data = json.load(f)
        qids = [str(q["id"]) for q in data["questions"]]
        per_file[Path(path).name] = qids
    return per_file


def _run_dict_to_bioasq_format(
    run_dict: dict[str, dict[str, float]],
    questions_by_id: dict[str, dict],
    top_k: int = 10,
) -> dict:
    """Convert run dict to BioASQ submission format."""
    questions_out = []
    for q_id, docs_dict in run_dict.items():
        q_meta = questions_by_id.get(q_id, {})
        sorted_docs = sorted(
            docs_dict.items(),
            key=lambda x: -x[1],
        )[:top_k]
        doc_urls = [f"http://www.ncbi.nlm.nih.gov/pubmed/{doc_id}" for doc_id, _ in sorted_docs]
        questions_out.append(
            {
                "id": q_id,
                "type": q_meta.get("type", "factoid"),
                "body": q_meta.get("body", ""),
                "documents": doc_urls,
                "snippets": [],
            }
        )
    return {"questions": questions_out}


def train_command(
    model_name: Annotated[str, typer.Option(help="HuggingFace model name or path")],
    positive_data_path: Annotated[str, typer.Option(help="JSONL with id, body, documents")],
    all_data_path: Annotated[str, typer.Option(help="JSONL with id, neg_docs")],
    val_files: Annotated[
        str | None,
        typer.Option(help="Comma-separated paths to golden JSON val files"),
    ] = None,
    output_dir: Annotated[str, typer.Option(help="Base directory for model outputs")] = "outputs",
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
    sampler: Annotated[str, typer.Option(help="basic, basicv2, exponential, shifter")] = "basic",
    mode: Annotated[
        str,
        typer.Option(help="pointwise, pairwise, multi_neg_pairwise"),
    ] = "pairwise",
    config_path: Annotated[
        Path | None,
        typer.Option(help="Path to YAML training config"),
    ] = None,
    sampler_max_epoch: Annotated[
        int,
        typer.Option(help="max_epoch for ShifterSampler (if sampler=shifter)"),
    ] = 10,
    margin: Annotated[float, typer.Option(help="Margin for pairwise/multi-neg loss")] = 1.0,
    bf16: Annotated[bool, typer.Option()] = True,
    fp16: Annotated[bool, typer.Option()] = False,
) -> None:
    """Train a reranker model."""
    set_seed(seed)

    model_dtype = _resolve_model_dtype(bf16=bf16, fp16=fp16)

    model, tokenizer = load_reranker_model(
        model_name=model_name,
        revision=None,
        max_length=max_length,
        dtype=model_dtype,
        num_labels=1,
    )

    preprocessor = get_preprocessor("basic", tokenizer, max_length=max_length)
    sampler_cls = get_sampler(sampler)

    sampler_kwargs: dict = _build_sampler_kwargs(
        sampler=sampler,
        sampler_max_epoch=sampler_max_epoch,
    )

    iterator = get_iterator(
        mode=mode,
        sample_preprocessing=preprocessor,  # type: ignore[arg-type]
        sampler_cls=sampler_cls,
        num_neg_samples=num_neg_samples,
        sampler_kwargs=sampler_kwargs,
    )

    val_files_list: list[Path] = []
    if val_files:
        val_files_list = [Path(p.strip()) for p in val_files.split(",") if p.strip()]

    train_dataset, _test_dataset, eval_pointwise, eval_pairwise, eval_multi_neg = (
        create_bioasq_datasets(
            positive_data_path=Path(positive_data_path),
            all_data_path=Path(all_data_path),
            iterator=iterator,
            test_sample_preprocessing=preprocessor,  # type: ignore[arg-type]  # type: ignore[arg-type]
            val_files=val_files_list if val_files_list else None,
            relevance_mapping={"documents": 1},
        )
    )

    collator = get_collator(mode, tokenizer)
    eval_dataset = (
        eval_pointwise
        if mode == "pointwise"
        else eval_multi_neg
        if mode == "multi_neg_pairwise"
        else eval_pairwise
    )

    loss_mode = (
        "Pointwise"
        if mode == "pointwise"
        else "MultiNegPairwise"
        if mode == "multi_neg_pairwise"
        else "Pairwise"
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
        Path(config_path) if config_path else DEFAULT_CONFIG,
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
        training_args._n_gpu = 1  # noqa: SLF001

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


def evaluate_command(
    model_name: Annotated[str, typer.Option(help="Model or checkpoint path")],
    positive_data_path: Annotated[str, typer.Option(help="JSONL with id, body, documents")],
    all_data_path: Annotated[str, typer.Option(help="JSONL with id, neg_docs")],
    val_files: Annotated[
        str | None,
        typer.Option(help="Comma-separated paths to golden JSON val files"),
    ] = None,
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
        str | None,
        typer.Option(help="Append results to this JSONL file"),
    ] = None,
) -> None:
    """Run inference and evaluate a reranker model on validation data."""
    val_files_list = (
        [Path(p.strip()) for p in val_files.split(",") if p.strip()] if val_files else None
    )

    if num_workers < 0:
        raise typer.BadParameter("num_workers must be >= 0")
    if inspect_samples < 0:
        raise typer.BadParameter("inspect_samples must be >= 0")
    if inspect_max_chars < 0:
        raise typer.BadParameter("inspect_max_chars must be >= 0")

    infer_dtype = _resolve_inference_dtype(inference_dtype)

    model, tokenizer = load_reranker_model(
        model_name=model_name,
        revision=None,
        max_length=max_length,
        dtype=infer_dtype,
        num_labels=1,
    )

    preprocessor = get_preprocessor("basic", tokenizer, max_length=max_length)
    sampler_cls = get_sampler("basic")
    iterator = get_iterator(
        mode="pointwise",
        sample_preprocessing=preprocessor,  # type: ignore[arg-type]
        sampler_cls=sampler_cls,
        num_neg_samples=1,
    )

    _, test_dataset, _, _, _ = create_bioasq_datasets(
        positive_data_path=Path(positive_data_path),
        all_data_path=Path(all_data_path),
        iterator=iterator,
        test_sample_preprocessing=preprocessor,  # type: ignore[arg-type]  # type: ignore[arg-type]
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
        dataloader,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        inspect_samples=inspect_samples,
        inspect_max_chars=inspect_max_chars,
        non_blocking=non_blocking,
        amp_dtype=infer_dtype,
        show_progress=show_progress,
    )
    qrels = test_dataset.get_qrels()
    per_file = _load_qids_per_val_file(val_files_list) if val_files_list else None
    results = evaluate_run(
        run_dict,
        qrels,
        metrics=DEFAULT_RETRIEVAL_METRICS,
        per_file_results=per_file,
    )
    for key, metrics_dict in results.items():
        typer.echo(f"{key}: {metrics_dict}")

    pred_path = save_predictions(run_dict, model_name)
    typer.echo(f"Predictions saved to {pred_path}")

    if results_file:
        metadata = {
            "model": model_name.replace("/", "-"),
            "val_files": val_files_list,
        }
        with Path(results_file).open("a") as f:
            f.write(json.dumps(metadata | results) + "\\n")


def inference_command(
    model_name: Annotated[str, typer.Option(help="Model or checkpoint path")],
    questions_path: Annotated[Path, typer.Option(help="JSON with questions in BioASQ format")],
    output_path: Annotated[Path, typer.Option(help="Path to save predictions in BioASQ format")],
    revision: Annotated[
        str | None, typer.Option(help="Revision of the model (ignored for local paths)")
    ] = None,
    batch_size: Annotated[int, typer.Option()] = 64,
    max_length: Annotated[int, typer.Option()] = 512,
    max_docs: Annotated[int, typer.Option(help="Max candidates per question")] = 100,
    inference_dtype: Annotated[
        Literal["float32", "bfloat16", "float16"],
        typer.Option(help="Inference data type: float32, bfloat16, or float16"),
    ] = "bfloat16",
    top_k: Annotated[int, typer.Option(help="Top-k documents per question in output")] = 10,  # noqa: ARG001
) -> None:
    """Run inference on a reranker model and save predictions in BioASQ format."""
    if not questions_path.exists():
        raise typer.BadParameter(f"Questions file not found: {questions_path}")

    infer_dtype = _resolve_inference_dtype(inference_dtype)

    questions_by_id: dict[str, dict] = {}
    with questions_path.open("rb") as f:
        raw = f.read()
    try:
        questions_data = orjson.loads(raw)
        questions_by_id = {str(q["id"]): q for q in questions_data["questions"]}
    except (orjson.JSONDecodeError, KeyError):
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            q = orjson.loads(line)
            questions_by_id[str(q["id"])] = q

    model, tokenizer = load_reranker_model(
        model_name=model_name,
        revision=revision,
        max_length=max_length,
        dtype=infer_dtype,
        num_labels=1,
    )

    preprocessor = get_preprocessor("basic", tokenizer, max_length=max_length)
    inference_dataset = create_inference_dataset_from_bioasq_json(
        questions_path,
        sample_preprocessing=preprocessor,  # type: ignore[arg-type]
        max_docs=max_docs,
    )

    if len(inference_dataset) == 0:
        typer.echo(
            "You must provide a question or data_path representing JSON queries"
            ". Ensure each question has 'documents', 'neg_docs', or 'bm25' with {id, text} entries."
        )
        raise typer.Exit(1)

    collator = RankingCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
    )

    run_dict = run_inference(
        model,
        dataloader,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        show_progress=True,
        amp_dtype=infer_dtype,
        device="cuda",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("wb") as f:
        f.write(orjson.dumps(run_dict))

    typer.echo(f"Predictions saved to {output_path}")
