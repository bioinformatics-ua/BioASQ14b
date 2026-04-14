"""
Reranker experiment runner CLI commands for Phase A.

Migrated from `phaseA-reranker/refactored-trainer/run_experiments.py`
and `run_llama_experiments.py`.
"""

import json
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pathlib import Path

import torch
import typer
import wandb
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from typer import Typer

from bioasq.common import PROJECT_DATA_DIR
from bioasq.common.config import get_wandb_run_id, set_seed, setup_wandb
from bioasq.common.metrics import DEFAULT_RETRIEVAL_METRICS
from bioasq.phase_a.reranker.data import create_bioasq_datasets
from bioasq.phase_a.reranker.evaluate import evaluate_run, run_inference, save_predictions
from bioasq.phase_a.reranker.factory import (
    get_collator,
    get_iterator,
    get_preprocessor,
    get_sampler,
    get_trainer_cls,
)
from bioasq.phase_a.reranker.trainer import EarlyStoppingOnGradNorm

app = Typer(
    name="experiments",
    help="Commands for running reranker experiments on BioASQ Phase A.",
    no_args_is_help=True,
)


# Shared Helpers
def _short_model_name(model_name: str) -> str:
    """Return the last path component for HF model names, else the name as-is."""
    if "://" in model_name or "/" not in model_name:
        return model_name
    return Path(model_name).name


def _output_dir(model_name: str, run_name: str | None = None, is_llama: bool = False) -> str:
    suffix = "_llama" if is_llama else ""
    base = f"./outputs/{model_name.replace('/', '_')}{suffix}"
    return f"{base}/{run_name}" if run_name else base


def _load_cached_results(
    model_name: str, run_name: str | None = None, is_llama: bool = False
) -> dict | None:
    result_path = Path(_output_dir(model_name, run_name, is_llama)) / "ranx_results.json"
    if result_path.exists():
        with Path(result_path).open() as f:
            return json.load(f)
    return None


def _find_latest_checkpoint(out_dir: Path) -> Path | None:
    checkpoints = [d for d in out_dir.glob("checkpoint-*") if d.is_dir()]
    if not checkpoints:
        return None

    def _step(p: Path) -> int:
        try:
            return int(p.name.replace("checkpoint-", ""))
        except ValueError:
            return 0

    latest = max(checkpoints, key=_step)
    if (latest / "model.safetensors").exists() or (latest / "pytorch_model.bin").exists():
        return latest
    return None


def _find_trainer_state(
    model_name: str, run_name: str | None = None, is_llama: bool = False
) -> Path | None:
    out_dir = Path(_output_dir(model_name, run_name, is_llama))
    if not out_dir.exists():
        return None
    root_state = out_dir / "trainer_state.json"
    if root_state.exists():
        return root_state
    checkpoints = list(out_dir.glob("checkpoint-*/trainer_state.json"))
    if not checkpoints:
        return None

    def _step(p: Path) -> int:
        try:
            return int(p.parent.name.replace("checkpoint-", ""))
        except ValueError:
            return 0

    return max(checkpoints, key=_step)


def _replay_log_history_to_wandb(state_path: Path) -> None:
    with Path(state_path).open() as f:
        state = json.load(f)
    log_history = state.get("log_history", [])
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        metrics = {k: v for k, v in entry.items() if k != "step" and isinstance(v, (int, float))}
        if metrics:
            wandb.log(metrics, step=step)


def _log_results_to_wandb(
    run_name: str, results: dict, model_name: str, is_llama: bool = False
) -> None:
    setup_wandb(run_name)
    run_id = get_wandb_run_id(run_name)
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "bioasq-14b-phaseA-reranker"),
        name=run_name,
        id=run_id,
        resume="allow",
        reinit=False,
    )
    state_path = _find_trainer_state(model_name, run_name, is_llama)
    if state_path is not None:
        _replay_log_history_to_wandb(state_path)
    if "total" in results:
        wandb.summary.update(results["total"])
    wandb.finish()


def _load_qids_per_val_file(val_files: list[str]) -> dict[str, list[str]]:
    per_file: dict[str, list[str]] = {}
    for path in val_files:
        with Path(path).open() as f:
            data = json.load(f)
        qids = [str(q["id"]) for q in data["questions"]]
        per_file[Path(path).name] = qids
    return per_file


def _sanitize_position_ids_buffers(model: torch.nn.Module) -> None:
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


def _setup_nemotron_tokenizer(tokenizer: PreTrainedTokenizerBase) -> None:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _setup_nemotron_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    if getattr(model.config, "num_labels", 1) != 1:
        model.config.num_labels = 1
        model.config.id2label = {0: "SCORE"}
        model.config.label2id = {"SCORE": 0}


def _run_experiment(
    model_name: str,
    config: dict,
    run_name: str | None = None,
    model_path: str | None = None,
) -> dict:
    load_path = model_path if model_path is not None else model_name
    print(f"--- Starting experiment for: {model_name} (loading from {load_path}) ---")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        load_path,
        num_labels=1,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16,
    )
    _sanitize_position_ids_buffers(model)

    preprocessor = get_preprocessor("basic", tokenizer=tokenizer, max_length=512)
    use_expanded_pos = bool(config.get("expanded_pos_path"))
    sampler_cls = get_sampler("exponential" if use_expanded_pos else "shifter")

    iterator = get_iterator(
        mode=config["mode"],
        sample_preprocessing=preprocessor,
        sampler_cls=sampler_cls,
        num_neg_samples=config["num_neg_samples"],
        sampler_kwargs={"max_epoch": config["epochs"]},
    )

    collator = get_collator(mode=config["mode"], tokenizer=tokenizer)
    trainer_cls = get_trainer_cls(mode=config["mode"])

    train_pos_path = config.get("expanded_pos_path") or config["train_pos_path"]
    train_ds, test_ds, eval_pointwise, eval_pairwise, eval_multi_neg = create_bioasq_datasets(
        positive_data_path=train_pos_path,
        all_data_path=config["train_neg_path"],
        iterator=iterator,
        test_sample_preprocessing=preprocessor,  # type: ignore[arg-type]
        val_files=config["val_files"],  #  if config.get("full_data", False) else None,
    )

    output_dir = _output_dir(model_name, run_name)
    if run_name is None:
        run_name = (
            f"{model_name}-E{config['epochs']}-S{config['num_neg_samples']}-M{config['mode']}"
        )
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config.get("per_device_train_batch_size", config["batch_size"]),
        per_device_eval_batch_size=config.get(
            "per_device_eval_batch_size", config["batch_size"] * 2
        ),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        learning_rate=config["learning_rate"],
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=500,
        bf16=True,
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        remove_unused_columns=False,
        report_to=config["report_to"],
    )

    eval_ds = (
        eval_pointwise
        if config["mode"] == "pointwise"
        else eval_multi_neg
        if config["mode"] == "multi_neg_pairwise"
        else eval_pairwise
    )

    callbacks = []
    if config.get("early_stop_on_grad", True):
        callbacks.append(
            EarlyStoppingOnGradNorm(
                grad_norm_threshold=config.get("grad_norm_threshold", 1e-6),
                patience=config.get("grad_norm_patience", 5),
            )
        )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        margin=1.0,  # type: ignore[call-arg]
        callbacks=callbacks,
    )

    trainer.train()

    print(f"Running inference for {model_name}...")
    inference_collator = get_collator(mode="pointwise", tokenizer=tokenizer)

    if config.get("full_data", True):
        _, test_ds, _, _, _ = create_bioasq_datasets(
            positive_data_path=train_pos_path,
            all_data_path=config["train_neg_path"],
            iterator=iterator,
            test_sample_preprocessing=preprocessor,  # type: ignore[arg-type]
            val_files=config["val_files"],
        )

    test_dataloader = DataLoader(
        test_ds,
        batch_size=config["batch_size"] * 4,
        collate_fn=inference_collator,
        num_workers=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dict = run_inference(
        model,
        test_dataloader,  # type: ignore[arg-type]
        device=device,
        amp_dtype=torch.bfloat16,
    )

    qrels_dict = test_ds.get_qrels()
    val_files_list = [str(p) for p in (config.get("val_files") or [])]
    per_file = _load_qids_per_val_file(val_files_list) if val_files_list else None
    results = evaluate_run(
        run_dict,
        qrels_dict,
        metrics=DEFAULT_RETRIEVAL_METRICS,
        per_file_results=per_file,
    )

    save_predictions(run_dict, output_dir)
    metadata = {
        "model": model_name.replace("/", "-"),
        "val_files": val_files_list,
    }
    result_path = f"{output_dir}/ranx_results.json"
    with Path(result_path).open("w") as f:
        json.dump(metadata | results, f, indent=4)

    print(f"Metrics for {model_name}: {results['total']}\\n")
    return results


def _run_inference_only(model_name: str, config: dict, run_name: str | None = None) -> dict:
    direct_path = Path(model_name)
    if direct_path.is_absolute() and direct_path.exists():
        output_dir = str(direct_path)
        load_path = str(direct_path)
    else:
        output_dir = _output_dir(model_name, run_name)
        model_path = Path(output_dir)
        load_path = model_name
        if model_path.exists():
            latest_ckpt = _find_latest_checkpoint(model_path)
            if latest_ckpt is not None:
                load_path = str(latest_ckpt)
            elif (model_path / "model.safetensors").exists() or (
                model_path / "pytorch_model.bin"
            ).exists():
                load_path = str(model_path)

    print(f"--- Inference-only for: {model_name} (loading from {load_path}) ---")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        load_path,
        num_labels=1,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16,
    )
    _sanitize_position_ids_buffers(model)

    preprocessor = get_preprocessor("basic", tokenizer=tokenizer, max_length=512)
    iterator = get_iterator(
        mode="pointwise",
        sample_preprocessing=preprocessor,
        sampler_cls=get_sampler("basic"),
        num_neg_samples=1,
    )

    _, test_ds, _, _, _ = create_bioasq_datasets(
        positive_data_path=config["train_pos_path"],
        all_data_path=config["train_neg_path"],
        iterator=iterator,
        test_sample_preprocessing=preprocessor,  # type: ignore[arg-type]
        val_files=config["val_files"],
    )

    inference_collator = get_collator(mode="pointwise", tokenizer=tokenizer)
    test_dataloader = DataLoader(
        test_ds,
        batch_size=config["batch_size"] * 4,
        collate_fn=inference_collator,  # type: ignore
        num_workers=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dict = run_inference(  # type: ignore[arg-type]
        model,
        test_dataloader,  # type: ignore[arg-type]
        device=device,
        amp_dtype=torch.bfloat16,
    )

    save_predictions(run_dict, output_dir)

    qrels_dict = test_ds.get_qrels()
    val_files_list = [str(p) for p in (config.get("val_files") or [])]
    per_file = _load_qids_per_val_file(val_files_list) if val_files_list else None
    results = evaluate_run(
        run_dict,
        qrels_dict,
        metrics=DEFAULT_RETRIEVAL_METRICS,
        per_file_results=per_file,
    )

    metadata = {
        "model": model_name.replace("/", "-"),
        "val_files": val_files_list,
    }
    result_path = f"{output_dir}/ranx_results.json"
    with Path(result_path).open("w") as f:
        json.dump(metadata | results, f, indent=4)

    print(f"Metrics for {model_name}: {results['total']}\\n")
    return results


# Llama Experiment
def _run_llama_experiment(model_name: str, config: dict, run_name: str | None = None) -> dict:
    print(f"--- Starting Llama/Nemotron experiment for: {model_name} ---")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _setup_nemotron_tokenizer(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16,
    )
    _sanitize_position_ids_buffers(model)
    _setup_nemotron_model(model, tokenizer)

    preprocessor = get_preprocessor("nemotron", tokenizer=tokenizer, max_length=512)
    use_expanded_pos = bool(config.get("expanded_pos_path"))
    sampler_cls = get_sampler("exponential" if use_expanded_pos else "shifter")

    iterator = get_iterator(
        mode=config["mode"],
        sample_preprocessing=preprocessor,
        sampler_cls=sampler_cls,
        num_neg_samples=config["num_neg_samples"],
        sampler_kwargs={"max_epoch": config["epochs"]},
    )

    collator = get_collator(mode=config["mode"], tokenizer=tokenizer)
    trainer_cls = get_trainer_cls(mode=config["mode"])

    train_pos_path = config.get("expanded_pos_path") or config["train_pos_path"]
    train_ds, test_ds, eval_pointwise, eval_pairwise, eval_multi_neg = create_bioasq_datasets(
        positive_data_path=train_pos_path,
        all_data_path=config["train_neg_path"],
        iterator=iterator,
        test_sample_preprocessing=preprocessor,  # type: ignore[arg-type]
        val_files=config["val_files"] if config.get("full_data", False) else None,
    )

    output_dir = _output_dir(model_name, run_name, is_llama=True)
    if run_name is None:
        loss = config.get("loss_type", "margin")
        run_name = f"{model_name}-E{config['epochs']}-S{config['num_neg_samples']}-M{config['mode']}-L{loss}"  # noqa: E501
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"] * 2,
        learning_rate=config["learning_rate"],
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        bf16=True,
        remove_unused_columns=False,
        report_to=config["report_to"],
    )

    eval_ds = (
        eval_pointwise
        if config["mode"] == "pointwise"
        else eval_multi_neg
        if config["mode"] == "multi_neg_pairwise"
        else eval_pairwise
    )

    loss_type = config.get("loss_type", "margin")
    infonce_temperature = config.get("infonce_temperature", 0.05)
    callbacks = []
    if config.get("early_stop_on_grad", True):
        callbacks.append(
            EarlyStoppingOnGradNorm(
                grad_norm_threshold=config.get("grad_norm_threshold", 1e-6),
                patience=config.get("grad_norm_patience", 5),
            )
        )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
        "margin": 1.0,
        "callbacks": callbacks,
    }
    if config["mode"] == "multi_neg_pairwise":
        trainer_kwargs["loss_type"] = loss_type
        trainer_kwargs["infonce_temperature"] = infonce_temperature

    trainer = trainer_cls(**trainer_kwargs)

    trainer.train()

    print(f"Running inference for {model_name}...")
    inference_collator = get_collator(mode="pointwise", tokenizer=tokenizer)

    if config.get("full_data", True):
        _, test_ds, _, _, _ = create_bioasq_datasets(
            positive_data_path=train_pos_path,
            all_data_path=config["train_neg_path"],
            iterator=iterator,
            test_sample_preprocessing=preprocessor,  # type: ignore[arg-type]
            val_files=config["val_files"],
        )

    test_dataloader = DataLoader(
        test_ds,
        batch_size=config["batch_size"] * 4,
        collate_fn=inference_collator,  # type: ignore
        num_workers=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dict = run_inference(  # type: ignore[arg-type]
        model,
        test_dataloader,  # type: ignore[arg-type]  # type: ignore
        device=device,
        amp_dtype=torch.bfloat16,
    )

    qrels_dict = test_ds.get_qrels()
    val_files_list = [str(p) for p in (config.get("val_files") or [])]
    per_file = _load_qids_per_val_file(val_files_list) if val_files_list else None
    results = evaluate_run(
        run_dict,
        qrels_dict,
        metrics=DEFAULT_RETRIEVAL_METRICS,
        per_file_results=per_file,
    )

    save_predictions(run_dict, output_dir)
    metadata = {
        "model": model_name.replace("/", "-"),
        "val_files": val_files_list,
    }
    result_path = f"{output_dir}/ranx_results.json"
    with Path(result_path).open("w") as f:
        json.dump(metadata | results, f, indent=4)

    print(f"Metrics for {model_name}: {results['total']}\\n")
    return results


def _run_llama_inference_only(model_name: str, config: dict, run_name: str | None = None) -> dict:
    direct_path = Path(model_name)
    if direct_path.is_absolute() and direct_path.exists():
        # model_name is a direct local path (e.g. a checkpoint dir)
        output_dir = str(direct_path)
        load_path = str(direct_path)
    else:
        if run_name is None:
            run_name = f"{model_name}-E{config['epochs']}-S{config['num_neg_samples']}-M{config['mode']}-L{config.get('loss_type', 'margin')}"  # noqa: E501
        output_dir = _output_dir(model_name, run_name, is_llama=True)
        model_path = Path(output_dir)
        load_path = model_name
        if model_path.exists():
            latest_ckpt = _find_latest_checkpoint(model_path)
            if latest_ckpt is not None:
                load_path = str(latest_ckpt)
            elif (model_path / "model.safetensors").exists() or (
                model_path / "pytorch_model.bin"
            ).exists():
                load_path = str(model_path)

    print(f"--- Inference-only for: {model_name} (loading from {load_path}) ---")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _setup_nemotron_tokenizer(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        load_path,
        num_labels=1,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16,
    )
    _sanitize_position_ids_buffers(model)
    _setup_nemotron_model(model, tokenizer)

    preprocessor = get_preprocessor("nemotron", tokenizer=tokenizer, max_length=512)
    iterator = get_iterator(
        mode="pointwise",
        sample_preprocessing=preprocessor,
        sampler_cls=get_sampler("basic"),
        num_neg_samples=1,
    )

    _, test_ds, _, _, _ = create_bioasq_datasets(
        positive_data_path=config["train_pos_path"],
        all_data_path=config["train_neg_path"],
        iterator=iterator,
        test_sample_preprocessing=preprocessor,  # type: ignore[arg-type]
        val_files=config["val_files"],
    )

    inference_collator = get_collator(mode="pointwise", tokenizer=tokenizer)
    test_dataloader = DataLoader(
        test_ds,
        batch_size=config["batch_size"] * 4,
        collate_fn=inference_collator,
        num_workers=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dict = run_inference(
        model,
        test_dataloader,  # type: ignore[arg-type]
        device=device,
        amp_dtype=torch.bfloat16,
    )

    save_predictions(run_dict, output_dir)

    qrels_dict = test_ds.get_qrels()
    val_files_list = [str(p) for p in (config.get("val_files") or [])]
    per_file = _load_qids_per_val_file(val_files_list) if val_files_list else None
    results = evaluate_run(
        run_dict,
        qrels_dict,
        metrics=DEFAULT_RETRIEVAL_METRICS,
        per_file_results=per_file,
    )

    metadata = {
        "model": model_name.replace("/", "-"),
        "val_files": val_files_list,
    }
    result_path = f"{output_dir}/ranx_results.json"
    with Path(result_path).open("w") as f:
        json.dump(metadata | results, f, indent=4)

    print(f"Metrics for {model_name}: {results['total']}\\n")
    return results


@app.command(name="run-experiments")
def run_experiments_command(
    inference_only: bool = typer.Option(
        False, "--inference-only", help="Skip training; run inference only (load from outputs)"
    ),
    config_file: str | None = typer.Option(
        None, "--config", help="Optional path to experiments JSON config"
    ),
) -> None:
    """Run baseline experiments from a config file."""
    set_seed(42)
    if config_file and Path(config_file).exists():
        with Path(config_file).open() as f:
            cfg = json.load(f)
        models_to_test = cfg.get("models_to_test", [])
        config = cfg.get("config", {})
    else:
        # Default fallback corresponding to run_experiments.py
        models_to_test = [
            # "google/medgemma-4b-pt"
            # "ncbi/MedCPT-Cross-Encoder",
            # "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            # "michiyasunaga/BioLinkBERT-base",
            # "michiyasunaga/BioLinkBERT-large",
            # "pritamdeka/S-PubMedBert-MS-MARCO",
            # "monologg/biobert_v1.1_pubmed",
            "nboost/pt-biobert-base-msmarco",
            # "cross-encoder/ms-marco-MiniLM-L-6-v2",
            # "BAAI/bge-reranker-base",
            # "BAAI/bge-reranker-v2-m3",
        ]

        # models_to_test = [
        #     "/home/ucloud/BioASQ13B/src/outputs/google_medgemma-4b-pt/medgemma-4b-pt-E2-S1-Mpairwise-Lmargin-FullData-/checkpoint-100",
        # ]

        config = {
            "mode": "pairwise",
            "num_neg_samples": 1,
            "loss_type": "margin",
            "report_to": "wandb",
            "epochs": 1,
            "batch_size": 32,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True,
            "learning_rate": 2e-5,
            "train_pos_path": PROJECT_DATA_DIR
            / "quality/training14b_inflated_clean_wContents.jsonl",
            "train_neg_path": PROJECT_DATA_DIR / "negatives_fixed.jsonl",
            "full_data": False,  # ALWAYS TRUEEE
            "val_files": [
                PROJECT_DATA_DIR / "val_data/13B1_golden.json",
                PROJECT_DATA_DIR / "val_data/13B2_golden.json",
                PROJECT_DATA_DIR / "val_data/13B3_golden.json",
                PROJECT_DATA_DIR / "val_data/13B4_golden.json",
            ],
        }

    failed_models = []
    all_results = {}
    is_full = config.get("full_data", False)
    run_name_tpl = (
        "{model}-E{epochs}-S{num_neg}-M{mode}-L{loss}-FullData-{expanded}"
        if is_full
        else "{model}-E{epochs}-S{num_neg}-M{mode}-L{loss}{expanded}"
    )

    for entry in models_to_test:
        if isinstance(entry, dict):
            model_name = entry["model"]
            model_path = entry.get("checkpoint")
        else:
            model_name = entry
            model_path = None

        expanded_suffix = "-Expanded" if config.get("expanded_pos_path") else ""
        sampler_suffix = "Exponential" if config.get("use_expanded_pos") else "BasicV2"

        run_name = run_name_tpl.format(
            model=_short_model_name(model_name).replace("/", "-"),
            epochs=config.get("epochs", 2),
            num_neg=config.get("num_neg_samples", 1),
            mode=config.get("mode", "pairwise"),
            loss=config.get("loss_type", "margin"),
            expanded=expanded_suffix,
            sampler=sampler_suffix,  # This sampler suffix is not in the run_name_tpl, but was in the original code's run_name_tpl. # noqa: E501
        )
        try:
            if inference_only:
                results = _run_inference_only(
                    model_name=model_name,
                    config=config,
                    run_name=run_name,
                )
                all_results[model_name] = results["total"]
            else:
                try:
                    wandb.init(
                        project=os.environ.get("WANDB_PROJECT", "bioasq-14b-phaseA-reranker"),
                        name=run_name,
                        id=get_wandb_run_id(run_name),
                        resume="allow",
                    )
                    results = _run_experiment(
                        model_name=model_name,
                        config=config,
                        run_name=run_name,
                        model_path=model_path,
                    )
                    all_results[model_name] = results["total"]
                except KeyboardInterrupt:
                    print(f"Experiment for {model_name} interrupted by user.")
                    print("Doing inference")
                    results = _run_inference_only(
                        model_name=model_name,
                        config=config,
                        run_name=run_name,
                    )
                    all_results[model_name] = results["total"]

        except Exception as e:
            print(f"Failed to run {model_name}: {e}")
            failed_models.append(model_name)
        finally:
            if not inference_only:
                wandb.finish()

    print(f"Failed models: {failed_models}")
    with Path("all_models_evaluation.json").open("a") as f:
        json.dump(all_results, f, indent=4)


@app.command(name="run-llama-experiments")
def run_llama_experiments_command(
    inference_only: bool = typer.Option(
        False, "--inference-only", help="Skip training; run inference only (load from outputs)"
    ),
    config_file: str | None = typer.Option(
        None, "--config", help="Optional path to experiments JSON config"
    ),
) -> None:
    """Run Llama/Nemotron baseline experiments from a config file."""
    set_seed(42)
    if config_file and Path(config_file).exists():
        with Path(config_file).open() as f:
            cfg = json.load(f)
        models_to_test = cfg.get("models_to_test", [])
        config = cfg.get("config", {})
    else:
        # Default fallback corresponding to run_llama_experiments.py
        models_to_test = [
            # "nvidia/llama-nemotron-rerank-1b-v2",
            "/home/ucloud/BioASQ13B/src/bioasq/phase_a/reranker/outputs/nvidia_llama-nemotron-rerank-1b-v2_llama/llama-nemotron-rerank-1b-v2-E2-S4-Mmulti_neg_pairwise-Linfonce-FullData-/checkpoint-1100"
        ]
        config = {
            "mode": "multi_neg_pairwise",
            "num_neg_samples": 4,
            "loss_type": "infonce",
            "infonce_temperature": 0.05,
            "early_stop_on_grad": True,
            "grad_norm_threshold": 1e-6,
            "grad_norm_patience": 5,
            "full_data": True,
            "report_to": "wandb",
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "train_pos_path": PROJECT_DATA_DIR
            / "quality/training14b_inflated_clean_wContents.jsonl",
            "train_neg_path": PROJECT_DATA_DIR / "negatives.jsonl",
            # "expanded_pos_path": PROJECT_DATA_DIR / "quality/training14b_expanded.jsonl",
            "val_files": [
                PROJECT_DATA_DIR / "val_data/13B1_golden.json",
                PROJECT_DATA_DIR / "val_data/13B2_golden.json",
                PROJECT_DATA_DIR / "val_data/13B3_golden.json",
                PROJECT_DATA_DIR / "val_data/13B4_golden.json",
            ],
        }

    failed_models = []
    all_results = {}
    is_full = config.get("full_data", False)
    run_name_tpl = (
        "{model}-E{epochs}-S{num_neg}-M{mode}-L{loss}-FullData-{expanded}"
        if is_full
        else "{model}-E{epochs}-S{num_neg}-M{mode}-L{loss}{expanded}"
    )

    for model_name in models_to_test:
        run_name = run_name_tpl.format(
            model=_short_model_name(model_name).replace("/", "-"),
            epochs=config.get("epochs", 2),
            num_neg=config.get("num_neg_samples", 4),
            mode=config.get("mode", "multi_neg_pairwise"),
            loss=config.get("loss_type", "infonce"),
            expanded="-Expanded" if config.get("expanded_pos_path") else "",
        )
        try:
            if inference_only:
                results = _run_llama_inference_only(
                    model_name=model_name,
                    config=config,
                    run_name=run_name,
                )
                all_results[model_name] = results["total"]
            else:
                setup_wandb(run_name)
                results = _run_llama_experiment(
                    model_name=model_name,
                    config=config,
                    run_name=run_name,
                )
                all_results[model_name] = results["total"]
        except Exception as e:
            print(f"Failed to run {model_name}: {e}")
            failed_models.append(model_name)
        finally:
            if not inference_only:
                wandb.finish()

    print(f"Failed models: {failed_models}")
    with Path("all_models_evaluation_llama.json").open("a") as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    app()
