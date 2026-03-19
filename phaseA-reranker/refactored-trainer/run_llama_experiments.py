"""
Run BioASQ reranker experiments for Llama/Nemotron models.

This is a copy of run_experiments.py with Nemotron-specific changes:
- Nemotron prompt template: question:{query} \\n \\n passage:{passage}
- Tokenizer: padding_side="left", pad_token=eos_token if None
- Model: pad_token_id, _sanitize_position_ids_buffers for RoPE
- Outputs go to outputs/{model}_llama/ to avoid clashing with run_experiments
"""
from pathlib import Path
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import wandb
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from torch.utils.data import DataLoader

# Import from your modules
from data import create_bioASQ_datasets
from factory import (
    get_sampler,
    get_preprocessor,
    get_collator,
    get_iterator,
    get_trainer_cls,
)
from trainer import EarlyStoppingOnGradNorm
from evaluation import DEFAULT_METRICS, run_inference, evaluate_run, save_predictions
from utils import set_seed, setup_wandb, get_wandb_run_id, build_output_dir_name, create_training_config

BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG = BASE_DIR / "config" / "train_config.yaml"


def _output_dir(model_name: str, run_name: str | None = None) -> str:
    """Output dir with _llama suffix to avoid clashing with run_experiments outputs."""
    base = f"./outputs/{model_name.replace('/', '_')}_llama"
    return f"{base}/{run_name}" if run_name else base


def _load_cached_results(model_name: str, run_name: str | None = None) -> dict | None:
    """If model was already trained and evaluated, return results; else None."""
    result_path = Path(_output_dir(model_name, run_name)) / "ranx_results.json"
    if result_path.exists():
        with open(result_path) as f:
            data = json.load(f)
        return data
    return None


def _find_latest_checkpoint(out_dir: Path) -> Path | None:
    """Return path to latest checkpoint directory (with model.safetensors)."""
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


def _find_trainer_state(model_name: str, run_name: str | None = None) -> Path | None:
    """Return path to trainer_state.json with full log_history (latest checkpoint)."""
    out_dir = Path(_output_dir(model_name, run_name))
    if not out_dir.exists():
        return None
    # Check root first (written at end of training)
    root_state = out_dir / "trainer_state.json"
    if root_state.exists():
        return root_state
    # Else use checkpoint with highest step
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
    """Replay Trainer log_history to current WandB run."""
    with open(state_path) as f:
        state = json.load(f)
    log_history = state.get("log_history", [])
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        # Log only scalar metrics (skip non-numeric)
        metrics = {k: v for k, v in entry.items() if k != "step" and isinstance(v, (int, float))}
        if metrics:
            wandb.log(metrics, step=step)


def _log_results_to_wandb(run_name: str, results: dict, model_name: str) -> None:
    """Create or update a WandB run, replay training history if available, then log final metrics."""
    setup_wandb(run_name)
    run_id = get_wandb_run_id(run_name)
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "bioasq-14b-phaseA-reranker"),
        name=run_name,
        id=run_id,
        resume="allow",  # Update existing run if it exists, else create
        reinit=False,
    )
    state_path = _find_trainer_state(model_name, run_name)
    if state_path is not None:
        _replay_log_history_to_wandb(state_path)
    if "total" in results:
        wandb.summary.update(results["total"])
    wandb.finish()


def _sanitize_position_ids_buffers(model: torch.nn.Module) -> None:
    """Ensure any `position_ids` buffers are monotonic 0..N-1.

    Some remote-code checkpoints (e.g. Llama-based) may load a corrupted
    `position_ids` buffer, which causes out-of-bounds indexing in RoPE.
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


def _setup_nemotron_tokenizer(tokenizer: PreTrainedTokenizerBase) -> None:
    """Apply Nemotron reranker tokenizer settings per model README."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _setup_nemotron_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
    """Apply Nemotron model config (pad_token_id) and label head sanitization."""
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    if getattr(model.config, "num_labels", 1) != 1:
        model.config.num_labels = 1
        model.config.id2label = {0: "SCORE"}
        model.config.label2id = {"SCORE": 0}


def _load_qids_per_val_file(val_files: list[str]) -> dict[str, list[str]]:
    """Return {basename: [qid, ...]} for each val file. Same as main.py."""
    per_file: dict[str, list[str]] = {}
    for path in val_files:
        with open(path) as f:
            data = json.load(f)
        qids = [str(q["id"]) for q in data["questions"]]
        per_file[os.path.basename(path)] = qids
    return per_file


def run_experiment(model_name: str, config: dict, run_name: str | None = None):
    print(f"--- Starting Llama/Nemotron experiment for: {model_name} ---")

    # 1. Load Tokenizer & Model (Nemotron-specific setup)
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

    # 2. Instantiate Components (Nemotron preprocessor: question:/passage: format)
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

    # 3. Create Datasets
    train_pos_path = config.get("expanded_pos_path") or config["train_pos_path"]
    train_ds, test_ds, eval_pointwise, eval_pairwise, eval_multi_neg = (
        create_bioASQ_datasets(
            positive_data_path=train_pos_path,
            all_data_path=config["train_neg_path"],
            iterator=iterator,
            test_sample_preprocessing=preprocessor,
            val_files=config["val_files"] if config.get("full_data", False) else None,
            use_expanded_pos=use_expanded_pos,
        )
    )

    # 4. Set up TrainingArguments
    output_dir = _output_dir(model_name, run_name)
    if run_name is None:
        loss = config.get("loss_type", "margin")
        run_name = f"{model_name}-E{config['epochs']}-S{config['num_neg_samples']}-M{config['mode']}-L{loss}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"] * 2,
        learning_rate=config["learning_rate"],
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        remove_unused_columns=False,  # Crucial for your custom dict inputs
        report_to=config["report_to"],  # change to "wandb" if you run setup_wandb
    )

    # 5. Initialize Trainer
    # Choose the correct eval dataset based on your mode
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

    # loss_type and infonce_temperature only apply to MultiNegativePairwiseRerankerTrainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        margin=1.0,
        callbacks=callbacks,
    )
    if config["mode"] == "multi_neg_pairwise":
        trainer_kwargs["loss_type"] = loss_type
        trainer_kwargs["infonce_temperature"] = infonce_temperature

    trainer = trainer_cls(**trainer_kwargs)

    # 6. Train
    trainer.train()

    # 7. Inference and Evaluation on Test Set
    print(f"Running inference for {model_name}...")
    inference_collator = get_collator(
        mode="pointwise", tokenizer=tokenizer
    )  # Inference is always pointwise scoring

    if config.get("full_data", True):
        # Recreate test dataset with validation data for inference (when trained on full data, test_ds was empty)
        _, test_ds, _, _, _ = create_bioASQ_datasets(
            positive_data_path=train_pos_path,
            all_data_path=config["train_neg_path"],
            iterator=iterator,
            test_sample_preprocessing=preprocessor,
            val_files=config["val_files"],
            use_expanded_pos=use_expanded_pos,
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
        test_dataloader,
        device=device,
        amp_dtype=torch.bfloat16,
    )

    # 8. Evaluate with ranx (same format as main.py inference)
    qrels_dict = test_ds.get_qrels()
    val_files_list = config.get("val_files") or []
    per_file = _load_qids_per_val_file(val_files_list) if val_files_list else None
    results = evaluate_run(
        run_dict,
        qrels_dict,
        metrics=DEFAULT_METRICS,
        per_file_results=per_file,
    )

    # 9. Save predictions and results
    save_predictions(run_dict, output_dir)
    metadata = {
        "model": model_name.replace("/", "-"),
        "val_files": val_files_list,
    }
    result_path = f"{output_dir}/ranx_results.json"
    with open(result_path, "w") as f:
        json.dump(metadata | results, f, indent=4)

    print(f"Metrics for {model_name}: {results['total']}\n")
    return results


def run_inference_only(model_name: str, config: dict, run_name: str | None = None) -> dict:
    """Run inference (and evaluation). Skips training.

    Loads from output_dir checkpoint if found; otherwise uses the base model.
    Predictions and results are saved to output_dir (created if needed).
    """
    if run_name is None:
        run_name = f"{model_name}-E{config['epochs']}-S{config['num_neg_samples']}-M{config['mode']}-L{config.get('loss_type', 'margin')}"
    output_dir = _output_dir(model_name, run_name)
    model_path = Path(output_dir)

    # Load from latest checkpoint if present; else use base model
    load_path = model_name
    if model_path.exists():
        latest_ckpt = _find_latest_checkpoint(model_path)
        if latest_ckpt is not None:
            load_path = str(latest_ckpt)
        elif (model_path / "model.safetensors").exists() or (model_path / "pytorch_model.bin").exists():
            load_path = str(model_path)

    print(f"--- Inference-only for: {model_name} (loading from {load_path}) ---")

    # Load tokenizer from base model (checkpoint may lack full tokenizer files)
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

    _, test_ds, _, _, _ = create_bioASQ_datasets(
        positive_data_path=config["train_pos_path"],
        all_data_path=config["train_neg_path"],
        iterator=iterator,
        test_sample_preprocessing=preprocessor,
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
        test_dataloader,
        device=device,
        amp_dtype=torch.bfloat16,
    )

    save_predictions(run_dict, output_dir)

    qrels_dict = test_ds.get_qrels()
    val_files_list = config.get("val_files") or []
    per_file = _load_qids_per_val_file(val_files_list) if val_files_list else None
    results = evaluate_run(
        run_dict,
        qrels_dict,
        metrics=DEFAULT_METRICS,
        per_file_results=per_file,
    )

    metadata = {
        "model": model_name.replace("/", "-"),
        "val_files": val_files_list,
    }
    result_path = f"{output_dir}/ranx_results.json"
    with open(result_path, "w") as f:
        json.dump(metadata | results, f, indent=4)

    print(f"Metrics for {model_name}: {results['total']}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run BioASQ reranker experiments for Llama/Nemotron models (nvidia/llama-nemotron-rerank-1b-v2)"
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Skip training; run inference only (load from outputs/{model}_llama/)",
    )
    args = parser.parse_args()

    set_seed(42)
    # You can loop through a subset first to test
    MODELS_TO_TEST = [
        "nvidia/llama-nemotron-rerank-1b-v2",

        
    ]

    CONFIG = {
        "mode": "multi_neg_pairwise",  # "pairwise" | "multi_neg_pairwise" | "pointwise"
        "num_neg_samples": 4,  # For pairwise: 1; for multi_neg_pairwise: 4-20+ depending on GPU memory
        "loss_type": "infonce",  # "margin" (hinge, can saturate) | "infonce" (always has gradient)
        "infonce_temperature": 0.05,  # For InfoNCE: smaller = sharper
        "early_stop_on_grad": True,  # Stop when grad_norm ~0 for N logs (margin loss only)
        "grad_norm_threshold": 1e-6,
        "grad_norm_patience": 5,
        "full_data": True,  # True: hold out val for test, recreate test_ds for inference. False: train on all, recreate test_ds for inference.
        "report_to": "wandb",
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "train_pos_path": "../../data/quality/training14b_inflated_clean_wContents.jsonl",
        "train_neg_path": "../../data/negatives.jsonl",
        "expanded_pos_path": "../../data/quality/training14b_expanded.jsonl",  # Use when training with expanded positives
        "val_files": ["../../data/val_data/13B3_golden.json", "../../data/val_data/13B1_golden.json", "../../data/val_data/13B2_golden.json", "../../data/val_data/13B4_golden.json"],

        # "force_retrain": True,
    }
    failed_models = []
    all_results = {}
    run_name_tpl = "{model}-E{epochs}-S{num_neg}-M{mode}-L{loss}-FullData-{expanded}" if CONFIG.get("full_data") else "{model}-E{epochs}-S{num_neg}-M{mode}-L{loss}{expanded}"
    for model_name in MODELS_TO_TEST:
        run_name = run_name_tpl.format(
            model=model_name.replace("/", "-"),
            epochs=CONFIG["epochs"],
            num_neg=CONFIG["num_neg_samples"],
            mode=CONFIG["mode"],
            loss=CONFIG.get("loss_type", "margin"),
            expanded="-Expanded" if CONFIG.get("expanded_pos_path") else "",
        )
        try:
            if args.inference_only:
                results = run_inference_only(model_name, CONFIG, run_name=run_name)
                all_results[model_name] = results["total"]
            else:
                cached = None if CONFIG.get("force_retrain") else _load_cached_results(model_name, run_name)
                if cached is not None:
                    print(f"--- Skipping (cached): {model_name} ---")
                    _log_results_to_wandb(run_name, cached, model_name)
                    all_results[model_name] = cached["total"]
                else:
                    setup_wandb(run_name)
                    results = run_experiment(model_name, CONFIG, run_name=run_name)
                    all_results[model_name] = results["total"]
        except Exception as e:
            print(f"Failed to run {model_name}: {e}")
            failed_models.append(model_name)
        finally:
            if not args.inference_only:
                wandb.finish()
    
    print(f"Failed models: {failed_models}")
    # Save a master JSON with all model comparisons
    with open("all_models_evaluation_llama.json", "a") as f:
        json.dump(all_results, f, indent=4)
