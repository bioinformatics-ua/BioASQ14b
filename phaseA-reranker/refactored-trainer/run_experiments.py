from pathlib import Path
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import wandb
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
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
from evaluation import DEFAULT_METRICS, run_inference, evaluate_run, save_predictions
from utils import set_seed, setup_wandb, get_wandb_run_id, build_output_dir_name, create_training_config

BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG = BASE_DIR / "config" / "train_config.yaml"


def _output_dir(model_name: str, run_name: str | None = None) -> str:
    return f"./outputs/{model_name.replace('/', '_')}/{run_name if run_name else ''}"


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


def _load_qids_per_val_file(val_files: list[str]) -> dict[str, list[str]]:
    """Return {basename: [qid, ...]} for each val file. Same as main.py."""
    per_file: dict[str, list[str]] = {}
    for path in val_files:
        with open(path) as f:
            data = json.load(f)
        qids = [str(q["id"]) for q in data["questions"]]
        per_file[os.path.basename(path)] = qids
    return per_file


def run_experiment(
    model_name: str,
    config: dict,
    run_name: str | None = None,
    model_path: str | None = None,
):
    """
    Args:
        model_name: Base model ID (for tokenizer and output dir). Also used as load path if model_path is None.
        model_path: Optional checkpoint path to load weights from. If set, load model from here instead of model_name.
    """
    load_path = model_path if model_path is not None else model_name
    print(f"--- Starting experiment for: {model_name} (loading from {load_path}) ---")

    # 1. Load Tokenizer & Model
    # Tokenizer from base model (checkpoints may lack full tokenizer files)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        load_path,
        num_labels=1,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        dtype=torch.bfloat16,
    )

    # 2. Instantiate Components via Factory
    preprocessor = get_preprocessor("basic", tokenizer=tokenizer, max_length=512)
    use_expanded_pos = bool(config.get("expanded_pos_path"))
    # Use exponential sampler when training with expanded positives (multi-tier relevance)
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
        run_name = f"{model_name}-E{config['epochs']}-S{config['num_neg_samples']}-M{config['mode']}"
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

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        margin=1.0,  # default from your losses
    )

    # 6. Train
    trainer.train()

    # 7. Inference and Evaluation on Test Set
    print(f"Running inference for {model_name}...")
    inference_collator = get_collator(
        mode="pointwise", tokenizer=tokenizer
    )  # Inference is always pointwise scoring

    if config.get("full_data", True):
        # RECREATE THE TEST DATASET WITH THE VALIDATION
        _, test_ds, _, _, _ = (
            create_bioASQ_datasets(
                positive_data_path=train_pos_path,
                all_data_path=config["train_neg_path"],
                iterator=iterator,
                test_sample_preprocessing=preprocessor,
                val_files=config["val_files"],
                use_expanded_pos=use_expanded_pos,
            )
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
    output_dir = _output_dir(model_name, run_name)
    model_path = Path(_output_dir(model_name, run_name))

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
    model = AutoModelForSequenceClassification.from_pretrained(
        load_path,
        num_labels=1,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16,
    )

    preprocessor = get_preprocessor("basic", tokenizer=tokenizer, max_length=512)
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
    parser = argparse.ArgumentParser(description="Run BioASQ reranker experiments")
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Skip training; run inference only for already-trained models (load from outputs/)",
    )
    args = parser.parse_args()

    set_seed(42)
    # You can loop through a subset first to test
    # Each entry can be:
    #   - str: model name (load from HuggingFace)
    #   - dict: {"model": "base-model-id", "checkpoint": "path/to/checkpoint-500"} to resume from checkpoint
    MODELS_TO_TEST = [
        "ncbi/MedCPT-Cross-Encoder",
        #{"model": "ncbi/MedCPT-Cross-Encoder", "checkpoint": "./outputs-E5-Pairwise/ncbi_MedCPT-Cross-Encoder/checkpoint-6375"},
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        # {"model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", "checkpoint": "./outputs-E5-Pairwise/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/checkpoint-6375"},
        # "dmis-lab/biobert-base-cased-v1.2",
        # "emilyalsentzer/Bio_ClinicalBERT",
        # {"model": "michiyasunaga/BioLinkBERT-base", "checkpoint": "./outputs-E5-Pairwise/michiyasunaga_BioLinkBERT-base/checkpoint-6375"},
        "michiyasunaga/BioLinkBERT-base",
        "michiyasunaga/BioLinkBERT-large",
        "pritamdeka/S-PubMedBert-MS-MARCO",
        #{"model": "pritamdeka/S-PubMedBert-MS-MARCO", "checkpoint": "./outputs-E5-Pairwise/pritamdeka_S-PubMedBert-MS-MARCO/checkpoint-6375"},
        # "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "monologg/biobert_v1.1_pubmed",
        # {"model": "monologg/biobert_v1.1_pubmed", "checkpoint": "./outputs-E5-Pairwise/monologg_biobert_v1.1_pubmed/checkpoint-6375"},
        "nboost/pt-biobert-base-msmarco",
        #{"model": "nboost/pt-biobert-base-msmarco", "checkpoint": "./outputs-E5-Pairwise/nboost_pt-biobert-base-msmarco/checkpoint-6375"},
        # "allenai/specter2_base",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        # {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "checkpoint": "./outputs-E5-Pairwise/cross-encoder_ms-marco-MiniLM-L-6-v2/checkpoint-6375"},
        # "cross-encoder/ms-marco-electra-base",
        "BAAI/bge-reranker-base",
        # {"model": "BAAI/bge-reranker-base", "checkpoint": "./outputs-E5-Pairwise/BAAI_bge-reranker-base/checkpoint-6375"},
        "BAAI/bge-reranker-v2-m3",
        # {"model": "BAAI/bge-reranker-v2-m3", "checkpoint": "./outputs-E5-Pairwise/BAAI_bge-reranker-v2-m3/checkpoint-6375"},
        # Example: start from checkpoint instead of base model:
        # {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "checkpoint": "./outputs/cross-encoder_ms-marco-MiniLM-L-6-v2/checkpoint-500"},
    ]

    CONFIG = {
        "mode": "pairwise",  # "pairwise" | "multi_neg_pairwise" | "pointwise"
        "num_neg_samples": 1,  # For pairwise: 1; for multi_neg_pairwise: 4-20+ depending on GPU memory
        "report_to": "wandb",
        "epochs": 2,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "train_pos_path": "../../data/quality/training14b_inflated_clean_wContents.jsonl",
        "train_neg_path": "../../data/negatives.jsonl",
        # "expanded_pos_path": "../../data/quality/training14b_expanded.jsonl",  # Use when training with expanded positives
        "full_data": True,
        # Train on val (13B1+13B2); hold out 13B3 for evaluation
        "val_files": ["../../data/val_data/13B3_golden.json", "../../data/val_data/13B1_golden.json", "../../data/val_data/13B2_golden.json", "../../data/val_data/13B4_golden.json"],
        # "force_retrain": True,
    }
    failed_models = []
    all_results = {}
    run_name_tpl = "{model}-E{epochs}-S{num_neg}-M{mode}-FullData{full_data}{expanded}"
    for entry in MODELS_TO_TEST:
        if isinstance(entry, dict):
            model_name = entry["model"]
            model_path = entry.get("checkpoint")
        else:
            model_name = entry
            model_path = None

        run_name = run_name_tpl.format(
            model=model_name.replace("/", "-"),
            epochs=CONFIG["epochs"],
            num_neg=CONFIG["num_neg_samples"],
            mode=CONFIG["mode"],
            full_data=CONFIG["full_data"],
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
                    results = run_experiment(
                        model_name, CONFIG, run_name=run_name, model_path=model_path
                    )
                    all_results[model_name] = results["total"]
        except Exception as e:
            print(f"Failed to run {model_name}: {e}")
            failed_models.append(model_name)
        finally:
            if not args.inference_only:
                wandb.finish()
    
    print(f"Failed models: {failed_models}")
    # Save a master JSON with all model comparisons
    with open("all_models_evaluation.json", "a") as f:
        json.dump(all_results, f, indent=4)
