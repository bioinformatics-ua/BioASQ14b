from pathlib import Path
import argparse
import os

# Set devices if needed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import wandb
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    AutoConfig
)
from torch.utils.data import DataLoader

# PEFT imports for LoRA
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig

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
from utils import set_seed, setup_wandb, build_output_dir_name, create_training_config

BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG = BASE_DIR / "config" / "train_config.yaml"


def _output_dir(model_name: str) -> str:
    return f"./outputs/{model_name.replace('/', '_')}_lora"


def _load_cached_results(model_name: str) -> dict | None:
    result_path = Path(_output_dir(model_name)) / "ranx_results.json"
    if result_path.exists():
        with open(result_path) as f:
            data = json.load(f)
        return data
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
    if (latest / "adapter_model.safetensors").exists() or (latest / "adapter_model.bin").exists():
        return latest
    return None


def run_experiment(model_name: str, config: dict, run_name: str | None = None):
    print(f"--- Starting LoRA experiment for: {model_name} ---")

    # 1. Load Tokenizer & Base Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Ensure labels are set properly for sequence classification
    if getattr(model.config, "num_labels", 1) != 1:
        model.config.num_labels = 1
        model.config.id2label = {0: "SCORE"}
        model.config.label2id = {"SCORE": 0}

    # 2. Apply LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS 
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() # This will show you the massive parameter reduction!

    # 3. Instantiate Components via Factory
    preprocessor = get_preprocessor("basic", tokenizer=tokenizer, max_length=512)
    sampler_cls = get_sampler("shifter")

    iterator = get_iterator(
        mode=config["mode"],
        sample_preprocessing=preprocessor,
        sampler_cls=sampler_cls,
        num_neg_samples=config["num_neg_samples"],
        sampler_kwargs={"max_epoch": config["epochs"]},
    )

    collator = get_collator(mode=config["mode"], tokenizer=tokenizer)
    trainer_cls = get_trainer_cls(mode=config["mode"])

    # 4. Create Datasets
    train_ds, test_ds, eval_pointwise, eval_pairwise, eval_multi_neg = (
        create_bioASQ_datasets(
            positive_data_path=config["train_pos_path"],
            all_data_path=config["train_neg_path"],
            iterator=iterator,
            test_sample_preprocessing=preprocessor,
            val_files=config["val_files"],
        )
    )

    # 5. Set up TrainingArguments
    output_dir = _output_dir(model_name)
    if run_name is None:
        run_name = f"{model_name}-E{config['epochs']}-S{config['num_neg_samples']}-M{config['mode']}-LoRA"
        
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
        remove_unused_columns=False,
        report_to=config["report_to"],
        optim="adamw_torch_fused",      # <--- ADD THIS LINE
        dataloader_num_workers=4,       # <--- ENSURE THIS IS AT LEAST 4
        dataloader_pin_memory=True,     # <--- ADD THIS LINE
    )

    eval_ds = eval_pairwise if config["mode"] == "pairwise" else eval_pointwise

    # 6. Initialize Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        margin=1.0,
    )

    # 7. Train and Save Adapter explicitly to the output_dir root
    trainer.train()
    trainer.save_model(output_dir) # Saves adapter_model.safetensors, adapter_config.json

    # 8. Inference and Evaluation on Test Set
    print(f"Running inference for {model_name}...")
    inference_collator = get_collator(mode="pointwise", tokenizer=tokenizer) 

    test_dataloader = DataLoader(
        test_ds,
        batch_size=config["batch_size"] * 4,
        collate_fn=inference_collator,
        num_workers=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We can use the same `model` because it is already wrapped with the trained PEFT weights in memory.
    run_dict = run_inference(
        model,
        test_dataloader,
        device=device,
        amp_dtype=torch.bfloat16,
    )

    # 9. Evaluate and Save
    qrels_dict = test_ds.get_qrels()
    results = evaluate_run(
        run_dict,
        qrels_dict,
        metrics=DEFAULT_METRICS,
        per_file_results=None,
    )

    save_predictions(run_dict, output_dir)
    metadata = {
        "model": model_name.replace("/", "-"),
        "val_files": config.get("val_files"),
    }
    result_path = f"{output_dir}/ranx_results.json"
    with open(result_path, "w") as f:
        json.dump(metadata | results, f, indent=4)

    print(f"Metrics for {model_name}: {results['total']}\n")
    return results


def run_inference_only(model_name: str, config: dict) -> dict:
    output_dir = _output_dir(model_name)
    model_path = Path(output_dir)

    load_path = model_name
    if model_path.exists():
        latest_ckpt = _find_latest_checkpoint(model_path)
        if latest_ckpt is not None:
            load_path = str(latest_ckpt)
        elif (model_path / "adapter_model.safetensors").exists():
            load_path = str(model_path)

    print(f"--- Inference-only for: {model_name} (loading adapter from {load_path}) ---")

    # 1. Base Model Check
    # We must load the original base model first, then apply the PEFT adapter
    try:
        peft_config = PeftConfig.from_pretrained(load_path)
        base_model_path = peft_config.base_model_name_or_path
    except Exception:
        # Fallback if the path given isn't a peft model, try loading directly
        base_model_path = model_name

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=1,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16,
    )
    
    if getattr(base_model.config, "num_labels", 1) != 1:
        base_model.config.num_labels = 1
        base_model.config.id2label = {0: "SCORE"}
        base_model.config.label2id = {"SCORE": 0}

    # Wrap the base model with the saved LoRA adapter
    if base_model_path != load_path:
        print(f"Applying LoRA adapter from {load_path} to base model {base_model_path}")
        model = PeftModel.from_pretrained(base_model, load_path)
    else:
        model = base_model

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
    results = evaluate_run(
        run_dict,
        qrels_dict,
        metrics=DEFAULT_METRICS,
        per_file_results=None,
    )

    metadata = {
        "model": model_name.replace("/", "-"),
        "val_files": config.get("val_files"),
    }
    result_path = f"{output_dir}/ranx_results.json"
    with open(result_path, "w") as f:
        json.dump(metadata | results, f, indent=4)

    print(f"Metrics for {model_name}: {results['total']}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BioASQ reranker experiments with LoRA")
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Skip training; run inference only for already-trained models (load from outputs/)",
    )
    args = parser.parse_args()

    set_seed(48)
    
    # Just running the Llama model for this LoRA script, but you can add others back.
    MODELS_TO_TEST = [
        "nvidia/llama-nemotron-rerank-1b-v2"
    ]

    CONFIG = {
        "mode": "pairwise",
        "num_neg_samples": 1,
        "report_to": "wandb", # Changed from wandb just to avoid prompt login issues; change back if you need it
        "epochs": 4,
        "batch_size": 16, # With LoRA you can potentially increase this on an A40
        "learning_rate": 1e-4, # Increased LR for LoRA training
        "train_pos_path": "../../data/quality/training14b_inflated_clean_wContents.jsonl",
        "train_neg_path": "../../data/negatives.jsonl",
        "val_files": ["../../data/val_data/13B1_golden.json", "../../data/val_data/13B2_golden.json"],
    }
    
    failed_models = []
    all_results = {}
    run_name_tpl = "{model}-E{epochs}-S{num_neg}-M{mode}-LoRA"
    
    for model_name in MODELS_TO_TEST:
        run_name = run_name_tpl.format(
            model=model_name.replace("/", "-"),
            epochs=CONFIG["epochs"],
            num_neg=CONFIG["num_neg_samples"],
            mode=CONFIG["mode"],
        )
        try:
            if args.inference_only:
                results = run_inference_only(model_name, CONFIG)
                all_results[model_name] = results["total"]
            else:
                cached = _load_cached_results(model_name)
                if cached is not None and not CONFIG.get("force_retrain"):
                    print(f"--- Skipping (cached): {model_name} ---")
                    all_results[model_name] = cached["total"]
                else:
                    setup_wandb(run_name) # Uncomment if using wandb
                    results = run_experiment(model_name, CONFIG, run_name=run_name)
                    all_results[model_name] = results["total"]
        except Exception as e:
            print(f"Failed to run {model_name}: {e}")
            failed_models.append(model_name)
        finally:
            if not args.inference_only:
                wandb.finish()
            
    print(f"Failed models: {failed_models}")
    
    with open("all_models_evaluation_lora.json", "w") as f:
        json.dump(all_results, f, indent=4)