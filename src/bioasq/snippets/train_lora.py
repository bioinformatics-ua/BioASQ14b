"""QLoRA fine-tuning for snippet extraction. ..."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from trl.trainer.sft_config import SFTConfig

app = typer.Typer()


@dataclass
class Gemma4TextCollator:
    """Pads a batch and injects zero token_type_ids + mm_token_type_ids."""

    tokenizer: object

    def __call__(self, features):
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
            "mm_token_type_ids": [],
            "labels": [],
        }
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append([1] * len(f["input_ids"]) + [0] * pad_len)
            batch["token_type_ids"].append([0] * max_len)
            batch["mm_token_type_ids"].append([0] * max_len)
            # Mask padding in labels with -100
            labels = f.get("labels", f["input_ids"])
            batch["labels"].append(labels + [-100] * pad_len)
        return {k: torch.tensor(v) for k, v in batch.items()}


@app.command()
def main(
    base_model: Annotated[str, typer.Option()] = "google/gemma-3-27b-it",
    tokenizer_name: Annotated[str, typer.Option()] = "",
    train_data: Annotated[Path, typer.Option()] = Path(
        "data/training/snippet_extraction/chat_train.jsonl"
    ),
    val_data: Annotated[Path, typer.Option()] = Path(
        "data/training/snippet_extraction/chat_val.jsonl"
    ),
    output_dir: Annotated[Path, typer.Option()] = Path(
        "data/training/snippet_extraction/lora_output"
    ),
    lora_r: Annotated[int, typer.Option()] = 16,
    lora_alpha: Annotated[int, typer.Option()] = 32,
    lora_dropout: Annotated[float, typer.Option()] = 0.05,
    epochs: Annotated[int, typer.Option()] = 3,
    batch_size: Annotated[int, typer.Option()] = 1,
    gradient_accumulation: Annotated[int, typer.Option()] = 8,
    lr: Annotated[float, typer.Option()] = 1e-4,
    max_seq_length: Annotated[int, typer.Option()] = 2048,
    warmup_ratio: Annotated[float, typer.Option()] = 0.05,
    use_4bit: Annotated[bool, typer.Option()] = True,
    bf16: Annotated[bool, typer.Option()] = True,
    attn_impl: Annotated[str, typer.Option()] = "flash_attention_2",
    wandb_project: Annotated[str, typer.Option()] = "bioasq-snippets",
    seed: Annotated[int, typer.Option()] = 42,
) -> None:
    import json
    import os

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedTokenizerBase,
    )
    from trl import SFTTrainer

    torch.manual_seed(seed)

    # ---- Load data ----
    def _load_chat_jsonl(path: Path) -> Dataset:
        records = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return Dataset.from_list(records)

    # ---- Precision ----
    def _resolve_precision(use_bf16: bool) -> tuple[bool, bool, torch.dtype]:
        if not use_bf16:
            return False, True, torch.float16
        bf16_supported = False
        if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported"):
            try:
                bf16_supported = bool(torch.cuda.is_bf16_supported())
            except Exception:
                bf16_supported = False
        if not bf16_supported:
            print("bf16 requested but unavailable; falling back to fp16.")
            return False, True, torch.float16
        return True, False, torch.bfloat16

    # ---- Tokenizer ----
    def _load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
        try:
            return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except AttributeError as exc:
            if "'list' object has no attribute 'keys'" not in str(exc):
                raise
            print("Tokenizer config has incompatible extra_special_tokens; retrying.")
        try:
            return AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, extra_special_tokens={}
            )
        except Exception as exc:
            print(f"Fast tokenizer load failed ({exc}); retrying with slow tokenizer.")
            return AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, extra_special_tokens={}, use_fast=False
            )

    bf16_enabled, fp16_enabled, model_dtype = _resolve_precision(bf16)
    print(f"Using precision: {'bf16' if bf16_enabled else 'fp16'} (model dtype: {model_dtype})")

    tokenizer_source = tokenizer_name or base_model
    print(f"Loading tokenizer from {tokenizer_source}")
    tokenizer = _load_tokenizer(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.environ["WANDB_PROJECT"] = wandb_project

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        dtype=model_dtype,  # <-- was torch_dtype, now dtype
        torch_dtype=model_dtype,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # ---- LoRA config ----
    module_names = {name for name, _ in model.named_modules()}
    wrapped_targets = ["q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear"]
    plain_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if all(any(name.endswith(t) for name in module_names) for t in wrapped_targets):
        target_modules = wrapped_targets
        print(f"Using wrapped LoRA targets: {target_modules}")
    else:
        target_modules = plain_targets
        print(f"Using standard LoRA targets: {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Pre-tokenize dataset (injects mm_token_type_ids) ----
    # SFTTrainer must receive already-tokenized data + custom collator;
    # dataset_text_field=None disables SFTTrainer's internal tokenization.
    def tokenize(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        enc = tokenizer(text, truncation=True, max_length=max_seq_length)
        enc["token_type_ids"] = [0] * len(enc["input_ids"])
        enc["mm_token_type_ids"] = [0] * len(enc["input_ids"])
        enc["labels"] = enc["input_ids"].copy()
        return enc

    raw_train = _load_chat_jsonl(train_data)
    raw_val = _load_chat_jsonl(val_data)
    print(f"Train: {len(raw_train)}, Val: {len(raw_val)}")

    train_ds = raw_train.map(tokenize, remove_columns=raw_train.column_names)
    val_ds = raw_val.map(tokenize, remove_columns=raw_val.column_names)

    # ---- Training arguments ----
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=f"snippet-lora-r{lora_r}-lr{lr}",
        seed=seed,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        remove_unused_columns=False,  # <-- critical: keeps mm_token_type_ids alive
        max_length=max_seq_length,
    )

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=Gemma4TextCollator(tokenizer),  # <-- injects zero tensors
    )

    print("Starting training...")
    trainer.train()

    adapter_path = output_dir / "final_adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Adapter saved to {adapter_path}")


if __name__ == "__main__":
    app()
