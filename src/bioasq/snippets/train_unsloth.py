"""LoRA fine-tuning for snippet extraction using Unsloth.

Uses Unsloth's FastModel for ~1.5x faster training with ~60% less VRAM.
Follows the official Unsloth Gemma-4 recipe:
https://unsloth.ai/docs/models/gemma-4/train
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import torch
import typer
import unsloth  # noqa: F401 -- ensures custom PEFT config is registered before importing SFTTrainer
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

app = typer.Typer()


@app.command()
def main(
    base_model: Annotated[
        str, typer.Option(help="HF model id or local path")
    ] = "unsloth/gemma-4-31B",
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
    lora_dropout: Annotated[float, typer.Option()] = 0.0,
    epochs: Annotated[int, typer.Option()] = 3,
    max_steps: Annotated[int, typer.Option(help="-1 uses epochs instead")] = -1,
    batch_size: Annotated[int, typer.Option()] = 1,
    gradient_accumulation: Annotated[int, typer.Option()] = 8,
    lr: Annotated[float, typer.Option()] = 2e-4,
    max_seq_length: Annotated[int, typer.Option()] = 2048,
    warmup_ratio: Annotated[float, typer.Option()] = 0.05,
    load_in_4bit: Annotated[bool, typer.Option()] = True,
    chat_template: Annotated[str, typer.Option(help="gemma-4 or gemma-4-thinking")] = "gemma-4",
    wandb_project: Annotated[str, typer.Option()] = "bioasq-snippets-unsloth",
    report_to: Annotated[str, typer.Option()] = "wandb",
    save_gguf: Annotated[bool, typer.Option(help="Also export GGUF q4_k_m")] = False,
    seed: Annotated[int, typer.Option()] = 3407,
) -> None:

    torch.manual_seed(seed)
    os.environ["WANDB_PROJECT"] = wandb_project

    # ------------------------------------------------------------------
    # 1. Load model with Unsloth
    # ------------------------------------------------------------------
    print(f"Loading {base_model} (4bit={load_in_4bit}, max_seq_length={max_seq_length})")
    model, tokenizer = FastModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect bf16/fp16
        load_in_4bit=load_in_4bit,
        full_finetuning=False,
    )

    # ------------------------------------------------------------------
    # 2. Attach LoRA adapters
    # ------------------------------------------------------------------
    # model = FastModel.get_peft_model(
    #     model,
    #     finetune_vision_layers=False,
    #     finetune_language_layers=True,
    #     finetune_attention_modules=True,
    #     finetune_mlp_modules=True,
    #     r=lora_r,
    #     lora_alpha=lora_alpha,
    #     lora_dropout=lora_dropout,
    #     bias="none",
    #     random_state=seed,
    # )

    # ------------------------------------------------------------------
    # 3. Apply chat template
    # ------------------------------------------------------------------
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    # ------------------------------------------------------------------
    # 4. Load & format dataset
    # ------------------------------------------------------------------
    def _load_chat_jsonl(path: Path) -> list[dict]:
        records = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    raw_train = _load_chat_jsonl(train_data)
    raw_val = _load_chat_jsonl(val_data)
    print(f"Train: {len(raw_train)}, Val: {len(raw_val)}")

    # Apply chat template eagerly (avoids pickling the tokenizer in dataset.map)
    def _apply_template(records: list[dict]) -> Dataset:
        texts = []
        for rec in records:
            text = tokenizer.apply_chat_template(
                rec["messages"], tokenize=False, add_generation_prompt=False
            )
            texts.append(text.removeprefix("<bos>"))
        return Dataset.from_dict({"text": texts})

    train_ds = _apply_template(raw_train)
    val_ds = _apply_template(raw_val)

    # ------------------------------------------------------------------
    # 5. Trainer
    # ------------------------------------------------------------------
    sft_args = SFTConfig(
        dataset_text_field="text",
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        optim="adamw_8bit",
        weight_decay=0.001,
        logging_steps=1,
        eval_strategy="steps" if max_steps <= 0 else "no",
        eval_steps=500 if max_steps <= 0 else None,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,
        load_best_model_at_end=max_steps <= 0,
        metric_for_best_model="eval_loss" if max_steps <= 0 else None,
        greater_is_better=False if max_steps <= 0 else None,
        report_to=report_to,
        run_name=f"snippet-unsloth-r{lora_r}-lr{lr}",
        seed=seed,
        max_length=max_seq_length,
        dataset_num_proc=1,  # avoid multiprocessing pickle of Unsloth tokenizer
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_args,
    )

    # ------------------------------------------------------------------
    # 6. Mask user/system tokens — train only on model responses
    # ------------------------------------------------------------------
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    # Verify masking on one example
    sample_labels = trainer.train_dataset[0]["labels"]
    n_masked = sum(1 for x in sample_labels if x == -100)
    n_total = len(sample_labels)
    print(f"Masking check: {n_masked}/{n_total} tokens masked ({100 * n_masked / n_total:.1f}%)")

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    print("Starting Unsloth training...")
    # trainer_stats = trainer.train()
    # print(f"Training complete. Loss: {trainer_stats.training_loss:.4f}")

    # ------------------------------------------------------------------
    # 8. Save adapter
    # ------------------------------------------------------------------
    adapter_path = output_dir / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"LoRA adapter saved to {adapter_path}")

    # ------------------------------------------------------------------
    # 9. Optional GGUF export
    # ------------------------------------------------------------------
    if save_gguf:
        gguf_dir = output_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        print(f"Exporting GGUF (q4_k_m) to {gguf_dir} ...")
        model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method="q4_k_m")
        print("GGUF export done.")


if __name__ == "__main__":
    app()
