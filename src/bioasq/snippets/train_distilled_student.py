"""Train a distilled student from teacher traces.

This script is task-agnostic: it consumes distillation JSONL with prompt
messages plus a teacher response. Current defaults point at the snippet
distillation data, but the same trainer can later be reused for normal answer
generation by swapping prompts and teacher traces.

Distillation modes:
  - always: sequence-level distillation on teacher responses
  - optional: token-level KL against a live teacher when tokenizers match

Usage:
    python -m bioasq.snippets.train_distilled_student \
        --train-data data/training/distillation/snippets/distill_train.jsonl \
        --val-data data/training/distillation/snippets/distill_val.jsonl \
        --student-model google/gemma-4-27b-it \
        --student-mode lora \
        --teacher-model google/gemma-4-31b-it \
        --kl-weight 0.2
"""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import torch
import torch.nn.functional as F
import typer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

app = typer.Typer()

DEFAULT_LORA_TARGETS = (
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)


@dataclass(slots=True)
class DistillationRecord:
    prompt_messages: list[dict[str, str]]
    teacher_response: str
    task: str | None = None
    metadata: dict[str, Any] | None = None


class DistillationDataset(Dataset):
    """Tokenized distillation records with prompt tokens masked from loss."""

    def __init__(
        self,
        records: list[DistillationRecord],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
    ) -> None:
        self.examples: list[dict[str, torch.Tensor]] = []
        skipped = 0

        for record in records:
            prompt_text = _render_messages(
                tokenizer=tokenizer,
                messages=record.prompt_messages,
                add_generation_prompt=True,
            )
            full_text = _render_messages(
                tokenizer=tokenizer,
                messages=[
                    *record.prompt_messages,
                    {"role": "assistant", "content": record.teacher_response},
                ],
                add_generation_prompt=False,
            )

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError("Prompt tokenization mismatch. Check chat template consistency.")

            if len(full_ids) > max_seq_length:
                full_ids = full_ids[:max_seq_length]

            prompt_len = min(len(prompt_ids), len(full_ids))
            if prompt_len >= len(full_ids):
                skipped += 1
                continue

            input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            attention_mask = torch.ones_like(input_ids)

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

        print(f"Tokenized {len(self.examples)} examples ({skipped} skipped after truncation)")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


class DistillationCollator:
    """Pad causal LM inputs while keeping prompt labels masked."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer needs a pad token before collation.")

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DistillationTrainer(Trainer):
    """Trainer that mixes sequence loss with optional token-level KL."""

    def __init__(
        self,
        *args: Any,
        teacher_model: torch.nn.Module | None = None,
        kl_weight: float = 0.0,
        distill_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.kl_weight = kl_weight
        self.distill_temperature = distill_temperature

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch

        outputs = model(**inputs, use_cache=False)
        loss = outputs.loss

        if self.teacher_model is not None and self.kl_weight > 0.0:
            teacher_device = next(self.teacher_model.parameters()).device
            if teacher_device != inputs["input_ids"].device:
                self.teacher_model.to(inputs["input_ids"].device)

            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                )

            kl_loss = _token_kl_loss(
                student_logits=outputs.logits,
                teacher_logits=teacher_outputs.logits,
                labels=inputs["labels"],
                temperature=self.distill_temperature,
            )
            loss = (1.0 - self.kl_weight) * loss + self.kl_weight * kl_loss

        if return_outputs:
            return loss, outputs
        return loss


def _load_records(path: Path) -> list[DistillationRecord]:
    """Load prompt/teacher traces from JSONL."""
    records: list[DistillationRecord] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            prompt_messages = obj.get("prompt_messages")
            teacher_response = obj.get("teacher_response")

            if prompt_messages is None and "messages" in obj:
                messages = obj["messages"]
                if not messages or messages[-1].get("role") != "assistant":
                    raise ValueError("messages must end with an assistant turn.")
                prompt_messages = messages[:-1]
                teacher_response = messages[-1]["content"]

            if prompt_messages is None or teacher_response is None:
                raise ValueError("Each record needs prompt_messages + teacher_response.")

            records.append(
                DistillationRecord(
                    prompt_messages=prompt_messages,
                    teacher_response=teacher_response,
                    task=obj.get("task"),
                    metadata=obj.get("metadata"),
                )
            )
    return records


def _fallback_render(messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    """Simple text rendering when tokenizer lacks a chat template."""
    chunks = []
    for message in messages:
        role = message["role"].upper()
        content = message["content"].strip()
        chunks.append(f"{role}:\n{content}")
    if add_generation_prompt:
        chunks.append("ASSISTANT:\n")
    return "\n\n".join(chunks)


def _render_messages(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    """Render chat messages using tokenizer template when available."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return _fallback_render(messages, add_generation_prompt)


def _compute_dtype() -> torch.dtype:
    """Prefer bf16 on supported GPUs, else fp16."""
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _build_quant_config(load_in_4bit: bool, compute_dtype: torch.dtype) -> BitsAndBytesConfig | None:
    """Create 4-bit quantization config when requested."""
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _maybe_add_lora(
    model: torch.nn.Module,
    mode: str,
    load_in_4bit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_targets: str,
) -> torch.nn.Module:
    """Attach LoRA adapters when running PEFT distillation."""
    if mode == "full":
        return model

    if lora_r <= 0:
        raise typer.BadParameter("LoRA mode needs --lora-r > 0.")

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = [item.strip() for item in lora_targets.split(",") if item.strip()]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def _tokenizers_match(
    student_tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase,
) -> bool:
    """Check whether teacher/student share token ids for KL distillation."""
    return (
        student_tokenizer.get_vocab() == teacher_tokenizer.get_vocab()
        and student_tokenizer.all_special_ids == teacher_tokenizer.all_special_ids
    )


def _token_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL loss only on supervised response tokens."""
    shift_mask = labels[..., 1:] != -100
    if not torch.any(shift_mask):
        return student_logits.new_zeros(())

    scaled_student = F.log_softmax(student_logits[..., :-1, :] / temperature, dim=-1)
    scaled_teacher = F.softmax(teacher_logits[..., :-1, :] / temperature, dim=-1)

    student_selected = scaled_student[shift_mask]
    teacher_selected = scaled_teacher[shift_mask]
    return F.kl_div(student_selected, teacher_selected, reduction="batchmean") * (temperature**2)


def _supports_kwarg(callable_obj: Any, kwarg_name: str) -> bool:
    """Return whether a callable explicitly supports a named keyword arg."""
    return kwarg_name in inspect.signature(callable_obj).parameters


@app.command()
def main(
    student_model: Annotated[
        str, typer.Option(help="Student model id or local path")
    ] = "google/gemma-4-27b-it",
    train_data: Annotated[
        Path, typer.Option(help="Distillation train JSONL")
    ] = Path("data/training/distillation/snippets/distill_train.jsonl"),
    val_data: Annotated[
        Path, typer.Option(help="Distillation val JSONL")
    ] = Path("data/training/distillation/snippets/distill_val.jsonl"),
    output_dir: Annotated[
        Path, typer.Option(help="Directory for checkpoints and final student")
    ] = Path("data/training/distillation/snippets/student_output"),
    student_mode: Annotated[
        str, typer.Option(help="Student update mode: lora or full")
    ] = "lora",
    load_in_4bit: Annotated[
        bool, typer.Option(help="Load student in 4-bit. Best paired with LoRA mode.")
    ] = True,
    lora_r: Annotated[int, typer.Option()] = 64,
    lora_alpha: Annotated[int, typer.Option()] = 128,
    lora_dropout: Annotated[float, typer.Option()] = 0.05,
    lora_targets: Annotated[
        str, typer.Option(help="Comma-separated module names for LoRA")
    ] = DEFAULT_LORA_TARGETS,
    teacher_model: Annotated[
        str, typer.Option(help="Optional live teacher model for token-level KL")
    ] = "",
    teacher_tokenizer: Annotated[
        str, typer.Option(help="Optional tokenizer path for teacher compatibility check")
    ] = "",
    teacher_load_in_4bit: Annotated[
        bool, typer.Option(help="Load live teacher in 4-bit if KL is enabled")
    ] = True,
    kl_weight: Annotated[
        float, typer.Option(help="Mix weight for token-level KL against live teacher")
    ] = 0.0,
    distill_temperature: Annotated[
        float, typer.Option(help="Temperature for token-level KL distillation")
    ] = 2.0,
    epochs: Annotated[int, typer.Option()] = 2,
    batch_size: Annotated[int, typer.Option()] = 1,
    gradient_accumulation: Annotated[int, typer.Option()] = 8,
    lr: Annotated[float, typer.Option()] = 2e-4,
    warmup_ratio: Annotated[float, typer.Option()] = 0.05,
    max_seq_length: Annotated[int, typer.Option()] = 2048,
    logging_steps: Annotated[int, typer.Option()] = 10,
    save_steps: Annotated[int, typer.Option()] = 250,
    eval_steps: Annotated[int, typer.Option()] = 250,
    report_to: Annotated[str, typer.Option()] = "wandb",
    wandb_project: Annotated[str, typer.Option()] = "bioasq-distillation",
    seed: Annotated[int, typer.Option()] = 3407,
) -> None:
    """Train a distilled student on teacher traces."""
    if student_mode not in {"lora", "full"}:
        raise typer.BadParameter("--student-mode must be 'lora' or 'full'.")
    if student_mode == "full" and load_in_4bit:
        raise typer.BadParameter("Full finetuning cannot run with --load-in-4bit.")
    if not 0.0 <= kl_weight <= 1.0:
        raise typer.BadParameter("--kl-weight must be between 0 and 1.")
    if load_in_4bit and not torch.cuda.is_available():
        raise typer.BadParameter("4-bit student loading needs CUDA.")

    os.environ["WANDB_PROJECT"] = wandb_project
    torch.manual_seed(seed)

    compute_dtype = _compute_dtype()
    quant_config = _build_quant_config(load_in_4bit, compute_dtype)

    print(f"Loading student {student_model} (mode={student_mode}, 4bit={load_in_4bit})")
    tokenizer = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(
        student_model,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        quantization_config=quant_config,
        device_map="auto" if load_in_4bit else None,
    )
    student.config.use_cache = False
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
    student = _maybe_add_lora(
        model=student,
        mode=student_mode,
        load_in_4bit=load_in_4bit,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets,
    )

    train_records = _load_records(train_data)
    val_records = _load_records(val_data)
    print(f"Loaded {len(train_records)} train records and {len(val_records)} val records")

    train_dataset = DistillationDataset(train_records, tokenizer, max_seq_length)
    val_dataset = DistillationDataset(val_records, tokenizer, max_seq_length)
    if len(train_dataset) == 0:
        raise typer.BadParameter("No train examples left after tokenization/truncation.")
    if len(val_dataset) == 0:
        raise typer.BadParameter("No val examples left after tokenization/truncation.")

    live_teacher: torch.nn.Module | None = None
    if teacher_model and kl_weight > 0.0:
        if teacher_load_in_4bit and not torch.cuda.is_available():
            raise typer.BadParameter("4-bit teacher loading needs CUDA.")

        teacher_tokenizer_name = teacher_tokenizer or teacher_model
        teacher_tok = AutoTokenizer.from_pretrained(teacher_tokenizer_name, trust_remote_code=True)
        if not _tokenizers_match(tokenizer, teacher_tok):
            print("Teacher tokenizer mismatch. Disabling token-level KL; keeping response distillation only.")
            kl_weight = 0.0
        else:
            print(f"Loading live teacher {teacher_model} for token-level KL")
            live_teacher = AutoModelForCausalLM.from_pretrained(
                teacher_model,
                trust_remote_code=True,
                torch_dtype=compute_dtype,
                quantization_config=_build_quant_config(teacher_load_in_4bit, compute_dtype),
                device_map="auto" if teacher_load_in_4bit else None,
            )
            live_teacher.config.use_cache = False
            live_teacher.eval()
            live_teacher.requires_grad_(False)
    elif kl_weight > 0.0:
        print("No live teacher model provided. Using sequence-level response distillation only.")
        kl_weight = 0.0

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "learning_rate": lr,
        "warmup_ratio": warmup_ratio,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "logging_steps": logging_steps,
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": 3,
        "report_to": report_to,
        "run_name": output_dir.name,
        "seed": seed,
        "bf16": torch.cuda.is_available() and compute_dtype == torch.bfloat16,
        "fp16": torch.cuda.is_available() and compute_dtype == torch.float16,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit" if load_in_4bit else "adamw_torch",
        "remove_unused_columns": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    if _supports_kwarg(TrainingArguments.__init__, "eval_strategy"):
        training_kwargs["eval_strategy"] = "steps"
    else:
        training_kwargs["evaluation_strategy"] = "steps"
    if _supports_kwarg(TrainingArguments.__init__, "save_safetensors"):
        training_kwargs["save_safetensors"] = True

    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": student,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": DistillationCollator(tokenizer),
        "teacher_model": live_teacher,
        "kl_weight": kl_weight,
        "distill_temperature": distill_temperature,
    }
    if _supports_kwarg(Trainer.__init__, "processing_class"):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = DistillationTrainer(**trainer_kwargs)

    print("Starting student distillation...")
    trainer.train()

    final_path = output_dir / "final_student"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Saved distilled student to {final_path}")


if __name__ == "__main__":
    app()