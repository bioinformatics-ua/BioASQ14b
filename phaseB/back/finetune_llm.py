#!/usr/bin/env python3
"""
LoRA finetuning script for BioASQ-style QA with retrieved-document context.

Supported data formats:
  - JSONL: one sample per line
  - JSON: list[dict] or {"questions": list[dict]}

Expected fields in each sample:
  - body (question text)
  - documents (list of {"text": ...} or raw strings)
  - ideal_answer (string or list of strings)
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


SYSTEM_PROMPT = (
    "You are a biomedical expert. Use the provided documents to answer the question "
    "accurately and concisely. If evidence is insufficient, say so explicitly."
)


@dataclass
class Record:
    question: str
    context: str
    answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune biomedical LLMs with BioASQ-style context QA data."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to input dataset (.json or .jsonl).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/medgemma-4b-it",
        help="Base model name (e.g. google/medgemma-4b-it, luisasousa/qwen35-pubmedqa).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="phaseB/outputs/finetune",
        help="Directory where adapter and tokenizer are saved.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=5,
        help="Maximum documents to include as context per question.",
    )
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=1800,
        help="Maximum characters per document to keep prompt length under control.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum tokenized sequence length.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.98,
        help="Fraction of examples to use for training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument("--epochs", type=float, default=2.0, help="Training epochs.")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="Learning rate."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Evaluate and save every N steps.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load base model in 4-bit (QLoRA style, requires bitsandbytes).",
    )
    return parser.parse_args()


def _read_json_data(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        content = json.load(f)

    if isinstance(content, list):
        return content
    if isinstance(content, dict) and "questions" in content:
        return content["questions"]
    raise ValueError("Unsupported JSON schema. Expected list or {'questions': [...]} .")


def _read_jsonl_data(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_raw_examples(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl_data(path)
    if path.suffix.lower() == ".json":
        return _read_json_data(path)
    raise ValueError("Input must be .json or .jsonl")


def _extract_doc_texts(raw_documents: object, max_docs: int, max_doc_chars: int) -> List[str]:
    if not isinstance(raw_documents, list):
        return []

    docs: List[str] = []
    for doc in raw_documents[:max_docs]:
        text = ""
        if isinstance(doc, dict):
            text = str(doc.get("text", "")).strip()
        elif isinstance(doc, str):
            text = doc.strip()
        if not text:
            continue
        docs.append(text[:max_doc_chars])
    return docs


def _extract_answer(raw_answer: object) -> str:
    if raw_answer is None:
        return ""
    if isinstance(raw_answer, list):
        cleaned = [str(x).strip() for x in raw_answer if str(x).strip()]
        return " ".join(cleaned)
    return str(raw_answer).strip()


def normalize_records(
    samples: Iterable[Dict], max_docs: int, max_doc_chars: int
) -> List[Record]:
    records: List[Record] = []
    for item in samples:
        question = str(item.get("body", "")).strip()
        answer = _extract_answer(item.get("ideal_answer"))
        docs = _extract_doc_texts(item.get("documents"), max_docs, max_doc_chars)

        if not question or not answer or not docs:
            continue

        context = "\n\n".join(
            f"[Document {idx + 1}]\n{txt}" for idx, txt in enumerate(docs)
        )
        records.append(Record(question=question, context=context, answer=answer))
    return records


def format_conversation(tokenizer, rec: Record) -> str:
    user_prompt = (
        f"Question:\n{rec.question}\n\n"
        f"Retrieved Documents:\n{rec.context}\n\n"
        "Provide the best possible answer grounded in the documents."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": rec.answer},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_datasets(
    tokenizer, records: List[Record], train_split: float, seed: int
) -> tuple[Dataset, Optional[Dataset]]:
    random.Random(seed).shuffle(records)
    split_idx = int(len(records) * train_split)
    split_idx = max(1, min(split_idx, len(records)))

    train_records = records[:split_idx]
    eval_records = records[split_idx:] if split_idx < len(records) else []

    train_data = [{"text": format_conversation(tokenizer, r)} for r in train_records]
    eval_data = [{"text": format_conversation(tokenizer, r)} for r in eval_records]

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data) if eval_data else None
    return train_ds, eval_ds


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"Loading data from: {data_path}")
    raw_examples = load_raw_examples(data_path)
    records = normalize_records(raw_examples, args.max_docs, args.max_doc_chars)
    if not records:
        raise RuntimeError("No valid training examples after preprocessing.")
    print(f"Valid examples: {len(records)}")

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=quant_config,
    )
    model.config.use_cache = False

    train_ds, eval_ds = build_datasets(tokenizer, records, args.train_split, args.seed)
    print(f"Train examples: {len(train_ds)}")
    print(f"Eval examples: {0 if eval_ds is None else len(eval_ds)}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    evaluation_strategy = "steps" if eval_ds is not None else "no"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=20,
        eval_steps=args.eval_steps if eval_ds is not None else None,
        save_steps=args.eval_steps,
        evaluation_strategy=evaluation_strategy,
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=training_args,
        packing=False,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved finetuned adapter/tokenizer to: {output_dir}")


if __name__ == "__main__":
    main()
