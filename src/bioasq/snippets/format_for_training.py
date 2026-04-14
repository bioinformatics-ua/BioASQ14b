"""Convert gold training pairs into chat-formatted JSONL for LoRA fine-tuning.

Reads the output of prepare_training_data (optionally with rationales from
generate_rationales) and converts each example into a chat-template JSONL
record compatible with TRL's SFTTrainer.

Output format (JSONL, one per line):
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }

Usage:
    python -m bioasq.snippets.format_for_training \
        --input data/training/snippet_extraction/gold_pairs_with_rationale.jsonl \
        --output-dir data/training/snippet_extraction/ \
        --val-fraction 0.1
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


SYSTEM_PROMPT = (
    "You are a biomedical snippet extractor. Given a question and a PubMed "
    "abstract, extract the most relevant text spans that help answer the question. "
    "Output valid JSON with two fields:\n"
    '- "thinking": 1-2 sentences explaining why the snippets are relevant.\n'
    '- "snippets": a list of verbatim text spans copied exactly from the abstract.'
)

USER_TEMPLATE = """\
Question: {question}

Abstract:
{doc_text}"""

ASSISTANT_TEMPLATE = """{{"thinking": {thinking}, "snippets": {snippets}}}"""


def _format_example(example: dict) -> dict:
    """Convert a single training pair to chat messages."""
    user_content = USER_TEMPLATE.format(
        question=example["question_body"],
        doc_text=example["doc_text"],
    )

    thinking = example.get("thinking", "")
    if not thinking:
        thinking = "These snippets contain information relevant to the question."

    assistant_content = ASSISTANT_TEMPLATE.format(
        thinking=json.dumps(thinking, ensure_ascii=False),
        snippets=json.dumps(example["snippets"], ensure_ascii=False),
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


@app.command()
def main(
    input_path: Annotated[
        Path, typer.Option("--input", help="Gold pairs JSONL (with or without rationale)")
    ] = Path("data/training/snippet_extraction/gold_pairs_with_rationale.jsonl"),
    output_dir: Annotated[Path, typer.Option(help="Output directory for train/val splits")] = Path(
        "data/training/snippet_extraction/"
    ),
    val_fraction: Annotated[float, typer.Option(help="Fraction of data for validation")] = 0.1,
    seed: Annotated[int, typer.Option(help="Random seed for splitting")] = 42,
    max_doc_chars: Annotated[
        int, typer.Option(help="Approx max doc length in chars (3500 chars ~ 1000 tokens)")
    ] = 3500,
) -> None:
    """Convert gold pairs to chat-formatted JSONL for SFT training."""
    random.seed(seed)

    # Load input
    examples: list[dict] = []
    skipped = 0
    with input_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            # Truncate very long documents rather than skip
            if len(ex.get("doc_text", "")) > max_doc_chars:
                ex["doc_text"] = ex["doc_text"][:max_doc_chars]
            # Skip if no snippets
            if not ex.get("snippets"):
                skipped += 1
                continue
            examples.append(ex)

    print(f"Loaded {len(examples)} examples ({skipped} skipped for no snippets)")

    # Shuffle and split
    random.shuffle(examples)
    val_size = max(1, int(len(examples) * val_fraction))
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")

    # Format and write
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train_examples), ("val", val_examples)]:
        out_path = output_dir / f"chat_{split_name}.jsonl"
        with out_path.open("w") as f:
            for ex in split_data:
                formatted = _format_example(ex)
                f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
        print(f"Wrote {len(split_data)} examples to {out_path}")


if __name__ == "__main__":
    app()
