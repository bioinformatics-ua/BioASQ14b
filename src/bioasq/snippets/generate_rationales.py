"""Generate rationale annotations for gold snippet training data.

Calls a large model (locally via vLLM, or via OpenRouter API) to explain
*why* each set of snippets answers the question.  The resulting
``thinking`` field is merged into the training data so the LoRA model
learns to produce both snippets **and** rationales.

Input:  gold_pairs.jsonl  (output of prepare_training_data)
Output: gold_pairs_with_rationale.jsonl  (same + ``thinking`` field)

Usage (local – 2×A40):
    python -m bioasq.snippets.generate_rationales \
        --input  data/training/snippet_extraction/gold_pairs.jsonl \
        --output data/training/snippet_extraction/gold_pairs_with_rationale.jsonl \
        --backend local \
        --model  Qwen/Qwen2.5-72B-Instruct-AWQ \
        --tensor-parallel-size 2

Usage (API):
    python -m bioasq.snippets.generate_rationales \
        --input  data/training/snippet_extraction/gold_pairs.jsonl \
        --output data/training/snippet_extraction/gold_pairs_with_rationale.jsonl \
        --backend openrouter \
        --model  google/gemini-2.5-flash
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

app = typer.Typer()


RATIONALE_PROMPT = """\
You are a biomedical expert. Given a question, a PubMed abstract, and \
the relevant snippets extracted from that abstract, explain in 1-2 concise \
sentences why these snippets are relevant to answering the question.

Question: {question}

Abstract:
{doc_text}

Relevant snippets:
{snippets}

Rationale:"""


def _build_rationale_prompt(example: dict) -> str:
    snippets_block = "\n".join(f"- {s}" for s in example["snippets"])
    return RATIONALE_PROMPT.format(
        question=example["question_body"],
        doc_text=example["doc_text"][:2000],  # truncate very long abstracts
        snippets=snippets_block,
    )


# ---------------------------------------------------------------------------
# OpenRouter / OpenAI-compatible API backend
# ---------------------------------------------------------------------------


def _call_api(
    client: object,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    delay: float,
) -> str:
    """Single OpenAI-compatible API call."""
    if delay > 0:
        time.sleep(delay)
    response = client.chat.completions.create(  # type: ignore[union-attr]
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""  # type: ignore[union-attr]


def _run_openrouter(
    examples: list[dict],
    done_keys: set[str],
    output_path: Path,
    model: str,
    max_tokens: int,
    temperature: float,
    delay: float,
    base_url: str,
) -> None:
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        typer.echo("Error: OPENROUTER_API_KEY not set.", err=True)
        raise typer.Exit(1)

    client = OpenAI(base_url=base_url, api_key=api_key)
    errors = 0

    with output_path.open("a") as out_f:
        for ex in tqdm(examples, desc="Generating rationales"):
            key = f"{ex['question_id']}_{ex['doc_pmid']}"
            if key in done_keys:
                continue

            prompt = _build_rationale_prompt(ex)
            try:
                rationale = _call_api(client, model, prompt, max_tokens, temperature, delay)
                ex["thinking"] = rationale.strip()
            except Exception as e:
                errors += 1
                print(f"\nAPI error for {key}: {e}")
                ex["thinking"] = ""

            out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"\nDone. Errors: {errors}. Output: {output_path}")


# ---------------------------------------------------------------------------
# Local vLLM backend  (batched, then explicit unload)
# ---------------------------------------------------------------------------


def _run_local(
    examples: list[dict],
    done_keys: set[str],
    output_path: Path,
    model: str,
    max_tokens: int,
    temperature: float,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    batch_size: int,
    tokenizer: str | None = None,
    hf_config_path: str | None = None,
    language_model_only: bool = False,
) -> None:
    """Generate rationales using a local vLLM model in batches, then unload."""
    import torch
    from vllm import LLM, SamplingParams

    # Filter to pending examples
    pending = [ex for ex in examples if f"{ex['question_id']}_{ex['doc_pmid']}" not in done_keys]
    if not pending:
        print("All examples already processed.")
        return

    print(f"Pending: {len(pending)} examples")
    print(f"Loading model {model} on {tensor_parallel_size} GPU(s)...")

    extra_kwargs: dict = {}
    if tokenizer:
        extra_kwargs["tokenizer"] = tokenizer
    if hf_config_path:
        extra_kwargs["hf_config_path"] = hf_config_path
    if language_model_only:
        extra_kwargs["language_model_only"] = True

    extra_kwargs["reasoning_parser"] = "qwen3"
    extra_kwargs["dtype"] = "bfloat16"

    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        **extra_kwargs,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    errors = 0
    with output_path.open("a") as out_f:
        for i in tqdm(range(0, len(pending), batch_size), desc="Generating rationales"):
            batch = pending[i : i + batch_size]
            conversations = [
                [{"role": "user", "content": _build_rationale_prompt(ex)}] for ex in batch
            ]
            try:
                outputs = llm.chat(
                    messages=conversations,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                for ex, out in zip(batch, outputs):
                    ex["thinking"] = out.outputs[0].text.strip()
                    out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            except Exception as e:
                errors += 1
                print(f"\nvLLM error at batch {i}: {e}")
                for ex in batch:
                    ex["thinking"] = ""
                    out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            out_f.flush()

    # ── Unload model and free GPU memory ──────────────────────────────
    print("Unloading model...")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Model unloaded. Done. Errors: {errors}. Output: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_path: Annotated[Path, typer.Option("--input", help="gold_pairs.jsonl")] = Path(
        "data/training/snippet_extraction/gold_pairs.jsonl"
    ),
    output_path: Annotated[
        Path, typer.Option("--output", help="Output JSONL with rationales")
    ] = Path("data/training/snippet_extraction/gold_pairs_with_rationale.jsonl"),
    backend: Annotated[str, typer.Option(help="Backend: 'local' (vLLM) or 'openrouter'")] = "local",
    model: Annotated[str, typer.Option(help="Model name / HF path")] = "Qwen/Qwen3.5-27B",
    max_tokens: Annotated[int, typer.Option(help="Max tokens for rationale")] = 150,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 1.0,
    # OpenRouter-only
    delay: Annotated[float, typer.Option(help="Seconds between API requests (openrouter)")] = 0.1,
    base_url: Annotated[
        str, typer.Option(help="API base URL (openrouter)")
    ] = "https://openrouter.ai/api/v1",
    # Local-only
    tensor_parallel_size: Annotated[int, typer.Option(help="Number of GPUs (local)")] = 2,
    gpu_memory_utilization: Annotated[
        float, typer.Option(help="vLLM GPU memory fraction (local)")
    ] = 0.90,
    max_model_len: Annotated[int, typer.Option(help="Max context length in tokens (local)")] = 4096,
    batch_size: Annotated[int, typer.Option(help="Batch size for local inference")] = 32,
    tokenizer: Annotated[
        str, typer.Option(help="Override tokenizer path (local, e.g. for GGUF models)")
    ] = "",
    hf_config_path: Annotated[
        str, typer.Option(help="Override HF config path (local, e.g. for GGUF models)")
    ] = "",
    language_model_only: Annotated[
        bool,
        typer.Option(help="Load only the language model head (local, for multimodal checkpoints)"),
    ] = False,
    # Resume
    resume: Annotated[bool, typer.Option(help="Resume from existing output")] = True,
) -> None:
    """Add rationale annotations to gold snippet training pairs."""
    # Load input
    examples: list[dict] = []
    with input_path.open() as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    print(f"Loaded {len(examples)} training pairs")

    # Resume support — skip already-processed pairs
    done_keys: set[str] = set()
    if resume and output_path.exists():
        with output_path.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    done_keys.add(f"{obj['question_id']}_{obj['doc_pmid']}")
        print(f"Resuming: {len(done_keys)} already processed")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if backend == "local":
        _run_local(
            examples=examples,
            done_keys=done_keys,
            output_path=output_path,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            batch_size=batch_size,
            tokenizer=tokenizer or None,
            hf_config_path=hf_config_path or None,
            language_model_only=language_model_only,
        )
    elif backend == "openrouter":
        _run_openrouter(
            examples=examples,
            done_keys=done_keys,
            output_path=output_path,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            delay=delay,
            base_url=base_url,
        )
    else:
        typer.echo(f"Unknown backend: {backend!r}. Use 'local' or 'openrouter'.", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
