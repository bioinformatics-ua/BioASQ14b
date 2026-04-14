"""Snippet extraction inference — run the LoRA model on new questions.

Given questions with retrieved documents (Phase A output), extracts
snippets and optional thinking/rationale using the fine-tuned LoRA adapter.

Supports two backends:
  - local: loads LoRA on top of base model with HF / vLLM
  - openrouter: uses an API model (for zero-shot baseline or distillation)

Output: enriched JSONL with snippets + thinking added to each question.

Usage:
    python -m bioasq.snippets.extract \
        --input  phaseA-reranker/runs_bioasq_format_hydrated/submission1_hydrated.jsonl \
        --output data/snippets/extracted_snippets.jsonl \
        --base-model google/gemma-3-27b-it \
        --adapter-path data/training/snippet_extraction/lora_output/final_adapter \
        --backend local
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any

import typer
from tqdm import tqdm

app = typer.Typer()


SYSTEM_PROMPT = """\
You are a biomedical snippet extractor. Given a question and a PubMed \
abstract, extract the most relevant text spans that help answer the question. \
Output valid JSON with two fields:
- "thinking": 1-2 sentences explaining why the snippets are relevant.
- "snippets": a list of verbatim text spans copied exactly from the abstract.\
"""

USER_TEMPLATE = """\
Question: {question}

Abstract:
{doc_text}"""


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def parse_extraction_output(text: str) -> dict[str, Any]:
    """Parse the model's JSON output into thinking + snippets."""
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())

    # Try to parse as JSON
    try:
        parsed = json.loads(text)
        return {
            "thinking": parsed.get("thinking", ""),
            "snippets": parsed.get("snippets", []),
        }
    except json.JSONDecodeError:
        pass

    # Fallback: find last JSON object in text
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in reversed(matches):
        try:
            parsed = json.loads(match)
            return {
                "thinking": parsed.get("thinking", ""),
                "snippets": parsed.get("snippets", []),
            }
        except json.JSONDecodeError:
            continue

    return {"thinking": "", "snippets": []}


# ---------------------------------------------------------------------------
# Offset recovery
# ---------------------------------------------------------------------------


def recover_offsets(
    snippet_text: str,
    doc_text: str,
) -> tuple[int, int] | None:
    """Find the character offsets of snippet_text within doc_text.

    Returns (start, end) or None if not found.
    Uses exact match first, then whitespace-normalized fuzzy match.
    """
    # Exact match
    idx = doc_text.find(snippet_text)
    if idx >= 0:
        return idx, idx + len(snippet_text)

    # Whitespace-normalised match
    norm_doc = re.sub(r"\s+", " ", doc_text).strip()
    norm_snip = re.sub(r"\s+", " ", snippet_text).strip()
    idx = norm_doc.find(norm_snip)
    if idx >= 0:
        return idx, idx + len(norm_snip)

    # Anchor match (first 80 chars)
    anchor = norm_snip[:80]
    idx = norm_doc.find(anchor)
    if idx >= 0:
        end = idx + len(norm_snip)
        return idx, min(end, len(norm_doc))

    return None


def _build_snippet_object(
    snippet_text: str,
    doc_text: str,
    doc_pmid: str,
) -> dict[str, Any] | None:
    """Build a BioASQ-formatted snippet dict with recovered offsets."""
    offsets = recover_offsets(snippet_text, doc_text)
    if offsets is None:
        return None

    start, end = offsets
    return {
        "text": snippet_text,
        "document": f"http://www.ncbi.nlm.nih.gov/pubmed/{doc_pmid}",
        "offsetInBeginSection": start,
        "offsetInEndSection": end,
        "beginSection": "abstract",
        "endSection": "abstract",
    }


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------


def _run_local(
    questions: list[dict],
    base_model: str,
    adapter_path: str | None,
    max_new_tokens: int,
    temperature: float,
) -> list[dict]:
    """Run snippet extraction using a local model (HF + optional LoRA)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print(f"Loaded LoRA adapter from {adapter_path}")

    model.eval()

    results = []
    for q in tqdm(questions, desc="Extracting snippets"):
        q_result = _extract_for_question(
            q, model, tokenizer, max_new_tokens, temperature, backend="local"
        )
        results.append(q_result)

    del model
    torch.cuda.empty_cache()
    return results


def _generate_local(
    model: Any,  # noqa: ANN401
    tokenizer: Any,  # noqa: ANN401
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate with a local HF model."""
    import torch

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
        )

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _run_api(
    questions: list[dict],
    model: str,
    max_new_tokens: int,
    temperature: float,
    base_url: str,
    delay: float,
) -> list[dict]:
    """Run snippet extraction using an OpenRouter/OpenAI-compatible API."""
    import os

    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        msg = "OPENROUTER_API_KEY not set"
        raise RuntimeError(msg)

    client = OpenAI(base_url=base_url, api_key=api_key)

    results = []
    for q in tqdm(questions, desc="Extracting snippets (API)"):
        q_result = _extract_for_question(
            q,
            client,
            None,
            max_new_tokens,
            temperature,
            backend="api",
            model_name=model,
            delay=delay,
        )
        results.append(q_result)

    return results


def _extract_for_question(
    question: dict,
    model_or_client: Any,  # noqa: ANN401
    tokenizer: Any,  # noqa: ANN401
    max_new_tokens: int,
    temperature: float,
    backend: str = "local",
    model_name: str = "",
    delay: float = 0.0,
) -> dict:
    """Extract snippets for a single question across all its documents."""
    import time

    q_body = question.get("body", "")
    documents = question.get("documents", [])
    all_snippets: list[dict] = []
    all_thinking: list[str] = []

    for doc in documents:
        if isinstance(doc, dict):
            doc_text = doc.get("text", "")
            doc_pmid = doc.get("id", "")
        else:
            continue

        if not doc_text:
            continue

        # Truncate very long documents
        doc_text_truncated = doc_text[:3500]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(question=q_body, doc_text=doc_text_truncated),
            },
        ]

        try:
            if backend == "local":
                raw = _generate_local(
                    model_or_client,
                    tokenizer,
                    messages,
                    max_new_tokens,
                    temperature,
                )
            else:
                if delay > 0:
                    time.sleep(delay)
                response = model_or_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
                raw = response.choices[0].message.content or ""

            parsed = parse_extraction_output(raw)
            thinking = parsed.get("thinking", "")
            snippet_texts = parsed.get("snippets", [])

            if thinking:
                all_thinking.append(thinking)

            for snip_text in snippet_texts:
                if not isinstance(snip_text, str) or not snip_text.strip():
                    continue
                snippet_obj = _build_snippet_object(snip_text, doc_text, doc_pmid)
                if snippet_obj is not None:
                    snippet_obj["thinking"] = thinking
                    all_snippets.append(snippet_obj)

        except Exception as e:
            print(f"\nError extracting from doc {doc_pmid}: {e}")

    return {
        "id": question.get("id", ""),
        "body": q_body,
        "type": question.get("type", "summary"),
        "documents": documents,
        "snippets": all_snippets,
        "thinking": all_thinking,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_path: Annotated[
        Path, typer.Option("--input", help="Input JSONL with questions + documents")
    ] = Path("data/val_data/13B1_golden_documents.jsonl"),
    output_path: Annotated[
        Path, typer.Option("--output", help="Output JSONL with extracted snippets")
    ] = Path("data/snippets/extracted_snippets.jsonl"),
    base_model: Annotated[str, typer.Option(help="Base model name")] = "google/gemma-3-27b-it",
    adapter_path: Annotated[
        str | None, typer.Option(help="Path to LoRA adapter (local only)")
    ] = None,
    backend: Annotated[str, typer.Option(help="Backend: local or openrouter")] = "local",
    model: Annotated[
        str, typer.Option(help="API model name (openrouter only)")
    ] = "google/gemini-2.5-flash",
    max_new_tokens: Annotated[int, typer.Option(help="Max tokens for output")] = 500,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.1,
    base_url: Annotated[str, typer.Option(help="API base URL")] = "https://openrouter.ai/api/v1",
    delay: Annotated[float, typer.Option(help="Delay between API requests")] = 0.1,
) -> None:
    """Extract snippets from documents using a LoRA model or API."""
    # Load questions
    questions: list[dict] = []
    with input_path.open() as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    print(f"Loaded {len(questions)} questions")

    # Run extraction
    if backend == "local":
        results = _run_local(questions, base_model, adapter_path, max_new_tokens, temperature)
    else:
        results = _run_api(questions, model, max_new_tokens, temperature, base_url, delay)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_snippets = 0
    with output_path.open("w") as f:
        for r in results:
            total_snippets += len(r.get("snippets", []))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Extracted {total_snippets} snippets for {len(results)} questions")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    app()
