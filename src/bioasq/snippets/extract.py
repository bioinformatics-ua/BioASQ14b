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

import json
import re
from collections.abc import Iterator
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

    # Try to parse as JSON directly
    try:
        parsed = json.loads(text)
        return {
            "thinking": parsed.get("thinking", ""),
            "snippets": parsed.get("snippets", []),
        }
    except json.JSONDecodeError:
        pass

    # Fallback: extract outermost JSON objects by brace matching
    results: list[dict] = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            in_string = False
            escape = False
            for j in range(i, len(text)):
                c = text[j]
                if escape:
                    escape = False
                    continue
                if c == "\\":
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[i : j + 1]
                        try:
                            parsed = json.loads(candidate)
                            results.append(parsed)
                        except json.JSONDecodeError:
                            pass
                        i = j
                        break
        i += 1

    # Return last valid JSON object with snippets
    for r in reversed(results):
        if "snippets" in r:
            return {
                "thinking": r.get("thinking", ""),
                "snippets": r.get("snippets", []),
            }
    if results:
        return {
            "thinking": results[-1].get("thinking", ""),
            "snippets": results[-1].get("snippets", []),
        }

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
) -> Iterator[dict]:
    """Run snippet extraction using Unsloth (4-bit) + optional LoRA."""
    import torch
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4-thinking")

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print(f"Loaded & merged LoRA adapter from {adapter_path}")

    FastModel.for_inference(model)

    try:
        for q in tqdm(questions, desc="Extracting snippets"):
            yield _extract_for_question(
                q, model, tokenizer, max_new_tokens, temperature, backend="local"
            )
    finally:
        del model
        torch.cuda.empty_cache()


def _generate_local(
    model: Any,  # noqa: ANN401
    tokenizer: Any,  # noqa: ANN401
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate with Unsloth-loaded model."""
    import torch

    # Render to text first, then tokenize via the processor using the text keyword.
    # Gemma-4's processor treats positional args as images, which breaks text-only calls.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
            use_cache=True,
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
    workers: int = 1,
) -> Iterator[dict]:
    """Run snippet extraction using an OpenRouter/OpenAI-compatible API."""
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        msg = "OPENROUTER_API_KEY not set"
        raise RuntimeError(msg)

    client = OpenAI(base_url=base_url, api_key=api_key)

    def _process(q: dict) -> dict:
        return _extract_for_question(
            q,
            client,
            None,
            max_new_tokens,
            temperature,
            backend="api",
            model_name=model,
            delay=delay,
        )

    if workers <= 1:
        for q in tqdm(questions, desc="Extracting snippets (API)"):
            yield _process(q)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process, q): q for q in questions}
            for future in tqdm(
                as_completed(futures), total=len(questions), desc="Extracting snippets (API)"
            ):
                yield future.result()


def _run_vllm(
    questions: list[dict],
    model: str,
    max_new_tokens: int,
    temperature: float,
    vllm_url: str,
    vllm_api_key: str,
    delay: float,
    adapter_path: str | None = None,
    workers: int = 1,
) -> Iterator[dict]:
    """Run snippet extraction against an externally-started vLLM server.

    Start the server with --tokenizer <hf-model-id> so that vLLM has a
    chat_template for /chat/completions (required since transformers v4.44):

        vllm serve /path/to/model.gguf \\
            --tokenizer unsloth/gemma-4-31B \\
            --host 0.0.0.0 --port 8000
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from openai import OpenAI

    client = OpenAI(base_url=vllm_url, api_key=vllm_api_key)

    def _process(q: dict) -> dict:
        return _extract_for_question(
            q,
            client,
            None,
            max_new_tokens,
            temperature,
            backend="api",
            model_name=model,
            delay=delay,
        )

    if workers <= 1:
        for q in tqdm(questions, desc="Extracting snippets (vLLM)"):
            yield _process(q)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process, q): q for q in questions}
            for future in tqdm(
                as_completed(futures), total=len(questions), desc="Extracting snippets (vLLM)"
            ):
                yield future.result()


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

    for doc in documents[:10]:
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
            import traceback

            print(f"\nError extracting from doc {doc_pmid}: {e}")
            traceback.print_exc()
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
        str | None,
        typer.Option(help="Path to LoRA adapter (local, or vllm with --enable-lora)"),
    ] = None,
    backend: Annotated[str, typer.Option(help="Backend: local, openrouter, or vllm")] = "local",
    model: Annotated[
        str, typer.Option(help="API model name (openrouter/vllm only)")
    ] = "google/gemini-2.5-flash",
    max_new_tokens: Annotated[int, typer.Option(help="Max tokens for output")] = 500,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.1,
    base_url: Annotated[
        str, typer.Option(help="API base URL (openrouter)")
    ] = "https://openrouter.ai/api/v1",
    delay: Annotated[float, typer.Option(help="Delay between API requests")] = 0.1,
    vllm_url: Annotated[
        str, typer.Option(help="vLLM server base URL")
    ] = "http://localhost:8000/v1",
    vllm_api_key: Annotated[str, typer.Option(help="vLLM API key (any string works)")] = "EMPTY",
    workers: Annotated[
        int, typer.Option(help="Parallel workers for API/vLLM backends (local always uses 1)")
    ] = 1,
    testset_path: Annotated[
        Path | None,
        typer.Option(
            "--testset",
            help="BioASQ Phase A testset JSON to hydrate missing body/type fields (for inputs using 'qid' instead of 'id')",
        ),
    ] = None,
) -> None:
    """Extract snippets from documents using a LoRA model or API."""
    # Load questions
    questions: list[dict] = []

    with input_path.open() as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    print(f"Loaded {len(questions)} questions")

    # Normalise inputs that use 'qid' instead of 'id'
    for q in questions:
        if "qid" in q and "id" not in q:
            q["id"] = q.pop("qid")

    # Hydrate body/type from a Phase A testset JSON when the JSONL has only doc lists
    if testset_path is not None:
        with testset_path.open() as f:
            testset = json.load(f)
        meta: dict[str, dict] = {tq["id"]: tq for tq in testset.get("questions", [])}
        for q in questions:
            if not q.get("body") and q.get("id") in meta:
                q["body"] = meta[q["id"]].get("body", "")
                q.setdefault("type", meta[q["id"]].get("type", "summary"))
        print(f"Hydrated question bodies from {testset_path}")

    # Resume from checkpoint if output already exists
    processed_ids: set[str] = set()
    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                if line.strip():
                    try:
                        processed_ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        if processed_ids:
            print(f"Resuming: skipping {len(processed_ids)} already-processed questions")
            questions = [q for q in questions if q.get("id") not in processed_ids]

    # Run extraction
    if backend == "local":
        result_iter = _run_local(questions, base_model, adapter_path, max_new_tokens, temperature)
    elif backend == "vllm":
        result_iter = _run_vllm(
            questions,
            base_model,
            max_new_tokens,
            temperature,
            vllm_url,
            vllm_api_key,
            delay,
            adapter_path,
            workers,
        )
    else:
        result_iter = _run_api(
            questions, model, max_new_tokens, temperature, base_url, delay, workers
        )

    # Save output incrementally (checkpoint after each question)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_snippets = 0
    n_written = 0
    mode = "a" if processed_ids else "w"
    with output_path.open(mode) as f:
        for r in result_iter:
            total_snippets += len(r.get("snippets", []))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n_written += 1

    print(f"Extracted {total_snippets} snippets for {n_written} questions")
    print(f"Output: {output_path}")


# ---------------------------------------------------------------------------
# JSONL → BioASQ submission format
# ---------------------------------------------------------------------------


@app.command(name="to-bioasq")
def to_bioasq(
    input_path: Annotated[Path, typer.Option("--input", help="Extracted snippets JSONL")] = Path(
        "data/snippets/extracted_snippets.jsonl"
    ),
    output_path: Annotated[Path, typer.Option("--output", help="BioASQ submission JSON")] = Path(
        "data/snippets/submission.json"
    ),
) -> None:
    """Convert extracted snippets JSONL to BioASQ submission JSON."""
    questions: list[dict] = []
    with input_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)

            # Convert documents: {id, text} dicts → PubMed URL strings
            docs_raw = q.get("documents", [])

            doc_urls: list[str] = []
            for d in docs_raw[:10]:
                if isinstance(d, dict):
                    pmid = d.get("id", "")
                    doc_urls.append(f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}")
                elif isinstance(d, str):
                    doc_urls.append(d)

            # Clean snippets: remove internal 'thinking' field
            snippets = []
            for s in q.get("snippets", [])[:10]:
                snippets.append(
                    {
                        "text": s["text"],
                        "document": s["document"],
                        "offsetInBeginSection": s["offsetInBeginSection"],
                        "offsetInEndSection": s["offsetInEndSection"],
                        "beginSection": s.get("beginSection", "abstract"),
                        "endSection": s.get("endSection", "abstract"),
                    }
                )

            questions.append(
                {
                    "id": q["id"],
                    "body": q.get("body", ""),
                    "type": q.get("type", "summary"),
                    "documents": doc_urls,
                    "snippets": snippets,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"questions": questions}, f, ensure_ascii=False, indent=2)

    total_snippets = sum(len(q["snippets"]) for q in questions)
    print(f"Wrote {len(questions)} questions, {total_snippets} snippets → {output_path}")


if __name__ == "__main__":
    app()
