"""
inference/run.py

BioASQ Phase B inference runner.

Loads a BioASQ JSONL file, builds all prompt combinations (num_support × prompt_ids)
into a single batch, runs inference once, and writes one output file per combination.

Output files: {output_dir}/{model_name}_{input_type}_{num_support}_{pid}.json
Output format: {qid: {"text": answer_text, "valid": bool}}

Usage:
    python inference/run.py \
        --data-path  data/batch1.jsonl \
        --output-dir outputs/ \
        --model      /path/to/model \
        --input-type abstracts \
        --num-support 3,5 \
        --prompt-ids 1,2
"""
import json
import re
import sys
import time
import click
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loaders.dataloader import BioASQDataLoader
from loaders.local import VLLMBackend
from loaders.cloud import OpenRouterBackend


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(prompts_path: str) -> dict:
    with open(prompts_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def build_context(question: dict, input_type: str, num_support: int) -> str:
    if input_type == "snippets":
        items = question["snippets"][:num_support]
    else:
        items = [d["text"] for d in question["documents"][:num_support]]

    if not items:
        return "(No context available)"

    return "\n\n".join(f"{input_type}: {x}" for x in items)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json(text: str) -> tuple[bool, str]:
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if matches:
        try:
            parsed = json.loads(matches[-1], strict=False)
            if "answer" in parsed:
                return True, parsed["answer"]
        except json.JSONDecodeError:
            pass
    return False, text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--data-path",   required=True,  help="Path to BioASQ JSONL data.")
@click.option("--output-dir",  required=True,  help="Directory to save output files.")

@click.option("--model",       required=True,  help="Model path (local) or model name (cloud).")
@click.option("--input-type",  default="abstracts", type=click.Choice(["snippets", "abstracts"]), help="What to feed the model.")
@click.option("--num-support", default="5",    help="Comma-separated list of support counts, e.g. '3,5,10'.")
@click.option("--prompt-ids",  default="1",    help="Comma-separated list of prompt IDs to use, e.g. '1,2,3'.")
@click.option("--prompts-file", default=None,  help="Path to prompts JSON (default: inference/prompts_generic.json).")

@click.option("--backend",     default="local", type=click.Choice(["local", "openrouter"]), help="Model backend.")
@click.option("--max-tokens",  default=1000,   type=int)
@click.option("--temperature", default=0.5,    type=float)
@click.option("--gpu-memory-utilization", default=0.90, type=float)
@click.option("--tensor-parallel-size",  default=1,    type=int)
@click.option("--max-model-len",         default=8192, type=int)
def main(data_path, output_dir, model, input_type, num_support, prompt_ids,
         prompts_file, backend, max_tokens, temperature,
         gpu_memory_utilization, tensor_parallel_size, max_model_len):

    if prompts_file is None:
        prompts_file = str(Path(__file__).parent / "prompts_generic.json")

    selected_counts: list[int]  = [int(n) for n in num_support.split(",")]
    selected_prompts = [str(p) for p in prompt_ids.split(",")]

    # Load data
    loader = BioASQDataLoader(path=data_path)
    questions = list(loader)
    print(f"Loaded {len(questions)} questions from {data_path}")

    # Load prompts
    prompts_templates = load_prompts(prompts_file)

    # Build all prompts: questions × num_support × prompt_ids
    prompt_list: list[str] = []
    prompt_info: list[tuple[str, int, str]] = []  # (qid, num_support, pid)

    for q in questions:
        for n in selected_counts:
            context = build_context(q, input_type, n)
            for pid in selected_prompts:
                if pid not in prompts_templates:
                    raise ValueError(f"Prompt id '{pid}' not found in {prompts_file}")
                template = prompts_templates[pid]["template"]
                prompt_list.append(template.format(
                    d_type=input_type,
                    context=context,
                    question=q["body"],
                ))
                prompt_info.append((q["id"], n, pid))

    print(f"Built {len(prompt_list)} prompts ({len(questions)} questions × {len(selected_counts)} counts × {len(selected_prompts)} prompt ids)")

    # Load backend
    if backend == "local":
        b = VLLMBackend(
            model_path=model,
            max_new_tokens=max_tokens,
            temperature=temperature,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )
    else:
        b = OpenRouterBackend(model=model, max_tokens=max_tokens, temperature=temperature)

    b.load()
    print("Model loaded. Running inference...")

    t0 = time.time()
    responses = b.generate_batch(prompt_list)
    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s")

    b.unload()

    # Collect results into nested dict: answer_dict[n][pid][qid]
    answer_dict: dict = {
        n: {pid: {} for pid in selected_prompts}
        for n in selected_counts
    }

    n_valid = 0
    for raw, (qid, n, pid) in zip(responses, prompt_info):
        valid, text = parse_json(raw)
        answer_dict[n][pid][qid] = {"text": text, "valid": valid}
        if valid:
            n_valid += 1

    total = len(prompt_list)
    print(f"Parsed: {n_valid}/{total} valid JSON responses")

    # Save one file per (num_support, pid) combination
    model_name = Path(model).name
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in answer_dict:
        for pid in answer_dict[n]:
            out_path = out_dir / f"{model_name}_{input_type}_{n}_{pid}.json"
            with open(out_path, "w") as f:
                json.dump(answer_dict[n][pid], f, indent=2)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
