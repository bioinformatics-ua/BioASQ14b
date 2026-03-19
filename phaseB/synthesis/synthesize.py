"""
synthesis/synthesize.py

Synthesis step for BioASQ Phase B.

Takes multiple run.py output files ({qid: {"text": ..., "valid": ...}})
and synthesizes a final ideal_answer via LLM, grid-searching over prompt IDs.

Output filename: {out_id}_{model_name}_{num_runs}_{pid}.json
Output format:   {qid: {"text": ideal_answer, "valid": bool}}

Usage:
    python synthesis/synthesize.py \
        outputs/run_A.json outputs/run_B.json \
        --data-path  data/batch1.jsonl \
        --output-dir outputs/synthesis/ \
        --model      /path/to/model \
        --prompt-ids 1,2,3 \
        --out-id     exp1
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
# Build answers block
# ---------------------------------------------------------------------------

def build_answers_block(runs: list[dict], qid: str) -> str:
    lines = []
    for i, run in enumerate(runs, 1):
        entry = run.get(qid, {})
        text = entry.get("text") or ""
        if text:
            lines.append(f"Answer {i}: {text}")
    return "\n\n".join(lines) if lines else "(No answers available)"


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
@click.argument("runs", nargs=-1, required=True)
@click.option("--data-path",    required=True,  help="Path to BioASQ JSONL (for question text).")
@click.option("--output-dir",   required=True,  help="Directory to save outputs.")
@click.option("--out-id",       required=True,  help="Identifier prefix for output filenames.")
@click.option("--model",        required=True,  help="Model path (local) or model name (cloud).")
@click.option("--prompt-ids",   default="1",    help="Comma-separated prompt IDs, e.g. '1,2,3'.")
@click.option("--prompts-file", default=None,   help="Path to prompts JSON (default: synthesis/prompts.json).")
@click.option("--backend",      default="local", type=click.Choice(["local", "openrouter"]))
@click.option("--max-tokens",   default=1000,   type=int)
@click.option("--temperature",  default=0.5,    type=float)
@click.option("--gpu-memory-utilization", default=0.90, type=float)
@click.option("--tensor-parallel-size",  default=1,    type=int)
@click.option("--max-model-len",         default=8192, type=int)
def main(runs, data_path, output_dir, out_id, model, prompt_ids, prompts_file,
         backend, max_tokens, temperature, gpu_memory_utilization,
         tensor_parallel_size, max_model_len):

    if prompts_file is None:
        prompts_file = str(Path(__file__).parent / "prompts.json")

    selected_prompts = [str(p) for p in prompt_ids.split(",")]

    # Load run files
    run_data: list[dict] = []
    for path in runs:
        with open(path) as f:
            run_data.append(json.load(f))
    print(f"Loaded {len(run_data)} run file(s), {len(run_data[0])} questions each")

    # Load question text from data
    loader = BioASQDataLoader(path=data_path)
    question_text = {q["id"]: q["body"] for q in loader}

    # Load prompts
    prompts_templates = load_prompts(prompts_file)

    all_qids = list(run_data[0].keys())

    # Build all prompts: questions × prompt_ids
    prompt_list: list[str] = []
    prompt_info: list[tuple[str, str]] = []  # (qid, pid)

    for pid in selected_prompts:
        if pid not in prompts_templates:
            raise ValueError(f"Prompt id '{pid}' not found in {prompts_file}")
        template = prompts_templates[pid]["template"]
        for qid in all_qids:
            answers = build_answers_block(run_data, qid)
            question = question_text.get(qid, "")
            prompt_list.append(template.format(answers=answers, question=question))
            prompt_info.append((qid, pid))

    print(f"Built {len(prompt_list)} prompts ({len(all_qids)} questions × {len(selected_prompts)} prompt ids)")

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
    print(f"Inference done in {time.time() - t0:.1f}s")

    b.unload()

    # Collect results: answer_dict[pid][qid]
    answer_dict: dict = {pid: {} for pid in selected_prompts}

    n_valid = 0
    for raw, (qid, pid) in zip(responses, prompt_info):
        valid, text = parse_json(raw)
        answer_dict[pid][qid] = {"text": text, "valid": valid}
        if valid:
            n_valid += 1

    print(f"Parsed: {n_valid}/{len(prompt_list)} valid JSON responses")

    # Save one file per pid
    model_name = Path(model).name
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pid in answer_dict:
        out_path = out_dir / f"{out_id}_{model_name}_{len(runs)}_{pid}.json"
        with open(out_path, "w") as f:
            json.dump(answer_dict[pid], f, indent=2)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
