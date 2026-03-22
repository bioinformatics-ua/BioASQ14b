"""
Exact-answer inference for BioASQ Phase B.

For yesno / factoid / list: extracts exact_answer from model JSON.

For summary: no exact answer — model outputs reasoning + ideal_answer (same narrative
role as ideal_answer elsewhere). Stored as ideal_answer plus exact_answer \"\".

Usage:
    python inference/run_exact.py \\
        --input  data/training14b/training14b.json \\
        --output dev/outputs/exact_p1.json \\
        --model  /path/to/model
"""
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loaders.dataloader import BioASQDataLoader
from loaders.local import VLLMBackend
from loaders.cloud import OpenRouterBackend


def load_prompts(path):
    with open(path) as f:
        return json.load(f)


def build_context(question, num_context, source):
    if source == "abstracts":
        items = [d["text"] for d in question["documents"] if d.get("text")][:num_context]
        label = "Abstract"
    else:
        items = question["snippets"][:num_context]
        label = "Snippet"
    if not items:
        return "(No context available)"
    return "\n\n".join(f"{label} {i}: {s}" for i, s in enumerate(items, 1))


def parse_exact(text, qtype):
    """
    Find the last JSON block in the model output and extract exact_answer.
    Promotes bare strings / flat lists to the expected shape per type.
    Returns (valid, exact_answer).
    """
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if not matches:
        return False, None
    try:
        exact = json.loads(matches[-1], strict=False).get("exact_answer")
        if exact is None:
            return False, None
        if qtype == "yesno" and isinstance(exact, str) and exact.lower() in ("yes", "no"):
            return True, exact.lower()
        if qtype == "factoid":
            return True, [exact] if isinstance(exact, str) else exact
        if qtype == "list":
            if isinstance(exact, list) and exact and not isinstance(exact[0], list):
                return True, [[item] for item in exact]   # promote flat → nested
            return True, exact
    except json.JSONDecodeError:
        pass
    return False, None


def parse_summary_ideal(text):
    """Extract ideal_answer from JSON (summary questions). Returns (valid, ideal_paragraph)."""
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if not matches:
        return False, None
    try:
        obj = json.loads(matches[-1], strict=False)
        ideal = obj.get("ideal_answer") if isinstance(obj, dict) else None
        if isinstance(ideal, str) and ideal.strip():
            return True, ideal.strip()
    except json.JSONDecodeError:
        pass
    return False, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",        required=True)
    p.add_argument("--output",       default=None,  help="Output file path (single prompt mode)")
    p.add_argument("--output-dir",   default=None,  help="Output directory (multi-prompt grid mode)")
    p.add_argument("--model",        default=None)
    p.add_argument("--backend",      default="local", choices=["local", "openrouter"])
    p.add_argument("--prompt-id",    default=None,  help="Single prompt ID (backward compat)")
    p.add_argument("--prompt-ids",   nargs="+", default=None,
                   help="One or more prompt IDs to run with one model load, e.g. 1 2 3 4. Use 'all' to run every prompt in the prompts file.")
    p.add_argument("--num-snippets", type=int, default=5)
    p.add_argument("--prompts-file", default=str(Path(__file__).parent / "prompts_exact.json"))
    p.add_argument("--max-tokens",   type=int,   default=500)
    p.add_argument("--temperature",  type=float, default=0.0)
    p.add_argument("--types",          nargs="+",  default=["yesno", "factoid", "list"],
                   choices=["yesno", "factoid", "list", "summary"],
                   help="Question types to run. summary → ideal_answer paragraph JSON.")
    p.add_argument("--context-source",         default="abstracts", choices=["abstracts", "snippets"],
                   help="abstracts = Phase A+ golden docs; snippets = Phase B passages.")
    p.add_argument("--tensor-parallel-size",   type=int,   default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len",          type=int,   default=8192)
    p.add_argument("--enforce-eager",          action="store_true", default=False)
    p.add_argument("--request-delay",          type=float, default=0.0,
                   help="Seconds between requests — use ~4.0 for free OpenRouter models (16 req/min limit)")
    args = p.parse_args()

    loader  = BioASQDataLoader(path=args.input)
    prompts = load_prompts(args.prompts_file)

    # Resolve which prompt IDs to run
    if args.prompt_ids:
        # Grid mode: --prompt-ids 1 2 3  or  --prompt-ids all
        if args.prompt_ids == ["all"]:
            # Union of all IDs across all types present in the prompts file
            all_ids = set()
            for qtype_prompts in prompts.values():
                all_ids.update(qtype_prompts.keys())
            prompt_ids = sorted(all_ids, key=lambda x: int(x) if x.isdigit() else x)
        else:
            prompt_ids = args.prompt_ids
        if not args.output_dir:
            p.error("--output-dir is required when using --prompt-ids")
        grid_mode = True
    else:
        # Single-prompt backward-compat mode
        prompt_ids = [args.prompt_id or "1"]
        if not args.output:
            p.error("--output is required when not using --prompt-ids")
        grid_mode = False

    questions = [q for q in loader if q["type"] in set(args.types)]
    model_slug = (args.model or "unknown").replace("/", "-").replace(".", "-").replace(":", "-")

    # In grid mode: compute output path per (pid, qtype) and skip existing files
    def grid_output_path(pid, qtype):
        return Path(args.output_dir) / f"{model_slug}_p{pid}_{args.context_source}_{qtype}.json"

    # Determine which (pid, qtype) combos still need work
    combos_to_run = []
    for pid in prompt_ids:
        qtypes_needed = {q["type"] for q in questions if pid in prompts.get(q["type"], {})}
        for qtype in sorted(qtypes_needed):
            if grid_mode and grid_output_path(pid, qtype).exists():
                print(f"Skipping prompt {pid} / {qtype} — output already exists")
                continue
            combos_to_run.append((pid, qtype))

    if not combos_to_run:
        print("All outputs already exist — nothing to run.")
        return

    # Non-grid mode: single output file — skip if it already exists
    if not grid_mode and Path(args.output).exists():
        print(f"Output already exists: {args.output} — skipping.")
        return

    print(f"{len(combos_to_run)} combo(s) to run. Loading model...")

    backend = VLLMBackend(args.model, max_new_tokens=args.max_tokens, temperature=args.temperature,
                          tensor_parallel_size=args.tensor_parallel_size,
                          gpu_memory_utilization=args.gpu_memory_utilization,
                          max_model_len=args.max_model_len,
                          enforce_eager=args.enforce_eager) \
              if args.backend == "local" else \
              OpenRouterBackend(args.model, max_tokens=args.max_tokens, temperature=args.temperature, request_delay=args.request_delay)

    backend.load()

    # Process each (pid, qtype) combo separately — save immediately after each
    saved_files = []
    for pid, qtype in combos_to_run:
        combo_prompts, combo_qids = [], []
        for q in questions:
            if q["type"] != qtype:
                continue
            if pid not in prompts.get(qtype, {}):
                continue
            template = prompts[qtype][pid]["template"]
            combo_prompts.append(template.format(
                question=q["body"],
                context=build_context(q, args.num_snippets, args.context_source)
            ))
            combo_qids.append(q["id"])

        if not combo_prompts:
            continue

        print(f"\nPrompt {pid} / {qtype} — {len(combo_prompts)} questions...")
        responses = backend.generate_batch(combo_prompts)

        results = {}
        for raw, qid in zip(responses, combo_qids):
            if qtype == "summary":
                valid, ideal = parse_summary_ideal(raw)
                results[qid] = {
                    "exact_answer": "",
                    "ideal_answer": ideal if valid else "",
                    "valid": valid,
                    "raw": raw,
                }
            else:
                valid, exact = parse_exact(raw, qtype)
                results[qid] = {"exact_answer": exact, "valid": valid, "raw": raw}

        n_valid = sum(1 for r in results.values() if r["valid"])
        print(f"Prompt {pid} / {qtype} — {n_valid}/{len(results)} valid")

        out = grid_output_path(pid, qtype) if grid_mode else Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {out}")
        saved_files.append(str(out))

    backend.unload()
    print(f"Inference done. {len(saved_files)} file(s) saved.")


if __name__ == "__main__":
    main()
