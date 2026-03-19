"""
Exact-answer inference for BioASQ Phase B.

Identical flow to run.py, with two differences:
  1. Summary questions are skipped (they have no exact answer)
  2. Only exact_answer is extracted from the model output

NOTE: build_context() currently uses snippets (training data).
When the dataloader is updated for Phase A+, swap snippets → abstracts here.

Usage:
    python inference/run_exact.py \\
        --input  data/training14b/training14b.json \\
        --output dev/outputs/exact_p1.json \\
        --model  /path/to/model
"""
import argparse, json, re, sys
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",        required=True)
    p.add_argument("--output",       required=True)
    p.add_argument("--model",        default=None)
    p.add_argument("--backend",      default="local", choices=["local", "openrouter"])
    p.add_argument("--prompt-id",    default="1")
    p.add_argument("--num-snippets", type=int, default=5)
    p.add_argument("--prompts-file", default=str(Path(__file__).parent / "prompts_exact.json"))
    p.add_argument("--max-tokens",   type=int,   default=500)
    p.add_argument("--temperature",  type=float, default=0.0)
    p.add_argument("--types",          nargs="+",  default=["yesno", "factoid", "list"],
                   choices=["yesno", "factoid", "list"],
                   help="Question types to run. Default: all three.")
    p.add_argument("--context-source",         default="abstracts", choices=["abstracts", "snippets"],
                   help="abstracts = Phase A+ golden docs; snippets = Phase B passages.")
    p.add_argument("--tensor-parallel-size",   type=int,   default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len",          type=int,   default=8192)
    args = p.parse_args()

    loader  = BioASQDataLoader(path=args.input)
    prompts = load_prompts(args.prompts_file)

    # Build one prompt per non-summary question
    prompt_list, meta_list = [], []
    run_types = set(args.types)
    for q in loader:
        if q["type"] not in run_types:
            continue
        template = prompts[q["type"]][args.prompt_id]["template"]
        prompt_list.append(template.format(question=q["body"], context=build_context(q, args.num_snippets, args.context_source)))
        meta_list.append((q["id"], q["type"]))

    print(f"{len(prompt_list)} questions to run. Loading model...")

    backend = VLLMBackend(args.model, max_new_tokens=args.max_tokens, temperature=args.temperature,
                          tensor_parallel_size=args.tensor_parallel_size,
                          gpu_memory_utilization=args.gpu_memory_utilization,
                          max_model_len=args.max_model_len) \
              if args.backend == "local" else \
              OpenRouterBackend(args.model, max_tokens=args.max_tokens, temperature=args.temperature)

    backend.load()
    responses = backend.generate_batch(prompt_list)
    backend.unload()

    # Parse and collect results
    results = {}
    for raw, (qid, qtype) in zip(responses, meta_list):
        valid, exact = parse_exact(raw, qtype)
        results[qid] = {"exact_answer": exact, "valid": valid, "raw": raw}

    n_valid = sum(1 for r in results.values() if r["valid"])
    print(f"Done — {n_valid}/{len(results)} valid")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
