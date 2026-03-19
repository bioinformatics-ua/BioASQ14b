"""
inference/run.py

Main inference runner for BioASQ Phase B.

Takes a BioASQ JSON file (training or competition batch), runs each question
through the selected prompt and model backend, and saves raw predictions to a
JSON file. This output is later consumed by the evaluator (dev) or the format
converter (competition submission).

Usage:
    python inference/run.py \
        --input  data/training14b/training14b.json \
        --output dev/outputs/run_p1.json \
        --model  /path/to/model \
        --prompt-id 1 \
        --num-snippets 5 \
        --limit 20
"""
import argparse
import json
import re
import sys
from pathlib import Path

# Allow imports from the project root (loaders/, inference/) regardless of
# which directory the script is called from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loaders.dataloader import BioASQDataLoader
from loaders.base import BaseModelBackend
from loaders.local import VLLMBackend
from loaders.cloud import OpenRouterBackend


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(prompts_path: str) -> dict:
    """Load prompt templates from prompts.json."""
    with open(prompts_path) as f:
        return json.load(f)


def get_template(prompts: dict, question_type: str, prompt_id: str) -> str:
    """
    Retrieve a specific prompt template for a given question type and prompt id.
    Raises a clear error if the combination doesn't exist.
    """
    if question_type not in prompts:
        raise ValueError(f"No prompts defined for question type: '{question_type}'")
    if prompt_id not in prompts[question_type]:
        raise ValueError(
            f"Prompt id '{prompt_id}' not found for type '{question_type}'. "
            f"Available: {list(prompts[question_type].keys())}"
        )
    return prompts[question_type][prompt_id]["template"]


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def build_context(question: dict, num_snippets: int) -> str:
    """
    Build the {context} string that is injected into the prompt.

    Uses snippets only — they are expert-curated extracts and provide the most
    focused evidence. The number of snippets to include is configurable via
    --num-snippets.
    """
    snippets = question["snippets"][:num_snippets]

    if not snippets:
        return "(No context available)"

    lines: list[str] = []
    for i, snippet in enumerate(snippets, 1):
        lines.append(f"Snippet {i}: {snippet}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# JSON output parsing
# ---------------------------------------------------------------------------

def parse_output(text: str, question_type: str) -> tuple[bool, str | None, object]:
    """
    Parse the model's raw text output into (valid, ideal_answer, exact_answer).

    Looks for the last JSON object in the output (after chain-of-thought reasoning).
    Returns valid=False and the raw text if parsing fails, so we can inspect
    failures without losing any output.
    """
    # Find all JSON-like blocks in the output
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)

    if matches:
        # Take the last match — model reasons first, then outputs JSON at the end
        last_json_str = matches[-1]
        try:
            parsed = json.loads(last_json_str, strict=False)
            ideal = parsed.get("ideal_answer", None)
            exact = parsed.get("exact_answer", None)

            # Both fields present (or just ideal for summary)
            if ideal is not None:
                return True, ideal, exact
        except json.JSONDecodeError:
            pass

    # Parsing failed — return raw output so nothing is silently lost
    return False, None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BioASQ Phase B inference runner")

    parser.add_argument("--input",        required=True,  help="Path to BioASQ JSON (training or batch)")
    parser.add_argument("--output",       required=True,  help="Path to save predictions JSON")
    parser.add_argument(
        "--backend",
        default="local",
        choices=["local", "openrouter"],
        help="Model backend to use (default: local)"
    )
    # --model meaning depends on backend:
    #   local      → path to model weights on disk
    #   openrouter → model name e.g. "anthropic/claude-sonnet-4-6"
    #   openai     → model name e.g. "gpt-4o"
    parser.add_argument("--model",        default=None,   help="Model path (local) or model name (cloud)")
    parser.add_argument("--prompt-id",    default="1",    help="Prompt variant to use (default: 1)")
    parser.add_argument("--mode",         default= "abstract", help="Abstract of snippet.")
    parser.add_argument("--num-support", type=int, default=5, help="Number of abstracts/snippets per question (default: 5)")
    # parser.add_argument(
    #     "--question-types",
    #     nargs="+",
    #     default=["yesno", "factoid", "list", "summary"],
    #     help="Question types to run (default: all four)"
    # )
    parser.add_argument(
        "--prompts-file",
        default=str(Path(__file__).parent / "prompts.json"),
        help="Path to prompts.json (default: inference/prompts.json)"
    )
    parser.add_argument("--max-tokens",             type=int,   default=1000, help="Max new tokens per generation")
    parser.add_argument("--temperature",            type=float, default=0.0,  help="Sampling temperature (default: 0 = greedy)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="Fraction of GPU VRAM vLLM may use (local only)")
    parser.add_argument("--tensor-parallel-size",   type=int,   default=1,    help="Number of GPUs to shard the model across (local only)")
    parser.add_argument("--max-model-len",          type=int,   default=8192, help="Max context length in tokens (local only)")

    args = parser.parse_args()

    # Validate: local backend requires a model path
    if args.backend == "local" and not args.model:
        parser.error("--model is required when using --backend local")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print(f"Loading data from: {args.input}")
    loader = BioASQDataLoader(
        path=args.input,
        # limit=args.limit,
        # question_types=args.question_types,
        # random_seed=args.random_seed,
    )
    print(f"Loaded {len(loader)} questions | types: {args.question_types} | prompt: {args.prompt_id}")

    # -----------------------------------------------------------------------
    # Load prompts
    # -----------------------------------------------------------------------
    prompts: dict = load_prompts(args.prompts_file)

    # -----------------------------------------------------------------------
    # Build all prompts upfront for batched inference
    #
    # vLLM is most efficient when given all prompts at once rather than one
    # at a time, so we collect everything before loading the model.
    # -----------------------------------------------------------------------
    print("Building prompts...")
    prompt_list: list[str] = []                  # ordered list of formatted prompt strings
    meta_list: list[tuple[str, str]] = []        # parallel list of (id, type) so we can map results back

    for question in loader:
        template: str = get_template(prompts, question["type"], args.prompt_id)
        context: str  = build_context(question, args.num_snippets)

        formatted: str = template.format(
            question=question["body"],
            context=context,
        )

        prompt_list.append(formatted)
        meta_list.append((question["id"], question["type"]))

    print(f"Built {len(prompt_list)} prompts. Loading model...")

    # -----------------------------------------------------------------------
    # Instantiate the selected backend
    #
    # All three backends share the same load/generate/unload interface so
    # the rest of the script is identical regardless of which is chosen.
    # -----------------------------------------------------------------------
    backend: BaseModelBackend

    if args.backend == "local":
        backend = VLLMBackend(
            model_path=args.model,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
    elif args.backend == "openrouter":
        backend = OpenRouterBackend(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    backend.load()
    print("Running inference...")

    # generate_batch: vLLM sends all prompts at once (fast),
    # cloud backends fall back to a loop via the base class default.
    responses: list[str] = backend.generate_batch(prompt_list)

    backend.unload()
    print(f"Inference complete. Parsing outputs...")

    # -----------------------------------------------------------------------
    # Parse outputs and collect results
    # -----------------------------------------------------------------------
    results: dict[str, dict] = {}
    n_valid   = 0
    n_invalid = 0

    for raw_text, (qid, qtype) in zip(responses, meta_list):
        valid, ideal_answer, exact_answer = parse_output(raw_text, qtype)

        results[qid] = {
            "ideal_answer": ideal_answer,
            "exact_answer": exact_answer,
            "valid":        valid,
            "raw":          raw_text,   # always kept for debugging
        }

        if valid:
            n_valid += 1
        else:
            n_invalid += 1

    print(f"Parsed: {n_valid} valid, {n_invalid} invalid (JSON parse failures)")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
