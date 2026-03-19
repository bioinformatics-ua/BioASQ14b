"""
synthesis/synthesize.py

Synthesis step for BioASQ Phase B answers.

Takes multiple run.py output files and:
  - Synthesizes ideal answers via LLM (grid-searches over prompt IDs)
  - Merges exact answers via voting/frequency strategies (no extra LLM call):
      yesno:   majority vote, tiebreak from --best-run
      factoid: frequency ranking top-5, tiebreak from --best-run if all disagree
      list:    majority vote per entity (>= ceil(N/2) runs), tiebreak from --best-run
      summary: None

Output format is identical to run.py so evaluate.py and
bioasq_format_converter.py work without modification.

Output filename: {out_id}_{model_name}_{num_runs}_{prompt_id}.json

Usage:
    python synthesis/synthesize.py \\
        dev/outputs/run_A.json dev/outputs/run_B.json dev/outputs/run_C.json \\
        --data-path  data/training14b/training14b.json \\
        --output-dir dev/outputs/synthesis \\
        --backend    openrouter \\
        --model      google/gemini-2.5-flash \\
        --prompt-ids 1,2,3 \\
        --out-id     exp1 \\
        --best-run   dev/outputs/run_A.json
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loaders.base import BaseModelBackend
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

def build_answers_block(runs: list[dict], qid: str) -> str:
    """
    Collect ideal_answer texts from all run files for a given question id
    and format them as a numbered list for the synthesis prompt.

    Skips runs where the question is missing or the answer is empty.
    """
    lines = []
    for i, run in enumerate(runs, 1):
        entry = run.get(qid, {})
        text = entry.get("ideal_answer") or ""
        if text:
            lines.append(f"Draft {i}: {text}")

    return "\n\n".join(lines) if lines else "(No draft answers available)"


# ---------------------------------------------------------------------------
# Exact answer merging
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Lowercase, strip whitespace and leading articles for comparison."""
    s = s.lower().strip()
    for article in ("the ", "a ", "an "):
        if s.startswith(article):
            s = s[len(article):]
    return s.strip()


def merge_exact_answers(qtype: str, runs: list[dict], qid: str, best_run_idx: int) -> object:
    """
    Merge exact answers from multiple runs using type-specific strategies.

    yesno:   majority vote — tiebreak from best_run
    factoid: frequency ranking top-5 — tiebreak from best_run if all disagree
    list:    majority vote per entity (>= ceil(N/2)) — tiebreak from best_run
    summary: None
    """
    if qtype == "summary":
        return None

    n = len(runs)
    best_exact = runs[best_run_idx].get(qid, {}).get("exact_answer")

    if qtype == "yesno":
        answers = [r.get(qid, {}).get("exact_answer") for r in runs]
        answers = [a for a in answers if a in ("yes", "no")]
        if not answers:
            return best_exact
        count = Counter(answers)
        if count["yes"] != count["no"]:
            return count.most_common(1)[0][0]
        return best_exact  # tie

    elif qtype == "factoid":
        all_candidates = []
        for r in runs:
            ans = r.get(qid, {}).get("exact_answer") or []
            if isinstance(ans, str):
                ans = [ans]
            all_candidates.extend(ans)

        if not all_candidates:
            return best_exact

        # Count by normalized form, preserve original casing
        norm_to_original: dict[str, str] = {}
        counts: Counter = Counter()
        for cand in all_candidates:
            key = normalize(str(cand))
            if key not in norm_to_original:
                norm_to_original[key] = cand
            counts[key] += 1

        top = counts.most_common()
        # If top candidate appears in more than one run, use frequency ranking
        if top[0][1] > 1 or n == 1:
            return [norm_to_original[k] for k, _ in top[:5]]
        # All runs disagree — fall back to best run
        return best_exact

    elif qtype == "list":
        threshold = math.ceil(n / 2)

        norm_to_original: dict[str, object] = {}
        counts: Counter = Counter()

        for r in runs:
            ans = r.get(qid, {}).get("exact_answer") or []
            for entity in ans:
                # Entities are nested lists: [['CD19'], ['CD3']]
                text = entity[0] if isinstance(entity, list) and entity else str(entity)
                key = normalize(str(text))
                if key not in norm_to_original:
                    norm_to_original[key] = entity
                counts[key] += 1

        merged = [norm_to_original[k] for k, c in counts.items() if c >= threshold]
        return merged if merged else best_exact

    return best_exact


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_output(text: str) -> tuple[bool, str | None]:
    """
    Extract ideal_answer from the model's output.
    Looks for the last JSON object containing 'ideal_answer'.
    Returns (valid, ideal_answer).
    """
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if matches:
        last_json_str = matches[-1]
        try:
            parsed = json.loads(last_json_str, strict=False)
            ideal = parsed.get("ideal_answer")
            if ideal:
                return True, str(ideal)
        except json.JSONDecodeError:
            pass
    return False, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Synthesize ideal answers from multiple run.py outputs"
    )

    # Positional: one or more run files to synthesize from
    parser.add_argument(
        "runs",
        nargs="+",
        help="Paths to run.py output JSON files to synthesize from"
    )

    parser.add_argument("--data-path",   required=True,  help="Path to BioASQ JSON (for question text)")
    parser.add_argument("--output-dir",  required=True,  help="Directory to save synthesis outputs")
    parser.add_argument("--out-id",      required=True,  help="Identifier prefix for output filenames")
    parser.add_argument(
        "--prompt-ids",
        default="1",
        help="Comma-separated synthesis prompt IDs to grid-search over (default: 1)"
    )
    parser.add_argument(
        "--backend",
        default="openrouter",
        choices=["local", "openrouter"],
        help="Model backend (default: openrouter)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model path (local) or model name (openrouter, e.g. google/gemini-2.5-flash)"
    )
    parser.add_argument(
        "--prompts-file",
        default=str(Path(__file__).parent / "prompts.json"),
        help="Path to synthesis prompts.json (default: synthesis/prompts.json)"
    )
    parser.add_argument("--max-tokens",             type=int,   default=500,  help="Max tokens per synthesis call (default: 500)")
    parser.add_argument("--temperature",            type=float, default=0.0,  help="Sampling temperature (default: 0 = greedy)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="VRAM fraction for vLLM (local only)")
    parser.add_argument("--max-model-len",          type=int,   default=8192, help="Max context length in tokens (local only)")
    parser.add_argument(
        "--best-run",
        default=None,
        help="Path to the run file to use as tiebreaker for exact answers (default: first run)"
    )

    args = parser.parse_args()

    if args.backend == "local" and not args.model:
        parser.error("--model is required when using --backend local")

    selected_prompt_ids = [p.strip() for p in args.prompt_ids.split(",")]

    # -----------------------------------------------------------------------
    # Load all input run files
    # -----------------------------------------------------------------------
    print(f"Loading {len(args.runs)} run file(s)...")
    runs: list[dict] = []
    for path in args.runs:
        with open(path) as f:
            runs.append(json.load(f))
    print(f"Loaded. Questions in first run: {len(runs[0])}")

    # Resolve best-run index for tiebreaking
    if args.best_run:
        try:
            best_run_idx = args.runs.index(args.best_run)
        except ValueError:
            print(f"Warning: --best-run '{args.best_run}' not in run list, falling back to index 0")
            best_run_idx = 0
    else:
        best_run_idx = 0
    print(f"Tiebreaker run: {args.runs[best_run_idx]}")

    # -----------------------------------------------------------------------
    # Load source data for question text
    # -----------------------------------------------------------------------
    print(f"Loading question text from: {args.data_path}")
    with open(args.data_path) as f:
        raw = json.load(f)

    # Support both BioASQ JSON format {"questions": [...]} and
    # lookup_abstract_B.py format {qid: {"question": ...}}
    if "questions" in raw:
        question_text = {q["id"]: q["body"]  for q in raw["questions"]}
        question_type = {q["id"]: q["type"]  for q in raw["questions"]}
    else:
        question_text = {qid: v["question"]  for qid, v in raw.items()}
        question_type = {qid: v.get("type", "summary") for qid, v in raw.items()}

    # -----------------------------------------------------------------------
    # Load synthesis prompts
    # -----------------------------------------------------------------------
    prompts = load_prompts(args.prompts_file)

    # -----------------------------------------------------------------------
    # Determine question IDs — union across all runs
    # -----------------------------------------------------------------------
    all_qids = list(runs[0].keys())
    print(f"Questions to synthesize: {len(all_qids)}")

    # -----------------------------------------------------------------------
    # Build prompts for every (question, prompt_id) combination
    #
    # Grid search: for each prompt_id we build one prompt per question,
    # then run all prompts for that id in one batched inference call.
    # -----------------------------------------------------------------------
    print(f"Grid searching over prompt IDs: {selected_prompt_ids}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_slug = (args.model or "local").replace("/", "-").replace(".", "-")

    # Load model once, run all prompt_id grids through it
    print(f"\nLoading backend: {args.backend} | model: {args.model}")
    backend: BaseModelBackend

    if args.backend == "local":
        backend = VLLMBackend(
            model_path=args.model,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    else:
        backend = OpenRouterBackend(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    backend.load()

    for pid in selected_prompt_ids:
        if pid not in prompts:
            print(f"Warning: prompt id '{pid}' not found in prompts.json, skipping.")
            continue

        template: str = prompts[pid]["template"]
        print(f"\n--- Prompt ID: {pid} ({prompts[pid]['description']}) ---")

        # Build one prompt per question
        prompt_list: list[str] = []
        meta_list: list[str] = []          # ordered list of qids

        for qid in all_qids:
            question = question_text.get(qid, "")
            answers_block = build_answers_block(runs, qid)

            formatted = template.format(
                question=question,
                answers=answers_block,
            )
            prompt_list.append(formatted)
            meta_list.append(qid)

        print(f"Running inference on {len(prompt_list)} prompts...")
        responses: list[str] = backend.generate_batch(prompt_list)

        # -------------------------------------------------------------------
        # Parse outputs and build result dict
        #
        # ideal_answer: from synthesis model
        # exact_answer: passed through from first input run (unchanged)
        # -------------------------------------------------------------------
        results: dict[str, dict] = {}
        n_valid   = 0
        n_invalid = 0

        for raw_text, qid in zip(responses, meta_list):
            valid, ideal_answer = parse_output(raw_text)

            # Fall back to the first run's ideal answer if synthesis fails
            if not valid or not ideal_answer:
                fallback = runs[0].get(qid, {})
                ideal_answer = fallback.get("ideal_answer") or ""
                n_invalid += 1
            else:
                n_valid += 1

            qtype = question_type.get(qid, "summary")
            exact_answer = merge_exact_answers(qtype, runs, qid, best_run_idx)

            results[qid] = {
                "ideal_answer": ideal_answer,
                "exact_answer": exact_answer,
                "valid":        valid,
                "raw":          raw_text,
            }

        print(f"Parsed: {n_valid} valid, {n_invalid} fell back to run[0]")

        # -------------------------------------------------------------------
        # Save — filename mirrors old summaries.py convention
        # {out_id}_{model_slug}_{num_runs}_{prompt_id}.json
        # -------------------------------------------------------------------
        out_filename = f"{args.out_id}_{model_slug}_{len(runs)}_{pid}.json"
        out_path = output_dir / out_filename

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved: {out_path}")

    backend.unload()
    print("\nAll prompt IDs done.")


if __name__ == "__main__":
    main()
