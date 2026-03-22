"""
inference/reconcile_exact.py

LLM-based reconciliation for factoid and list exact answers.

Unlike ensemble_exact.py (statistical merging), this makes a new LLM call per
question that sees ALL candidate answers from multiple models and synthesizes
the best final answer.

Why this helps:
  factoid — models often output partial/rephrased versions of the same answer
             (e.g. "FOLFOXIRI Plus Bevacizumab" vs "mFOLFOXIRI and Bevacizumab").
             The reconciler recognizes these as the same concept and outputs the
             most complete canonical form.

  list    — models miss different entities. The reconciler takes the union of
             all candidates and verifies which ones actually answer the question,
             combining recall (p3-style) with precision (p5-style) automatically.

Usage:
    python inference/reconcile_exact.py \\
        --inputs  dev/outputs/exact/model1_p4_snippets_factoid.json \\
                  dev/outputs/exact/model2_p3_abstracts_factoid.json \\
        --ground-truth  data/val_data/13B1_golden_documents.jsonl \\
        --output  dev/outputs/reconciled/factoid_reconciled.json \\
        --type    factoid \\
        --model   google/gemini-2.0-flash-001 \\
        --backend openrouter
"""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loaders.dataloader import BioASQDataLoader
from loaders.cloud import OpenRouterBackend
from loaders.local import VLLMBackend


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(question: dict, num_context: int, source: str) -> str:
    if source == "abstracts":
        items = [d["text"] for d in question["documents"] if d.get("text")][:num_context]
        label = "Abstract"
    else:
        items = question["snippets"][:num_context]
        label = "Snippet"
    if not items:
        return "(No context available)"
    return "\n\n".join(f"{label} {i+1}: {s}" for i, s in enumerate(items))


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def factoid_reconcile_prompt(question: str, context: str, candidates: list[list]) -> str:
    cand_block = ""
    for i, ranked_list in enumerate(candidates, 1):
        # Show all ranked candidates (BioASQ factoid allows up to 5)
        top = [r[0] if isinstance(r, list) else r for r in ranked_list]
        cand_block += f"  Model {i}: {' | '.join(str(x) for x in top)}\n"

    return (
        "You are a biomedical expert evaluating candidate answers to a factoid question.\n"
        "BioASQ factoid answers are short and precise — typically a name, acronym, number, or brief phrase.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Candidate answers from multiple models (all ranked candidates, separated by |):\n{cand_block}\n"
        "Task: Output a ranked list of up to 5 answers.\n"
        "Rules:\n"
        "  1. VERIFY each candidate against the context — do not pick by majority vote alone.\n"
        "     A candidate supported by fewer models but explicitly stated in the context beats\n"
        "     one agreed on by many models but not supported by the context.\n"
        "  2. Rank-1 must be the shortest correct answer that directly answers the question.\n"
        "  3. Do NOT add explanations, descriptions, or extra words beyond what is needed.\n"
        "  4. If candidates refer to the same entity with different phrasing, pick the shortest precise form.\n"
        "  5. Match the style of gold BioASQ answers: '13%' not '13% of cases', 'SATB1' not 'SATB1 protein',\n"
        "     '2-8%' not '5%' if the context gives a range.\n\n"
        "Output ONLY this JSON (no markdown):\n"
        "{\"exact_answer\": [[\"best answer\"], [\"second\"], [\"third\"]]}"
    )


def list_reconcile_prompt(question: str, context: str, candidates: list[list]) -> str:
    # Collect union of all candidate entities across models
    all_entities = []
    seen = set()
    for ranked_list in candidates:
        for item in ranked_list:
            entity = item[0] if isinstance(item, list) else item
            norm = str(entity).lower().strip()
            if norm not in seen:
                seen.add(norm)
                all_entities.append(str(entity))

    entity_block = "\n".join(f"  - {e}" for e in all_entities)

    return (
        "You are a biomedical expert. Multiple AI systems have answered the list question below.\n"
        "The combined candidate pool contains all their suggestions — some correct, some wrong, some redundant.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Combined candidate entities from all models:\n{entity_block}\n\n"
        "Task — follow ALL rules strictly:\n\n"
        "  1. INCLUSION BIAS: Your default is to INCLUDE. Only exclude an entity if you are confident\n"
        "     it does not answer the question. Uncertain? Keep it.\n\n"
        "  2. COMPLETENESS: Scan the context carefully for correct entities NOT in the candidate list\n"
        "     and add them. Missing a correct entity costs exactly as much as including a wrong one.\n\n"
        "  3. GRANULARITY: Use the broadest correct term instead of listing sub-types separately.\n"
        "     Example: 'muscular symptoms' instead of myopathy + myalgia + cramps listed separately.\n\n"
        "  4. DEDUPLICATION: If two candidates mean the same thing (e.g. 'MI' and 'myocardial infarction'),\n"
        "     keep only the shorter/more common form.\n\n"
        "  5. SCOPE: Only trim the list if the question explicitly asks for 'main', 'primary', or 'most\n"
        "     important' items. Otherwise include every entity that answers the question.\n\n"
        "Output ONLY this JSON (no markdown):\n"
        "{\"exact_answer\": [[\"entity one\"], [\"entity two\"], [\"entity three\"]]}"
    )


# ---------------------------------------------------------------------------
# Response parser (reuses same logic as run_exact.py)
# ---------------------------------------------------------------------------

def parse_exact(text: str, qtype: str):
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if not matches:
        return False, None
    try:
        exact = json.loads(matches[-1], strict=False).get("exact_answer")
        if exact is None:
            return False, None
        if qtype == "factoid":
            if isinstance(exact, list) and exact:
                if not isinstance(exact[0], list):
                    return True, [[item] for item in exact]
            return True, exact
        if qtype == "list":
            if isinstance(exact, list) and exact:
                if not isinstance(exact[0], list):
                    return True, [[item] for item in exact]
            return True, exact
    except json.JSONDecodeError:
        pass
    return False, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs",         nargs="+", required=True,
                   help="Two or more prediction JSON files to reconcile")
    p.add_argument("--ground-truth",   required=True,
                   help="Input JSONL with questions and context (for building prompts)")
    p.add_argument("--output",         required=True,
                   help="Output JSON file for reconciled predictions")
    p.add_argument("--type",           required=True, choices=["factoid", "list"],
                   help="Question type to reconcile")
    p.add_argument("--model",          default="google/gemini-2.0-flash-001")
    p.add_argument("--backend",        default="openrouter", choices=["local", "openrouter"])
    p.add_argument("--context-source", default="abstracts", choices=["abstracts", "snippets"])
    p.add_argument("--num-context",    type=int, default=10)
    p.add_argument("--max-tokens",     type=int, default=1000)
    p.add_argument("--temperature",    type=float, default=0.0)
    p.add_argument("--request-delay",  type=float, default=0.0)
    p.add_argument("--tensor-parallel-size",   type=int,   default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len",          type=int,   default=8192)
    args = p.parse_args()

    # Load all input prediction files
    all_preds = []
    for path in args.inputs:
        with open(path) as f:
            all_preds.append(json.load(f))
    print(f"Loaded {len(all_preds)} input file(s)")

    # Load questions for context
    loader = BioASQDataLoader(path=args.ground_truth)
    questions = {q["id"]: q for q in loader if q["type"] == args.type}
    print(f"Found {len(questions)} {args.type} questions in ground truth")

    # Find questions that appear in at least one input file
    qids = [qid for qid in questions if any(qid in preds for preds in all_preds)]
    print(f"{len(qids)} questions to reconcile. Building prompts...")

    # Build one prompt per question
    prompts, meta = [], []
    for qid in qids:
        q = questions[qid]
        context = build_context(q, args.num_context, args.context_source)

        # Collect candidate answer lists from each model (skip missing)
        candidates = []
        for preds in all_preds:
            if qid in preds and preds[qid].get("exact_answer"):
                candidates.append(preds[qid]["exact_answer"])

        if not candidates:
            continue

        if args.type == "factoid":
            prompt = factoid_reconcile_prompt(q["body"], context, candidates)
        else:
            prompt = list_reconcile_prompt(q["body"], context, candidates)

        prompts.append(prompt)
        meta.append(qid)

    print(f"{len(prompts)} prompts built. Loading model...")

    if args.backend == "local":
        backend = VLLMBackend(
            model_path=args.model,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    else:
        backend = OpenRouterBackend(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            request_delay=args.request_delay,
        )

    backend.load()
    responses = backend.generate_batch(prompts)
    backend.unload()

    # Parse results
    results = {}
    for raw, qid in zip(responses, meta):
        valid, exact = parse_exact(raw, args.type)
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
