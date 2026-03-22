"""
inference/merge_answers.py

LLM-based synthesis of top-scoring answers per question.

For each question:
  1. Rank candidates by the judge's overall score (scores reflect the judge-drafted ideal paragraph)
  2. Take the top-K (prompts include judge ideal_answer text + structured exact when available)
  3. Ask an LLM to produce final exact_answer JSON and an ideal_answer paragraph

This combines quality-aware selection with reconciliation for both BioASQ exact and ideal fields.

Usage:
    python inference/merge_answers.py \
        --scores  outputs/judged/scores.json \
        --inputs  outputs/exact/model1_factoid.json outputs/exact/model2_factoid.json \
        --input-data ../data/BioASQ-task14bPhaseB-testset1_hydrated.jsonl \
        --output  outputs/merged/final.json \
        --top-k 3 \
        --model anthropic/claude-sonnet-4-6 \
        --backend openrouter
"""

import argparse, json, re, sys, time
from pathlib import Path

CHECKPOINT_EVERY = 20  # save partial results every N items

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loaders.dataloader import BioASQDataLoader
from loaders.cloud import OpenRouterBackend
from loaders.local import VLLMBackend


# ---------------------------------------------------------------------------
# Context builder (same as run_exact.py)
# ---------------------------------------------------------------------------

def build_context(question, num_context, source):
    if source == "abstracts":
        items = [d["text"] for d in question.get("documents", []) if d.get("text")][:num_context]
        label = "Abstract"
    else:
        items = question.get("snippets", [])[:num_context]
        label = "Snippet"
    if not items:
        return "(No context available)"
    return "\n\n".join(f"{label} {i+1}: {s}" for i, s in enumerate(items))


# ---------------------------------------------------------------------------
# Answer formatting for merge prompts
# ---------------------------------------------------------------------------

def format_answer_for_merge(pred, qtype):
    if qtype == "summary":
        ia = pred.get("ideal_answer")
        if isinstance(ia, str) and ia.strip():
            return ia.strip()
        return "(no draft summary)"
    ea = pred.get("exact_answer")
    if ea is None:
        return "(no answer)"
    if qtype == "yesno":
        return str(ea)
    if qtype in ("factoid", "list"):
        if isinstance(ea, list):
            items = [x[0] if isinstance(x, list) else x for x in ea]
            return json.dumps(items, ensure_ascii=False)
        return str(ea)
    return str(ea)


def format_candidate_for_merge(score_dict, pred, qtype):
    """Prefer judge-generated ideal narrative; always include structured exact for the merger."""
    structured = format_answer_for_merge(pred, qtype)
    ideal = (score_dict.get("ideal_answer") or "").strip()
    if ideal:
        return f"Ideal answer (judge-drafted): {ideal}\nStructured exact: {structured}"
    return structured


# ---------------------------------------------------------------------------
# Type-specific merge prompts
# ---------------------------------------------------------------------------

def build_merge_prompt_yesno(question, context, candidates):
    cand_block = ""
    for i, (source, answer, score) in enumerate(candidates, 1):
        cand_block += f"  Answer {i} (judge score {score:.2f}): {answer}\n"

    return (
        "You are a biomedical expert. Multiple AI models have answered a yes/no question.\n"
        "Each answer has been scored by a quality judge (0-1 scale).\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Candidate answers (with judge scores):\n{cand_block}\n"
        "Based on the context evidence (not just the scores), determine the correct answer.\n"
        "Also write a short ideal_answer: one fluent plain-text paragraph (no markdown, under 200 words) "
        "that states the conclusion and key evidence.\n\n"
        "Output ONLY this JSON (no markdown):\n"
        '{"exact_answer": "yes", "ideal_answer": "..."}\n'
        'or {"exact_answer": "no", "ideal_answer": "..."}'
    )


def build_merge_prompt_factoid(question, context, candidates):
    cand_block = ""
    for i, (source, answer, score) in enumerate(candidates, 1):
        cand_block += f"  Model {i} (judge score {score:.2f}): {answer}\n"

    return (
        "You are a biomedical expert. Multiple AI models have answered a factoid question.\n"
        "Each has been scored by a quality judge. Use the context to determine the best final answer.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Candidate answers (with judge scores):\n{cand_block}\n"
        "Task: Produce a ranked list of up to 5 answers.\n"
        "Rules:\n"
        "  1. VERIFY each candidate against the context — higher-scored candidates are likely\n"
        "     better but always check the evidence.\n"
        "  2. Rank-1 must be the most directly supported answer.\n"
        "  3. If candidates refer to the same entity differently, use the shortest correct form.\n"
        "  4. Use exact names from the context.\n"
        "  5. Match BioASQ style: short, precise (e.g. 'SATB1' not 'SATB1 protein').\n"
        "  6. ideal_answer: one fluent plain-text paragraph (no markdown, under 200 words) summarizing "
        "the answer and main supporting facts from the context.\n\n"
        "Output ONLY this JSON (no markdown):\n"
        '{"exact_answer": ["best answer", "second", "third"], "ideal_answer": "..."}'
    )


def build_merge_prompt_list(question, context, candidates):
    # Collect all candidate entities across sources
    all_entities = []
    seen = set()
    for source, answer, score in candidates:
        try:
            items = json.loads(answer) if isinstance(answer, str) else answer
        except (json.JSONDecodeError, TypeError):
            items = []
        if isinstance(items, list):
            for item in items:
                entity = item[0] if isinstance(item, list) else item
                norm = str(entity).lower().strip()
                if norm not in seen:
                    seen.add(norm)
                    all_entities.append(str(entity))

    entity_block = "\n".join(f"  - {e}" for e in all_entities)

    score_summary = ""
    for i, (source, answer, score) in enumerate(candidates, 1):
        score_summary += f"  Model {i} (judge score {score:.2f}): {answer}\n"

    return (
        "You are a biomedical expert. Multiple AI models have answered a list question.\n"
        "Each has been scored by a quality judge. Combine the best answers into the final list.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Model answers (with judge scores):\n{score_summary}\n"
        f"Combined candidate pool:\n{entity_block}\n\n"
        "Task: Produce the final entity list.\n"
        "Rules:\n"
        "  1. INCLUSION BIAS: Default to include. Only exclude if confident it doesn't answer the question.\n"
        "  2. COMPLETENESS: Check context for entities missed by all models.\n"
        "  3. DEDUPLICATION: Keep shorter/common form for synonyms.\n"
        "  4. Use exact names from the context.\n"
        "  5. ideal_answer: one fluent plain-text paragraph (no markdown, under 200 words) summarizing "
        "the list and how it answers the question.\n\n"
        "Output ONLY this JSON (no markdown):\n"
        '{"exact_answer": [["entity one"], ["entity two"], ["entity three"]], "ideal_answer": "..."}'
    )


def build_merge_prompt_summary(question, context, candidates):
    cand_block = ""
    for i, (source, answer, score) in enumerate(candidates, 1):
        cand_block += f"  Model {i} (judge score {score:.2f}): {answer}\n"

    return (
        "You are a biomedical expert. Multiple AI models produced draft **summary** answers "
        "(overview narratives) for an open-ended question.\n"
        "Each draft was scored by a quality judge. Use the context to produce one best summary.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Candidate drafts (with judge scores):\n{cand_block}\n"
        "Task: Write a single final **ideal_answer** — one fluent plain-text paragraph (no markdown, "
        "under ~200 words) that synthesizes the main biomedical points from the context. "
        "This is a topic summary, not a list of entities and not yes/no.\n"
        "Set exact_answer to an empty string (summary questions have no exact answer).\n\n"
        "Output ONLY this JSON (no markdown):\n"
        '{"exact_answer": "", "ideal_answer": "Your final summary paragraph here."}'
    )


# ---------------------------------------------------------------------------
# Response parser (same logic as run_exact.py)
# ---------------------------------------------------------------------------

def _extract_json_object(text):
    s = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    s = s.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, flags=re.DOTALL)
    for chunk in reversed(matches):
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict) and (
                "exact_answer" in obj or "ideal_answer" in obj
            ):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def parse_exact(text, qtype):
    blob = _extract_json_object(text)
    if not blob:
        return False, None, None
    try:
        if qtype == "summary":
            ideal = blob.get("ideal_answer")
            ideal_str = str(ideal).strip() if ideal is not None else ""
            if ideal_str:
                return True, "", ideal_str
            return False, None, None

        exact = blob.get("exact_answer")
        ideal = blob.get("ideal_answer")
        ideal_str = str(ideal).strip() if ideal is not None else None
        if exact is None:
            return False, None, ideal_str
        if qtype == "yesno" and isinstance(exact, str) and exact.lower() in ("yes", "no"):
            return True, exact.lower(), ideal_str
        if qtype == "factoid":
            if isinstance(exact, list) and exact and not isinstance(exact[0], list):
                return True, [[item] for item in exact], ideal_str
            return True, exact, ideal_str
        if qtype == "list":
            if isinstance(exact, list) and exact and not isinstance(exact[0], list):
                return True, [[item] for item in exact], ideal_str
            return True, exact, ideal_str
    except (json.JSONDecodeError, TypeError):
        pass
    return False, None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True,
                   help="Judge scores JSON from judge_answers.py")
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Prediction JSON files (same ones that were judged)")
    p.add_argument("--input-data", required=True,
                   help="Hydrated JSONL with questions and context")
    p.add_argument("--output", required=True)
    p.add_argument("--top-k", type=int, default=3,
                   help="Number of top answers to show to the merger LLM")
    p.add_argument("--types", nargs="+", default=["yesno", "factoid", "list"],
                   choices=["yesno", "factoid", "list", "summary"])
    p.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    p.add_argument("--backend", default="openrouter", choices=["local", "openrouter"])
    p.add_argument("--context-source", default="abstracts", choices=["abstracts", "snippets"])
    p.add_argument("--num-context", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--request-delay", type=float, default=0.0)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=8192)
    args = p.parse_args()

    out = Path(args.output)

    # Resume: load existing results if output already exists
    results = {}
    if out.exists():
        with open(out) as f:
            results = json.load(f)
        print(f"Resuming — {len(results)} question(s) already merged from {out}")

    # Load judge scores
    with open(args.scores) as f:
        judge_data = json.load(f)
    all_scores = judge_data["scores"]

    # Load predictions
    sources = {}
    for path in args.inputs:
        name = Path(path).stem
        with open(path) as f:
            sources[name] = json.load(f)

    # Load questions
    loader = BioASQDataLoader(path=args.input_data)
    questions = {q["id"]: q for q in loader if q["type"] in set(args.types)}
    print(f"Loaded {len(questions)} questions, {len(sources)} source(s)")

    # Build merge prompts — skip questions already in results
    prompts, meta = [], []
    for qid, q in questions.items():
        if qid in results:
            continue  # already merged in a previous run
        if qid not in all_scores:
            continue

        qtype = q["type"]
        context = build_context(q, args.num_context, args.context_source)

        # Rank candidates by judge score (descending)
        candidates = []
        for source_name, score_dict in all_scores[qid].items():
            if source_name not in sources or qid not in sources[source_name]:
                continue
            pred = sources[source_name][qid]
            if not pred.get("valid", False):
                continue
            answer_str = format_candidate_for_merge(score_dict, pred, qtype)
            overall = score_dict.get("overall", 0.0)
            candidates.append((source_name, answer_str, overall))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[2], reverse=True)
        top_candidates = candidates[: args.top_k]

        # Build type-specific merge prompt
        if qtype == "yesno":
            prompt = build_merge_prompt_yesno(q["body"], context, top_candidates)
        elif qtype == "factoid":
            prompt = build_merge_prompt_factoid(q["body"], context, top_candidates)
        elif qtype == "summary":
            prompt = build_merge_prompt_summary(q["body"], context, top_candidates)
        else:
            prompt = build_merge_prompt_list(q["body"], context, top_candidates)

        prompts.append(prompt)
        meta.append((qid, qtype))

    if not prompts:
        print("All questions already merged — nothing to do.")
    else:
        print(f"{len(prompts)} question(s) to merge. Loading model...")

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

        out.parent.mkdir(parents=True, exist_ok=True)

        def _save_checkpoint():
            with open(out, "w") as f:
                json.dump(results, f, indent=2)

        if args.backend == "openrouter":
            # Cloud: process one-by-one with periodic checkpoints
            backend.load()
            for i, (prompt, (qid, qtype)) in enumerate(zip(prompts, meta)):
                raw = backend.generate(prompt)
                if args.request_delay > 0 and i < len(prompts) - 1:
                    time.sleep(args.request_delay)
                valid, exact, ideal_a = parse_exact(raw, qtype)
                row = {"exact_answer": exact, "valid": valid, "raw": raw}
                if ideal_a:
                    row["ideal_answer"] = ideal_a
                elif qtype == "summary":
                    row["ideal_answer"] = ""
                results[qid] = row
                if (i + 1) % CHECKPOINT_EVERY == 0:
                    _save_checkpoint()
                    print(f"  Checkpoint saved ({i+1}/{len(prompts)})")
            backend.unload()
        else:
            # Local (vLLM): batch inference, then parse + checkpoint incrementally
            backend.load()
            responses = backend.generate_batch(prompts)
            backend.unload()
            for i, (raw, (qid, qtype)) in enumerate(zip(responses, meta)):
                valid, exact, ideal_a = parse_exact(raw, qtype)
                row = {"exact_answer": exact, "valid": valid, "raw": raw}
                if ideal_a:
                    row["ideal_answer"] = ideal_a
                elif qtype == "summary":
                    row["ideal_answer"] = ""
                results[qid] = row
                if (i + 1) % CHECKPOINT_EVERY == 0:
                    _save_checkpoint()
                    print(f"  Checkpoint saved ({i+1}/{len(prompts)})")

        _save_checkpoint()

    n_valid = sum(1 for r in results.values() if r["valid"])
    print(f"Done — {n_valid}/{len(results)} valid")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
