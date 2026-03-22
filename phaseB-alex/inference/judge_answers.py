"""
inference/judge_answers.py

LLM judge: draft an ideal answer from context (BioASQ-style narrative) and score it.

Uses the same “ideal answer” task as phaseB/inference/prompts_generic.json (template 7):
recall, precision, single paragraph, <200 words. A baseline structured prediction from
the generator is shown as a hint only.

Output per (question, source): ideal_answer text + correctness/faithfulness/completeness/overall
for that ideal paragraph (no gold labels required).

Usage:
    python inference/judge_answers.py \
        --inputs  outputs/exact/model1_factoid.json outputs/exact/model2_factoid.json \
        --input-data ../data/BioASQ-task14bPhaseB-testset1_hydrated.jsonl \
        --output  outputs/judged/scores.json \
        --model   anthropic/claude-sonnet-4-6 \
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
# Answer formatting
# ---------------------------------------------------------------------------

def format_answer(pred, qtype):
    """Format prediction for human-readable display in the judge prompt."""
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
            if qtype == "factoid":
                return " > ".join(str(x) for x in items[:5])
            return ", ".join(str(x) for x in items)
        return str(ea)
    return str(ea)


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
# Judge prompt
# ---------------------------------------------------------------------------

_TYPE_GUIDANCE = {
    "yesno": (
        "The answer should be 'yes' or 'no'. Evaluate whether this is the correct "
        "answer based on the evidence in the context. If the context clearly supports "
        "the claim in the question, 'yes' is correct; if it contradicts or fails to "
        "support it, 'no' is correct."
    ),
    "factoid": (
        "The answer should be one or more named entities (ranked, best first). Evaluate "
        "whether the top-ranked entity actually answers the question and is explicitly "
        "mentioned in the context. Also consider whether better answers exist in the "
        "context that were missed."
    ),
    "list": (
        "The answer should list all relevant entities. Evaluate whether the listed "
        "entities actually answer the question, are grounded in the context, and whether "
        "any relevant entities from the context were missed."
    ),
    "summary": (
        "The answer must be a single coherent **summary** paragraph: synthesize the main "
        "biomedical points from the context that address the question (definitions, mechanisms, "
        "key findings). It must not be a entity list or a yes/no. Prefer completeness and "
        "faithfulness to the sources over speculation."
    ),
}


def _default_ideal_template_path():
    return Path(__file__).resolve().parents[2] / "phaseB" / "inference" / "prompts_generic.json"


def load_ideal_task_template(prompts_path: Path, prompt_id: str) -> str:
    with open(prompts_path) as f:
        data = json.load(f)
    entry = data.get(prompt_id) or data.get(str(prompt_id))
    if not entry or "template" not in entry:
        raise KeyError(f"prompt id {prompt_id!r} not found in {prompts_path}")
    return entry["template"]


def build_judge_prompt(
    question_text,
    qtype,
    context,
    answer_str,
    *,
    base_template: str,
):
    """
    `base_template` uses {context} and {question} placeholders (prompts_generic.json #7).
    We append type guidance, baseline structured hint, and the required JSON schema.
    """
    core = base_template.format(context=context, question=question_text)
    baseline = (
        f"\n\nQuestion type (for calibration only): {qtype}\n"
        f"Type-specific guidance: {_TYPE_GUIDANCE.get(qtype, '')}\n\n"
        "Structured baseline from a generator model (hint only — verify everything against "
        "the context; it may be wrong or incomplete):\n"
        f"{answer_str}\n"
    )
    ideal_line = (
        "strictly under 200 words, answering the question from the context.\n"
        if qtype != "summary"
        else (
            "strictly under 200 words, as a **unified topic summary** (overview narrative), "
            "not a bullet list.\n"
        )
    )
    tail = (
        "\n\nFINAL STEP — Output ONLY one valid JSON object (no markdown, no text outside JSON) "
        "with exactly these keys:\n"
        "- reasoning: string, your step-by-step synthesis notes (can be brief).\n"
        "- ideal_answer: string, a single fluent paragraph, plain text only (no lists, no markdown), "
        + ideal_line +
        "- correctness, faithfulness, completeness, overall: numbers from 0.0 to 1.0 scoring "
        "YOUR ideal_answer paragraph against the context and the question.\n"
        "- rationale: string, one short sentence explaining the overall score.\n"
    )
    return core + baseline + tail


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_judge_response(text):
    """Extract JSON with ideal_answer and scores from judge response."""
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
            if isinstance(obj, dict) and any(
                k in obj for k in ("correctness", "overall", "ideal_answer")
            ):
                return obj
        except json.JSONDecodeError:
            continue
    return {
        "reasoning": "",
        "ideal_answer": "",
        "correctness": 0.0,
        "faithfulness": 0.0,
        "completeness": 0.0,
        "overall": 0.0,
        "rationale": "parse_failed",
    }


def _coerce(d, key, default=0.0):
    v = d.get(key)
    if v is None:
        return default
    try:
        return max(0.0, min(1.0, float(v)))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Prediction JSON files to judge")
    p.add_argument("--input-data", required=True,
                   help="Hydrated JSONL with questions and context")
    p.add_argument("--output", required=True,
                   help="Output JSON with judge scores")
    p.add_argument("--types", nargs="+", default=["yesno", "factoid", "list"],
                   choices=["yesno", "factoid", "list", "summary"])
    p.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    p.add_argument("--backend", default="openrouter", choices=["local", "openrouter"])
    p.add_argument("--context-source", default="abstracts", choices=["abstracts", "snippets"])
    p.add_argument("--num-context", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--request-delay", type=float, default=0.0)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument(
        "--prompts-file",
        default=str(_default_ideal_template_path()),
        help="JSON file with generic prompts (default: phaseB/inference/prompts_generic.json)",
    )
    p.add_argument(
        "--judge-prompt-id",
        default="7",
        help="Prompt template id for the ideal-answer task (default: 7 = BioASQ ideal template)",
    )
    p.add_argument(
        "--redo-parse-failed",
        action="store_true",
        help="Re-run judge rows where judge_format is set but ideal_answer is empty / parse_failed",
    )
    args = p.parse_args()

    prompts_path = Path(args.prompts_file)
    base_template = load_ideal_task_template(prompts_path, args.judge_prompt_id)

    # Load input data
    loader = BioASQDataLoader(path=args.input_data)
    questions = {q["id"]: q for q in loader if q["type"] in set(args.types)}
    print(f"Loaded {len(questions)} questions ({', '.join(args.types)})")

    # Load prediction files
    sources = {}
    for path in args.inputs:
        name = Path(path).stem
        with open(path) as f:
            sources[name] = json.load(f)
        print(f"  Loaded {path} ({len(sources[name])} predictions)")

    out = Path(args.output)

    # Resume: load existing scores if output already exists
    scores = {}
    if out.exists():
        with open(out) as f:
            existing = json.load(f)
        scores = existing.get("scores", {})
        already_done = sum(len(v) for v in scores.values())
        print(f"Resuming — {already_done} (question, source) pair(s) already in scores file")

    def _skip_judge(ent):
        if not ent:
            return False
        if ent.get("judge_format") != "ideal_v1":
            return False
        ia = (ent.get("ideal_answer") or "").strip()
        if not ia:
            return False
        if args.redo_parse_failed and str(ent.get("rationale", "")) == "parse_failed":
            return False
        return True

    # Build judge prompts — skip pairs already completed (ideal_v1)
    prompts, meta = [], []
    for qid, q in questions.items():
        context = build_context(q, args.num_context, args.context_source)
        for source_name, preds in sources.items():
            if qid not in preds:
                continue
            pred = preds[qid]
            if not pred.get("valid", False):
                continue
            if _skip_judge(scores.get(qid, {}).get(source_name)):
                continue
            answer_str = format_answer(pred, q["type"])
            prompt = build_judge_prompt(
                q["body"], q["type"], context, answer_str, base_template=base_template
            )
            prompts.append(prompt)
            meta.append((qid, source_name, q["type"]))

    if not prompts:
        print("All pairs already have judge ideal answers — nothing to do.")
    else:
        print(f"\n{len(prompts)} (question, source) pair(s) to judge. Loading model...")

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
                json.dump(
                    {
                        "judge_model": args.model,
                        "judge_prompts_file": str(prompts_path),
                        "judge_prompt_id": args.judge_prompt_id,
                        "sources": list(sources.keys()),
                        "scores": scores,
                    },
                    f,
                    indent=2,
                )

        if args.backend == "openrouter":
            # Cloud: process one-by-one so we can checkpoint frequently
            backend.load()
            for i, (prompt, (qid, source_name, qtype)) in enumerate(zip(prompts, meta)):
                raw = backend.generate(prompt)
                print(i, raw)
                if args.request_delay > 0 and i < len(prompts) - 1:
                    time.sleep(args.request_delay)
                parsed = parse_judge_response(raw)
                c  = _coerce(parsed, "correctness")
                fi = _coerce(parsed, "faithfulness")
                cp = _coerce(parsed, "completeness")
                ov = _coerce(parsed, "overall")
                if ov == 0.0 and any(v > 0 for v in (c, fi, cp)):
                    ov = (c + fi + cp) / 3.0
                ideal = str(parsed.get("ideal_answer") or "").strip()
                scores.setdefault(qid, {})[source_name] = {
                    "judge_format": "ideal_v1",
                    "judge_prompt_id": args.judge_prompt_id,
                    "reasoning": str(parsed.get("reasoning", "")),
                    "ideal_answer": ideal,
                    "correctness": c,
                    "faithfulness": fi,
                    "completeness": cp,
                    "overall": ov,
                    "rationale": str(parsed.get("rationale", "")),
                }
                if (i + 1) % CHECKPOINT_EVERY == 0:
                    _save_checkpoint()
                    print(f"  Checkpoint saved ({i+1}/{len(prompts)})")
            backend.unload()
        else:
            # Local (vLLM): batch inference, then parse + checkpoint incrementally
            backend.load()
            responses = backend.generate_batch(prompts)
            backend.unload()
            for i, (raw, (qid, source_name, qtype)) in enumerate(zip(responses, meta)):
                parsed = parse_judge_response(raw)
                c  = _coerce(parsed, "correctness")
                fi = _coerce(parsed, "faithfulness")
                cp = _coerce(parsed, "completeness")
                ov = _coerce(parsed, "overall")
                if ov == 0.0 and any(v > 0 for v in (c, fi, cp)):
                    ov = (c + fi + cp) / 3.0
                ideal = str(parsed.get("ideal_answer") or "").strip()
                scores.setdefault(qid, {})[source_name] = {
                    "judge_format": "ideal_v1",
                    "judge_prompt_id": args.judge_prompt_id,
                    "reasoning": str(parsed.get("reasoning", "")),
                    "ideal_answer": ideal,
                    "correctness": c,
                    "faithfulness": fi,
                    "completeness": cp,
                    "overall": ov,
                    "rationale": str(parsed.get("rationale", "")),
                }
                if (i + 1) % CHECKPOINT_EVERY == 0:
                    _save_checkpoint()
                    print(f"  Checkpoint saved ({i+1}/{len(prompts)})")

        _save_checkpoint()

    # Summary
    print(f"\nJudged {len(scores)} questions across {len(sources)} source(s)")
    for source_name in sources:
        source_scores = [s[source_name] for s in scores.values() if source_name in s]
        if source_scores:
            avg = sum(s["overall"] for s in source_scores) / len(source_scores)
            print(f"  {source_name}: avg overall = {avg:.3f} ({len(source_scores)} questions)")

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
