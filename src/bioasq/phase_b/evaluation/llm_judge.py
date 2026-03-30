import re
from typing import Any, cast

import orjson

from bioasq.phase_b.backends.base import BaseModelBackend
from bioasq.phase_b.backends.cloud import OpenRouterBackend
from bioasq.phase_b.backends.local import VLLMBackend
from bioasq.phase_b.evaluation.prediction_normalize import (
    context_from_question,
    gold_references_html,
    ideal_answer_text_for_judge,
)
from bioasq.phase_b.evaluation.schemas import JudgeScores

_JUDGE_SYSTEM = (
    "You are a biomedical question-answering evaluator. "
    "Your only output is a single raw JSON object — no markdown, no code fences, no explanation. "
    "Keys: correctness (number 0-1), faithfulness (number 0-1), completeness (number 0-1), "
    "overall (number 0-1), rationale (string under 80 words). "
    'Example: {"correctness": 0.9, "faithfulness": 0.8, "completeness": 0.7, "overall": 0.8, "rationale": "..."}'
)


def make_judge_messages(
    question: str,
    qtype: str,
    context: str,
    references_block: str,
    candidate: str,
) -> list[dict]:
    user = (
        "/no_think\n"
        f"Question type: {qtype}\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Reference ideal answers (multiple phrasings may all be valid):\n{references_block or '(none)'}\n\n"
        f"Candidate answer:\n{candidate}\n\n"
        "Score the candidate and output the JSON object now."
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_judge_json(text: str) -> dict[str, object]:
    # Strip Qwen3 thinking blocks
    s = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    s = s.strip()
    try:
        obj = orjson.loads(s)
        if isinstance(obj, dict):
            return obj
    except orjson.JSONDecodeError:
        pass
    m = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, flags=re.DOTALL)
    for chunk in reversed(m):
        try:
            obj = orjson.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except orjson.JSONDecodeError:
            continue
    # Last resort: truncated JSON repair — close any open string then the object
    for candidate in (s, text):
        if candidate.strip().startswith("{"):
            repaired = candidate.rstrip().rstrip(",")
            if not repaired.endswith("}"):
                if repaired.endswith('"'):
                    repaired += "}"
                else:
                    repaired += '"}'
            try:
                obj = orjson.loads(repaired)
                if isinstance(obj, dict) and any(
                    k in obj for k in ("correctness", "faithfulness", "overall")
                ):
                    print("[judge] WARNING: repaired truncated JSON", flush=True)
                    return obj
            except orjson.JSONDecodeError:
                pass
    raise ValueError(
        f"judge response did not contain valid JSON object.\n"
        f"--- RAW RESPONSE (repr) ---\n{text!r}\n"
        f"--- RAW RESPONSE (stripped) ---\n{s!r}\n"
        f"---"
    )


def coerce_judge_scores(d: dict[str, object]) -> JudgeScores:
    def f(name: str, default: float = 0.0) -> float:
        v = d.get(name)
        if v is None:
            return default
        x = float(v)
        return max(0.0, min(1.0, x))

    c = f("correctness")
    fh = f("faithfulness")
    cp = f("completeness")
    ov = d.get("overall")
    overall = max(0.0, min(1.0, float(ov))) if ov is not None else (c + fh + cp) / 3.0
    rationale = d.get("rationale")
    r = rationale or ""
    return JudgeScores(
        correctness=c, faithfulness=fh, completeness=cp, overall=overall, rationale=r
    )


def build_backend(
    backend: str,
    model: str,
    max_tokens: int,
    temperature: float,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> BaseModelBackend:
    if backend == "local":
        return VLLMBackend(
            model_path=model,
            max_new_tokens=max_tokens,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    if backend == "openrouter":
        return OpenRouterBackend(model=model, max_tokens=max_tokens, temperature=temperature)
    raise ValueError(f"unknown backend {backend}")


def run_judge_batch(
    backend: BaseModelBackend,
    pairs: list[tuple[dict[str, object], dict[str, object]]],
    context_max_chars: int,
) -> dict[str, JudgeScores]:
    if not pairs:
        return {}
    backend.load()
    try:
        results: dict[str, JudgeScores] = {}
        for q, pred in pairs:
            qid = str(q["id"])
            cand = ideal_answer_text_for_judge(cast("dict[str, Any]", pred))
            ia_raw = q.get("ideal_answer")
            strs: list[str] = (
                [x for x in ia_raw if isinstance(x, str)] if isinstance(ia_raw, list) else []
            )
            rb = gold_references_html(strs)
            ctx = context_from_question(cast("dict[str, Any]", q), context_max_chars)
            messages = make_judge_messages(
                str(q["body"]),
                str(q["type"]),
                ctx,
                rb,
                cand or "(empty answer)",
            )
            raw = backend.generate_chat(messages)
            print(f"[judge] qid={qid} raw={raw!r}", flush=True)
            parsed = parse_judge_json(raw)
            results[qid] = coerce_judge_scores(cast("dict[str, object]", parsed))
        return results
    finally:
        backend.unload()


def judge_score_means(scores: dict[str, JudgeScores]) -> dict[str, float]:
    if not scores:
        return {
            "correctness": 0.0,
            "faithfulness": 0.0,
            "completeness": 0.0,
            "overall": 0.0,
            "n": 0.0,
        }
    n = len(scores)
    return {
        "correctness": sum(s.correctness for s in scores.values()) / n,
        "faithfulness": sum(s.faithfulness for s in scores.values()) / n,
        "completeness": sum(s.completeness for s in scores.values()) / n,
        "overall": sum(s.overall for s in scores.values()) / n,
        "n": float(n),
    }
