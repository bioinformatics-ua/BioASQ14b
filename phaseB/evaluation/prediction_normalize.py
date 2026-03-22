from __future__ import annotations

from typing import Any, cast


def metrics_ready_predictions(raw: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for qid, rec in raw.items():
        d = dict(rec)
        ia = d.get("ideal_answer")
        if ia is None or ia == "":
            t = d.get("text")
            if isinstance(t, str) and t:
                d["ideal_answer"] = t
        elif isinstance(ia, list):
            d["ideal_answer"] = " ".join(str(x) for x in ia)
        out[qid] = d
    return out


def ideal_answer_text_for_judge(pred: dict[str, Any]) -> str:
    ia = pred.get("ideal_answer")
    if isinstance(ia, str) and ia.strip():
        return ia.strip()
    t = pred.get("text")
    if isinstance(t, str) and t.strip():
        return t.strip()
    return ""


def gold_references_html(ideal_answer: list[str] | None) -> str:
    if not ideal_answer:
        return ""
    lines = [f"- {s.strip()}" for s in ideal_answer if isinstance(s, str) and s.strip()]
    return "\n".join(lines)


def context_from_question(q: dict[str, Any], max_chars: int) -> str:
    docs = q.get("documents") or []
    parts: list[str] = []
    for d in docs:
        if isinstance(d, dict) and isinstance(d.get("text"), str):
            parts.append(cast(str, d["text"]))
    snippets = q.get("snippets") or []
    for s in snippets:
        if isinstance(s, str):
            parts.append(s)
    blob = "\n\n".join(parts)
    if len(blob) <= max_chars:
        return blob
    return blob[: max_chars - 20] + "\n\n[truncated]"
