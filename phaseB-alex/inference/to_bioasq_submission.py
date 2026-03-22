#!/usr/bin/env python3
"""
Convert phaseB-alex merged predictions (id-keyed dict) into official BioASQ submission JSON.

The reference format is a top-level object:
  { "questions": [ { "id", "type", "body", "ideal_answer", "exact_answer" }, ... ] }

See phaseB/batch01/submission/system0.json.

Merged files from merge_answers.py look like:
  { "<qid>": { "exact_answer", "ideal_answer"?, "valid", "raw" }, ... }

For type "summary", only ideal_answer is used for content: we resolve text from
ideal_answer, then raw JSON, then a string wrongly placed in exact_answer, then
gold ideal from --input-data if present; exact_answer is always "".

Usage:
  uv run python inference/to_bioasq_submission.py \\
    --predictions phaseB-testset1/outputs/merged/submission_1_topk1.json \\
    --input-data ../data/BioASQ-task14bPhaseB-testset1_hydrated.jsonl \\
    --output phaseB-testset1/submission/system_submission_1.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loaders.dataloader import BioASQDataLoader


def _strip_json_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _ideal_from_raw_blob(raw: object) -> str:
    if raw is None or not isinstance(raw, str) or not raw.strip():
        return ""
    s = _strip_json_fence(raw)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            v = obj.get("ideal_answer")
            if isinstance(v, str) and v.strip():
                return v.strip()
    except json.JSONDecodeError:
        pass
    matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, flags=re.DOTALL)
    for chunk in reversed(matches):
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                v = obj.get("ideal_answer")
                if isinstance(v, str) and v.strip():
                    return v.strip()
        except json.JSONDecodeError:
            continue
    return ""


def _gold_ideal_to_text(gold) -> str:
    """Hydrated/training `ideal_answer` may be a list of strings."""
    if gold is None:
        return ""
    if isinstance(gold, str):
        return gold.strip()
    if isinstance(gold, list) and gold:
        parts = []
        for item in gold:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif item is not None:
                parts.append(str(item).strip())
        return " ".join(parts).strip()
    return ""


def resolve_summary_ideal_answer(q: dict, pred: dict) -> str:
    """
    Summary questions are scored on `ideal_answer` only; submission `exact_answer` is "".

    Prefer model `ideal_answer`, then JSON `ideal_answer` inside `raw`, then a string
    wrongly stored in `exact_answer`, then gold ideal from the hydrated file if present.
    """
    ideal = pred.get("ideal_answer")
    if isinstance(ideal, str) and ideal.strip():
        return ideal.strip()

    from_raw = _ideal_from_raw_blob(pred.get("raw"))
    if from_raw:
        return from_raw

    ea = pred.get("exact_answer")
    if isinstance(ea, str) and ea.strip():
        return ea.strip()
    if isinstance(ea, list) and ea:
        # Single paragraph accidentally stored as one-element list
        if len(ea) == 1 and isinstance(ea[0], str) and ea[0].strip():
            return ea[0].strip()
        if all(isinstance(x, str) for x in ea):
            joined = " ".join(x.strip() for x in ea if x.strip())
            if joined:
                return joined

    gold = _gold_ideal_to_text(q.get("ideal_answer"))
    if gold:
        return gold

    return ""


def normalize_exact_for_submission(raw, qtype: str):
    """
    Map internal exact_answer shapes to submission schema.

    submission exact_answer:
      summary → ""
      yesno   → "yes" | "no"
      factoid → list[str]           (ranked strings, NOT list[list[str]])
      list    → list[list[str]]     (each concept as [STRING])
    """
    if qtype == "summary":
        return ""

    if qtype == "yesno":
        if raw is None:
            return "no"
        s = raw if isinstance(raw, str) else str(raw)
        s = s.lower().strip()
        if s in ("yes", "no"):
            return s
        return "no"

    if qtype == "factoid":
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw] if raw.strip() else []
        if not isinstance(raw, list):
            return [str(raw)]
        out: list[str] = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) > 0:
                out.append(str(item[0]))
            elif item is not None and str(item).strip():
                out.append(str(item))
        return out

    if qtype == "list":
        if raw is None:
            return []
        if not isinstance(raw, list):
            return [[str(raw)]]
        out: list[list[str]] = []
        for item in raw:
            if isinstance(item, (list, tuple)):
                if not item:
                    continue
                cell = item[0] if not isinstance(item[0], (list, tuple)) else item[0][0]
                out.append([str(cell)])
            else:
                out.append([str(item)])
        return out

    return raw


def build_questions(loader: BioASQDataLoader, predictions: dict) -> list[dict]:
    questions: list[dict] = []
    for q in loader:
        qid = q["id"]
        qtype = q["type"]
        body = q["body"]
        pred = predictions.get(qid) or {}
        ea_in = pred.get("exact_answer")

        if qtype == "summary":
            ideal = resolve_summary_ideal_answer(q, pred)
            ea_out = ""
        else:
            ideal = pred.get("ideal_answer")
            if ideal is None or not isinstance(ideal, str):
                ideal = ""
            else:
                ideal = ideal.strip()
            ea_out = normalize_exact_for_submission(ea_in, qtype)

        questions.append(
            {
                "id": qid,
                "type": qtype,
                "body": body,
                "ideal_answer": ideal,
                "exact_answer": ea_out,
            }
        )
    return questions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, help="Merged JSON (id → prediction dict)")
    p.add_argument(
        "--input-data",
        required=True,
        help="Hydrated JSONL defining question order and id/type/body",
    )
    p.add_argument("--output", required=True, help="Output path (submission JSON)")
    args = p.parse_args()

    pred_path = Path(args.predictions)
    with open(pred_path) as f:
        predictions = json.load(f)
    if not isinstance(predictions, dict):
        raise SystemExit("--predictions must be a JSON object mapping question id → fields")

    loader = BioASQDataLoader(path=args.input_data)
    payload = {"questions": build_questions(loader, predictions)}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {len(payload['questions'])} questions → {out}")


if __name__ == "__main__":
    main()
