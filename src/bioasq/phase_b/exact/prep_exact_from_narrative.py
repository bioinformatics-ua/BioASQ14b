#!/usr/bin/env python3
"""
Fill ideal_answer + exact_answer from a sweep file that only has narrative `text`.

If you pass --exact-from (e.g. merge_answers output like submission_1_topk1.json),
exact_answer is copied from that file per question type; heuristics apply only when
a stored exact is missing or empty.

Usage:
  uv run python inference/prep_exact_from_narrative.py \\
    --predictions path/to/claude_....json \\
    --hydrated ../../data/BioASQ-task14bPhaseB-testset1_hydrated.jsonl \\
    --exact-from phaseB-testset1/outputs/merged/submission_1_topk1.json \\
    --output path/to/claude_...._submission_ready.json
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated

import orjson
import typer

if TYPE_CHECKING:
    from pathlib import Path

app = typer.Typer()


def load_types(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            d = orjson.loads(line)
            out[d["id"]] = d["type"]
    return out


def exact_from_merged_usable(qtype: str, exact: object) -> bool:
    if exact is None:
        return False
    if qtype == "yesno":
        return isinstance(exact, str) and exact.lower().strip() in ("yes", "no")
    if qtype == "factoid":
        if isinstance(exact, str):
            return bool(exact.strip())
        if isinstance(exact, list):
            return bool(exact)
        return False
    if qtype == "list":
        return isinstance(exact, list) and bool(exact)
    return False


def pick_exact_answer(
    qtype: str,
    text: str,
    merged_row: dict | None,
) -> object:
    if merged_row and qtype != "summary":
        cand = merged_row.get("exact_answer")
        if exact_from_merged_usable(qtype, cand):
            if qtype == "yesno" and isinstance(cand, str):
                return cand.lower().strip()
            return cand
    if qtype == "summary":
        return ""
    if qtype == "yesno":
        return guess_yesno(text)
    if qtype == "factoid":
        return guess_factoid(text)
    if qtype == "list":
        return guess_list(text)
    raise ValueError(qtype)


def guess_yesno(text: str) -> str:
    low = text.strip().lower()
    if low.startswith(("yes,", "yes ")):
        return "yes"
    if low.startswith(("no,", "no ")):
        return "no"
    head = low[:320]
    if re.match(r"^current evidence (does not|has not)", head):
        return "no"
    if re.match(r"^the .{0,120} does not ", head):
        return "no"
    neg = (
        "does not appear",
        "does not conclusively",
        "has not been shown",
        "have not been",
        "is not inherently",
        "are not ",
        "cannot be used",
        "cannot ",
        "no evidence",
        "not been demonstrated",
        "unlikely that",
    )
    if any(p in head for p in neg):
        return "no"
    pos = (
        "research has demonstrated",
        "studies confirm",
        "multiple examples",
        "multiple studies",
        "is well documented",
        "clearly ",
    )
    if any(p in head for p in pos):
        return "yes"
    return "no"


def first_sentence(text: str, max_len: int = 400) -> str:
    t = text.strip()
    for sep in (". ", "? ", "! "):
        if sep in t:
            t = t.split(sep)[0] + sep.strip()
            break
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_len:
        t = t[: max_len - 1].rsplit(" ", 1)[0] + "…"
    return t


def guess_factoid(text: str) -> list[str]:
    s = first_sentence(text, max_len=450)
    return [s] if s else []


def _split_enum_chunk(chunk: str) -> list[str]:
    chunk = chunk.strip()
    if not chunk:
        return []
    parts = re.split(r",\s*and\s+", chunk, flags=re.I)
    if len(parts) == 1:
        parts = re.split(r"\s+and\s+", chunk)
    if len(parts) == 1:
        parts = re.split(r";\s+", chunk)
    if len(parts) == 1:
        parts = re.split(r",\s+", chunk)
    out: list[str] = []
    for p in parts:
        p = p.strip().rstrip(".,;").strip()
        if 2 < len(p) < 220:
            out.append(p)
    return out


def guess_list(text: str) -> list[list[str]]:
    items: list[str] = []

    m = re.search(r":\s*([^.]{10,800})\.", text)
    if m:
        got = _split_enum_chunk(m.group(1))
        if len(got) >= 2:
            items.extend(got)

    if len(items) < 2:
        for sent in re.split(r"(?<=[.!?])\s+", text):
            mm = re.search(r"\binclude[s]?\s+([^.]{8,500})", sent, re.I)
            if not mm:
                mm = re.search(r"\b(such as|including)\s+([^.]{8,500})", sent, re.I)
                if mm:
                    chunk = mm.group(2)
                else:
                    continue
            else:
                chunk = mm.group(1)
            got = _split_enum_chunk(chunk)
            for g in got:
                if g not in items:
                    items.append(g)
            if len(items) >= 3:
                break

    if len(items) < 2:
        for sent in re.split(r"(?<=[.!?])\s+", text):
            s = sent.strip()
            if len(s) < 30 or len(s) > 350:
                continue
            if s.lower().startswith(("these ", "this ", "they ")):
                continue
            items.append(s)
            if len(items) >= 6:
                break

    dedup: list[str] = []
    seen: set[str] = set()
    for it in items:
        key = it.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(it)
    return [[x] for x in dedup[:20]]


@app.command()
def main(
    predictions: Annotated[Path, typer.Option(..., help="Path to predictions JSON file")],
    hydrated: Annotated[Path, typer.Option(..., help="Path to hydrated jsonl file")],
    exact_from: Annotated[Path | None, typer.Option(default=None, help="Path to merged JSON file")],
    output: Annotated[Path, typer.Option(..., help="Path to output JSON file")],
) -> None:
    types = load_types(hydrated)
    preds: dict = orjson.loads(predictions.read_bytes())

    merged: dict | None = None
    if exact_from:
        merged = orjson.loads(exact_from.read_bytes())

    out: dict = {}
    for qid, row in preds.items():
        qtype = types.get(qid)
        if qtype is None:
            raise SystemExit(f"unknown id in predictions: {qid}")
        text = (row.get("text") or "").strip()
        base = {"text": text, "valid": row.get("valid", True)}
        ideal = text
        mrow = merged.get(qid) if merged else None
        if merged is not None and qid not in merged and qtype != "summary":
            typer.echo(
                f"warning: --exact-from missing id {qid} ({qtype}) — using heuristic exact",
                err=True,
            )
        ea = pick_exact_answer(qtype, text, mrow)
        out[qid] = {**base, "ideal_answer": ideal, "exact_answer": ea}

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(orjson.dumps(out))


if __name__ == "__main__":
    app()
