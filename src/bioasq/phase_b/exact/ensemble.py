"""
ensemble_exact.py — Combine multiple run_exact.py outputs into one ensemble prediction.

Merging strategies per type:
  yes/no  : majority vote — pick whichever answer appears most across all inputs
  factoid : Reciprocal Rank Fusion (RRF) — score each candidate by sum of 1/(rank+60),
            re-rank and return top-5
  list    : frequency threshold — keep entities that appear in >= threshold fraction
            of input files (default 0.5, i.e. majority)

Usage:
    python inference/ensemble_exact.py \\
        --inputs dev/outputs/exact/model1_p4_yesno.json dev/outputs/exact/model2_p4_yesno.json \\
        --output dev/outputs/exact/ensemble_yesno.json

    # Or use glob patterns (quote them to avoid shell expansion):
    python inference/ensemble_exact.py \\
        --inputs "dev/outputs/exact/*_p4_*_yesno.json" \\
        --output dev/outputs/exact/ensemble_yesno.json
"""

import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated, Any, Literal

import orjson
import typer

app = typer.Typer()

# ---------------------------------------------------------------------------
# Normalisation (same as evaluation_exact.py)
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# Per-type ensemble functions
# ---------------------------------------------------------------------------


def ensemble_yesno(predictions: list[str | None]) -> str:
    """Majority vote over yes/no predictions. Ties go to 'yes'."""
    votes = [p.lower().strip() for p in predictions if p and p in ("yes", "no")]
    if not votes:
        return "yes"
    counts = Counter(votes)
    return counts.most_common(1)[0][0]


def ensemble_factoid(
    ranked_lists: list[list[str] | None], k: int = 60, top_n: int = 5
) -> list[str]:
    """
    Reciprocal Rank Fusion across multiple ranked candidate lists.
    Each list is ordered best-first (rank 1 = index 0).
    Returns the top_n candidates by fused RRF score.
    """
    scores: dict[str, float] = defaultdict(float)
    # Map normalised form → original text (first seen wins)
    originals: dict[str, str] = {}

    for ranked in ranked_lists:
        if not ranked:
            continue
        for rank, candidate in enumerate(ranked, start=1):
            norm = _normalise(candidate)
            scores[norm] += 1.0 / (rank + k)
            if norm not in originals:
                originals[norm] = candidate

    if not scores:
        return []

    sorted_norms = sorted(scores, key=lambda n: scores[n], reverse=True)
    return [originals[n] for n in sorted_norms[:top_n]]


def ensemble_list(entity_lists: list[list[str] | None], threshold: float = 0.5) -> list[str]:
    """
    Frequency-based ensemble for list answers.
    Keep entities that appear (after normalisation) in >= threshold fraction of inputs.
    Ties are broken by frequency (descending).
    """
    valid = [lst for lst in entity_lists if lst]
    if not valid:
        return []

    n_inputs = len(valid)
    min_count = max(1, round(threshold * n_inputs))

    # Count normalised occurrences; track canonical form (first seen)
    counts: Counter = Counter()
    originals: dict[str, str] = {}

    for lst in valid:
        seen_in_this = set()
        for entity in lst:
            norm = _normalise(entity)
            if norm not in seen_in_this:
                counts[norm] += 1
                seen_in_this.add(norm)
            if norm not in originals:
                originals[norm] = entity

    return [originals[norm] for norm, cnt in counts.most_common() if cnt >= min_count]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_inputs(paths: list[Path]) -> list[dict[str, Any]]:
    """Load all prediction files matched by the given glob patterns."""
    data = []
    for path in paths:
        data.append(orjson.loads(path.read_bytes()))
        print(f"  Loaded: {path}  ({len(data[-1])} questions)")
    return data


def ensemble(
    inputs: list[dict],
    threshold: float,
    rrf_k: int,
    qtype: Literal["auto", "yesno", "factoid", "list"],
) -> dict:
    """Merge N prediction dicts into one ensemble prediction dict."""
    # Collect all question IDs across all inputs
    all_ids = set()
    for inp in inputs:
        all_ids.update(inp.keys())

    results = {}
    for qid in all_ids:
        entries = [inp[qid] for inp in inputs if qid in inp]
        if not entries:
            continue

        # Determine question type by inspecting answers
        # yes/no: exact_answer is "yes" or "no"
        # factoid: exact_answer is a list of strings (ranked)
        # list: exact_answer is a list of strings (unordered set)
        # We infer type from the first valid entry
        sample_answer = next(
            (e["exact_answer"] for e in entries if e.get("exact_answer") is not None), None
        )

        if sample_answer is None:
            results[qid] = {"exact_answer": None, "valid": False}
            continue

        if isinstance(sample_answer, str):
            # yes/no
            preds = [e["exact_answer"] for e in entries if isinstance(e.get("exact_answer"), str)]
            results[qid] = {"exact_answer": ensemble_yesno(preds), "valid": True}

        elif isinstance(sample_answer, list):
            # Both factoid and list store lists — we treat them the same way but
            # factoid uses RRF (order matters) and list uses frequency threshold.
            # Caller can force the type via --type flag; otherwise we use RRF
            # since it degrades gracefully to frequency when ordering is weak.
            lists = [e["exact_answer"] for e in entries if isinstance(e.get("exact_answer"), list)]

            if qtype == "list":
                merged = ensemble_list(lists, threshold=threshold)
            else:
                # factoid or auto — use RRF
                merged = ensemble_factoid(lists, k=rrf_k)

            results[qid] = {"exact_answer": merged, "valid": bool(merged)}

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    inputs: Annotated[list[Path], typer.Option(help="Input JSON files")],
    output: Annotated[Path, typer.Option(help="Path to write the merged output JSON")],
    qtype: Annotated[
        Literal["auto", "yesno", "factoid", "list"], typer.Option("auto", help="Question type")
    ],
    threshold: Annotated[
        float, typer.Option(default=0.5, help="Frequency threshold for list ensembling")
    ],
    rrf_k: Annotated[int, typer.Option(default=60, help="RRF k constant for factoid ensembling")],
) -> None:

    print(f"Loading {len(inputs)} input pattern(s)...")
    inputs = load_inputs(inputs)
    print(
        f"Ensembling {len(inputs)} files over"
        f"{len(set().union(*[d.keys() for d in inputs]))} questions..."
    )

    merged = ensemble(inputs, threshold=threshold, rrf_k=rrf_k, qtype=qtype)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        f.write(orjson.dumps(merged))

    valid = sum(1 for v in merged.values() if v["valid"])
    print(f"Done — {valid}/{len(merged)} valid predictions saved to {output}")


if __name__ == "__main__":
    app()
