"""Synthesis step for BioASQ Phase B answers.

Takes multiple inference run outputs and synthesises final answers:
- **Ideal answers**: via LLM synthesis (grid-search over prompt IDs)
- **Exact answers**: via voting/frequency strategies (no LLM):
    - yesno: majority vote, tiebreak from best run
    - factoid: frequency ranking top-5
    - list: majority vote per entity (≥ ceil(N/2))
    - summary: None

Refactored from ``phaseB-alex/synthesis/synthesize.py`` (the more
feature-complete version with exact answer merging).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence

from bioasq.common.protocols import BaseModelBackend
from bioasq.common.types import SynthesisResult

# Exact answer type: can be a string, list of strings, list of synonym lists, or None
type ExactAnswer = list[str] | list[list[str]] | str | None

# Type for run result dictionaries: {qid: {field: value}}
type RunResultEntry = Mapping[str, str | ExactAnswer]
type RunResults = Mapping[str, RunResultEntry]

# Prompt template types
type PromptTemplate = dict[str, str]
type PromptTemplates = Mapping[str, PromptTemplate]


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def _normalize(s: str) -> str:
    """Lowercase, strip whitespace and leading articles for comparison."""
    s = s.lower().strip()
    for article in ("the ", "a ", "an "):
        if s.startswith(article):
            s = s[len(article):]
    return s.strip()


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def build_answers_block(
    runs: Sequence[RunResults],
    qid: str,
) -> str:
    """Format draft ideal answers from multiple runs for the synthesis prompt."""
    lines: list[str] = []
    for i, run in enumerate(runs, 1):
        entry: RunResultEntry = run.get(qid, {})
        text: str = str(entry.get("ideal_answer", ""))
        if text:
            lines.append(f"Draft {i}: {text}")
    return "\n\n".join(lines) if lines else "(No draft answers available)"


# ---------------------------------------------------------------------------
# Exact answer merging
# ---------------------------------------------------------------------------


def merge_exact_answers(
    qtype: str,
    runs: Sequence[RunResults],
    qid: str,
    best_run_idx: int,
) -> ExactAnswer:
    """Merge exact answers from multiple runs using type-specific strategies.

    Parameters
    ----------
    qtype:
        Question type (``yesno``, ``factoid``, ``list``, ``summary``).
    runs:
        List of run result dicts.
    qid:
        Question ID.
    best_run_idx:
        Index of the tiebreaker run.
    """
    if qtype == "summary":
        return None

    n: int = len(runs)
    best_entry: RunResultEntry = runs[best_run_idx].get(qid, {})
    best_exact: ExactAnswer = best_entry.get("exact_answer")  # type: ignore[assignment]

    if qtype == "yesno":
        answers: list[str] = []
        for r in runs:
            entry: RunResultEntry = r.get(qid, {})
            a: str | ExactAnswer = entry.get("exact_answer", "")
            if isinstance(a, str) and a in ("yes", "no"):
                answers.append(a)
        if not answers:
            return best_exact
        count: Counter[str] = Counter(answers)
        if count["yes"] != count["no"]:
            return count.most_common(1)[0][0]
        return best_exact

    elif qtype == "factoid":
        all_candidates: list[str] = []
        for r in runs:
            entry = r.get(qid, {})
            ans: str | ExactAnswer = entry.get("exact_answer")  # type: ignore[assignment]
            if ans is None:
                continue
            if isinstance(ans, str):
                all_candidates.append(ans)
            elif isinstance(ans, list):
                for item in ans:
                    if isinstance(item, str):
                        all_candidates.append(item)
                    elif isinstance(item, list):
                        for sub in item:
                            all_candidates.append(str(sub))

        if not all_candidates:
            return best_exact

        norm_to_original: dict[str, str] = {}
        counts: Counter[str] = Counter()
        for cand in all_candidates:
            key: str = _normalize(str(cand))
            if key not in norm_to_original:
                norm_to_original[key] = cand
            counts[key] += 1

        top: list[tuple[str, int]] = counts.most_common()
        if top[0][1] > 1 or n == 1:
            return [norm_to_original[k] for k, _ in top[:5]]
        return best_exact

    elif qtype == "list":
        threshold: int = math.ceil(n / 2)
        norm_to_original_list: dict[str, str | list[str]] = {}
        counts = Counter()

        for r in runs:
            entry = r.get(qid, {})
            ans = entry.get("exact_answer")
            if ans is None:
                continue
            if isinstance(ans, str):
                ans = [ans]
            if isinstance(ans, list):
                for entity in ans:
                    text: str
                    if isinstance(entity, list) and entity:
                        text = entity[0]
                    else:
                        text = str(entity)
                    key = _normalize(text)
                    if key not in norm_to_original_list:
                        norm_to_original_list[key] = entity  # type: ignore[assignment]
                    counts[key] += 1

        merged: list[str | list[str]] = [
            norm_to_original_list[k] for k, c in counts.items() if c >= threshold  # type: ignore[misc]
        ]
        return merged if merged else best_exact  # type: ignore[return-value]

    return best_exact


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def parse_synthesis_output(text: str) -> tuple[bool, str | None]:
    """Extract ``ideal_answer`` from the model's output."""
    matches: list[str] = re.findall(r"\{.*?\}", text, re.DOTALL)
    if matches:
        try:
            parsed: dict[str, str] = json.loads(matches[-1], strict=False)
            ideal: str | None = parsed.get("ideal_answer")
            if ideal:
                return True, str(ideal)
        except json.JSONDecodeError:
            pass
    return False, None


# ---------------------------------------------------------------------------
# Core synthesis function
# ---------------------------------------------------------------------------


def run_synthesis(
    runs: Sequence[RunResults],
    question_text: Mapping[str, str],
    question_type: Mapping[str, str],
    backend: BaseModelBackend,
    prompts_templates: PromptTemplates,
    selected_prompt_ids: Sequence[str] = ("1",),
    best_run_idx: int = 0,
) -> dict[str, dict[str, SynthesisResult]]:
    """Synthesise answers from multiple runs via LLM.

    Parameters
    ----------
    runs:
        List of run result dicts.
    question_text:
        ``{qid: body_text}``.
    question_type:
        ``{qid: "yesno"|"factoid"|"list"|"summary"}``.
    backend:
        Loaded LLM backend.
    prompts_templates:
        Prompt templates loaded via :func:`load_prompts`.
    selected_prompt_ids:
        Prompt IDs to grid-search over.
    best_run_idx:
        Index of the run to use as tiebreaker for exact answers.

    Returns
    -------
    ``{prompt_id: {qid: SynthesisResult}}``
    """
    all_qids: list[str] = list(runs[0].keys())
    results: dict[str, dict[str, SynthesisResult]] = {}

    for pid in selected_prompt_ids:
        if pid not in prompts_templates:
            print(f"Warning: prompt id '{pid}' not found, skipping.")
            continue

        template: str = prompts_templates[pid]["template"]

        prompt_list: list[str] = []
        meta_list: list[str] = []

        for qid in all_qids:
            question: str = question_text.get(qid, "")
            answers_block: str = build_answers_block(runs, qid)
            prompt_list.append(template.format(question=question, answers=answers_block))
            meta_list.append(qid)

        responses: list[str] = backend.generate_batch(prompt_list)

        pid_results: dict[str, SynthesisResult] = {}
        for raw_text, qid in zip(responses, meta_list, strict=True):
            valid, ideal_answer = parse_synthesis_output(raw_text)

            if not valid or not ideal_answer:
                fallback: RunResultEntry = runs[0].get(qid, {})
                ideal_answer = str(fallback.get("ideal_answer", ""))

            qtype: str = question_type.get(qid, "summary")
            exact_answer: ExactAnswer = merge_exact_answers(
                qtype, runs, qid, best_run_idx
            )

            pid_results[qid] = SynthesisResult(
                ideal_answer=ideal_answer or "",
                exact_answer=exact_answer,
                valid=valid,
                raw=raw_text,
            )

        results[pid] = pid_results

    return results
