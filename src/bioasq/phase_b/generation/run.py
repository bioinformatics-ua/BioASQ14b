"""Phase B inference runner — generate answers for BioASQ questions.

Loads a BioASQ JSONL file, builds all prompt combinations
(num_support x prompt_ids) into a single batch, runs inference once,
and writes one output file per combination.

Refactored from ``phaseB/inference/run.py``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from bioasq.common.io import load_json
from bioasq.common.types import GeneratedAnswer

if TYPE_CHECKING:
    from pathlib import Path

    from bioasq.common.protocols import BaseModelBackend

# Type for prompt template dictionaries
type PromptTemplate = dict[str, str]
type PromptTemplates = dict[str, Any]

# Type for question dicts (from JSONL data)
type QuestionRecord = Mapping[str, str | list[str] | list[dict[str, str]]]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def load_prompts(prompts_path: str | Path) -> PromptTemplates:
    """Load prompt templates from a JSON file."""
    return load_json(prompts_path)  # type: ignore[return-value]


def _extract_snippet_texts(question: QuestionRecord, num_support: int) -> list[str]:
    raw_snippets: str | list[str] | list[dict[str, str]] = question.get("snippets", [])
    if isinstance(raw_snippets, list):
        return [str(s) for s in raw_snippets[:num_support]]
    return [str(raw_snippets)]


def _extract_snippets_with_thinking(question: QuestionRecord, num_support: int) -> list[str]:
    """Extract snippet texts with their thinking/rationale appended."""
    raw_snippets = question.get("snippets", [])
    if not isinstance(raw_snippets, list):
        return [str(raw_snippets)]

    items: list[str] = []
    for s in raw_snippets[:num_support]:
        if isinstance(s, dict):
            text = str(s.get("text", s))
            thinking = s.get("thinking", "")
            if thinking:
                items.append(f"{text}\nrelevance: {thinking}")
            else:
                items.append(text)
        else:
            items.append(str(s))
    return items


def _parse_document_list(raw_docs: list[Any], num_support: int) -> list[str]:
    doc_items: list[str] = []
    for d in raw_docs[:num_support]:
        if isinstance(d, dict):
            doc_items.append(str(d.get("text", "")))
            continue
        doc_items.append(str(d))
    return doc_items


def _extract_document_texts(question: QuestionRecord, num_support: int) -> list[str]:
    raw_docs: str | list[str] | list[dict[str, str]] = question.get("documents", [])
    if isinstance(raw_docs, list):
        return _parse_document_list(raw_docs, num_support)
    return [str(raw_docs)]


def build_context(
    question: QuestionRecord,
    input_type: str,
    num_support: int,
) -> str:
    """Build context block from a question's documents or snippets."""
    items: list[str] = []

    if input_type == "snippets":
        items = _extract_snippet_texts(question, num_support)
    elif input_type == "snippets_with_thinking":
        items = _extract_snippets_with_thinking(question, num_support)
    else:
        items = _extract_document_texts(question, num_support)

    if not items:
        return "(No context available)"

    label = "snippets" if input_type.startswith("snippets") else input_type
    return "\n\n".join(f"{label}: {x}" for x in items)


def _parse_json_from_text(text: str) -> dict[str, Any] | None:
    matches: list[str] = re.findall(r"\{.*?\}", text, re.DOTALL)
    if not matches:
        return None
    try:
        return json.loads(matches[-1], strict=False)
    except json.JSONDecodeError:
        return None


def parse_json_answer(text: str) -> tuple[bool, str]:
    """Extract ``answer`` from the model's output (last JSON object)."""
    parsed = _parse_json_from_text(text)
    if parsed is not None and "answer" in parsed:
        return True, str(parsed["answer"])
    return False, text


def _extract_list_answer(exact: Any) -> tuple[bool, list[list[str]] | list[str]]:  # noqa: ANN401
    if isinstance(exact, list) and exact and not isinstance(exact[0], list):
        return True, [[item] for item in exact]
    return True, exact


def parse_exact(text: str, qtype: str) -> tuple[bool, list[str] | list[list[str]] | str | None]:
    """Find the last JSON block and extract exact_answer based on question type."""
    parsed = _parse_json_from_text(text)
    if parsed is None:
        return False, None

    exact = parsed.get("exact_answer")
    if exact is None:
        return False, None

    if qtype == "yesno" and isinstance(exact, str) and exact.lower() in ("yes", "no"):
        return True, exact.lower()

    if qtype == "factoid":
        if isinstance(exact, str):
            return True, [exact]
        return True, exact

    if qtype == "list":
        return _extract_list_answer(exact)

    return False, None


def parse_summary_ideal(text: str) -> tuple[bool, str]:
    """Extract ideal_answer from JSON (for summary questions in exact mode)."""
    parsed = _parse_json_from_text(text)
    if parsed is None:
        return False, ""

    ideal = parsed.get("ideal_answer")
    if isinstance(ideal, str) and ideal.strip():
        return True, ideal.strip()

    return False, ""


def _resolve_prompts_for_question(
    q_type: str, prompts_templates: PromptTemplates
) -> PromptTemplates:
    if any(k in prompts_templates for k in ("yesno", "factoid", "list", "summary")):
        return prompts_templates.get(q_type, {})
    return prompts_templates


def _get_template_string(p_item: Any) -> str:  # noqa: ANN401
    if isinstance(p_item, dict):
        return str(p_item["template"])
    return str(p_item)


def _build_prompts_for_question(
    q: QuestionRecord,
    prompts_templates: PromptTemplates,
    input_type: str,
    selected_counts: Sequence[int],
    selected_prompts: Sequence[str],
) -> tuple[list[str], list[tuple[str, int, str, str]]]:
    q_id: str = str(q.get("id", ""))
    q_body: str = str(q.get("body", ""))
    q_type: str = str(q.get("type", "summary"))

    q_prompts = _resolve_prompts_for_question(q_type, prompts_templates)

    prompt_list: list[str] = []
    prompt_info: list[tuple[str, int, str, str]] = []

    for n in selected_counts:
        context: str = build_context(q, input_type, n)
        for pid in selected_prompts:
            if pid not in q_prompts:
                continue

            template: str = _get_template_string(q_prompts[pid])
            prompt_list.append(
                template.format(
                    d_type=input_type,
                    context=context,
                    question=q_body,
                )
            )
            prompt_info.append((q_id, n, pid, q_type))

    return prompt_list, prompt_info


def _parse_exact_answer(raw: str, qtype: str) -> GeneratedAnswer:
    if qtype == "summary":
        valid, ideal = parse_summary_ideal(raw)
        return GeneratedAnswer(
            text=ideal if valid else "",
            exact_answer="",
            valid=valid,
            raw=raw,
        )

    valid, exact = parse_exact(raw, qtype)
    return GeneratedAnswer(
        text="",
        exact_answer=exact,
        valid=valid,
        raw=raw,
    )


def _parse_regular_answer(raw: str) -> GeneratedAnswer:
    valid, text = parse_json_answer(raw)
    return GeneratedAnswer(
        text=text,
        exact_answer=None,
        valid=valid,
        raw=raw,
    )


def _process_response(raw: str, qtype: str, extract_exact: bool) -> GeneratedAnswer:
    if extract_exact:
        return _parse_exact_answer(raw, qtype)
    return _parse_regular_answer(raw)


def _initialize_answer_dict(
    selected_counts: Sequence[int], selected_prompts: Sequence[str]
) -> dict[int, dict[str, dict[str, GeneratedAnswer]]]:
    return {n: {pid: {} for pid in selected_prompts} for n in selected_counts}


def run_generation(
    questions: Sequence[QuestionRecord],
    backend: BaseModelBackend,
    prompts_templates: PromptTemplates,
    input_type: str = "abstracts",
    selected_counts: Sequence[int] = (5,),
    selected_prompts: Sequence[str] = ("1",),
    extract_exact: bool = False,
) -> dict[int, dict[str, dict[str, GeneratedAnswer]]]:
    """Run generation for all (question x num_support x prompt_id) combos."""
    prompt_list: list[str] = []
    prompt_info: list[tuple[str, int, str, str]] = []

    for q in questions:
        pl, pi = _build_prompts_for_question(
            q, prompts_templates, input_type, selected_counts, selected_prompts
        )
        prompt_list.extend(pl)
        prompt_info.extend(pi)

    answer_dict = _initialize_answer_dict(selected_counts, selected_prompts)

    if not prompt_list:
        return answer_dict

    responses: list[str] = backend.generate_batch(prompt_list)

    for raw, (qid, n, pid, qtype) in zip(responses, prompt_info, strict=True):
        ans = _process_response(raw, qtype, extract_exact)
        answer_dict[n][pid][qid] = ans

    return answer_dict
