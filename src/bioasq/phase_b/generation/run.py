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
from typing import TYPE_CHECKING

from bioasq.common.io import load_json
from bioasq.common.types import GeneratedAnswer

if TYPE_CHECKING:
    from pathlib import Path

    from bioasq.common.protocols import BaseModelBackend

# Type for prompt template dictionaries
type PromptTemplate = dict[str, str]
type PromptTemplates = dict[str, PromptTemplate]

# Type for question dicts (from JSONL data)
type QuestionRecord = Mapping[str, str | list[str] | list[dict[str, str]]]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def load_prompts(prompts_path: str | Path) -> PromptTemplates:
    """Load prompt templates from a JSON file."""
    return load_json(prompts_path)  # type: ignore[return-value]


def build_context(
    question: QuestionRecord,
    input_type: str,
    num_support: int,
) -> str:
    """Build context block from a question's documents or snippets.

    Parameters
    ----------
    question:
        BioASQ question dict with ``snippets`` and/or ``documents``.
    input_type:
        ``"snippets"`` or ``"abstracts"``.
    num_support:
        Number of supporting items to include.
    """
    items: list[str]
    if input_type == "snippets":
        raw_snippets: str | list[str] | list[dict[str, str]] = question.get("snippets", [])
        if isinstance(raw_snippets, list):
            items = [str(s) for s in raw_snippets[:num_support]]
        else:
            items = [str(raw_snippets)]
    else:
        raw_docs: str | list[str] | list[dict[str, str]] = question.get("documents", [])
        if isinstance(raw_docs, list):
            doc_items: list[str] = []
            for d in raw_docs[:num_support]:
                if isinstance(d, dict):
                    doc_items.append(d.get("text", ""))
                else:
                    doc_items.append(str(d))
            items = doc_items
        else:
            items = [str(raw_docs)]

    if not items:
        return "(No context available)"

    return "\n\n".join(f"{input_type}: {x}" for x in items)


def parse_json_answer(text: str) -> tuple[bool, str]:
    """Extract ``answer`` from the model's output (last JSON object)."""
    matches: list[str] = re.findall(r"\{.*?\}", text, re.DOTALL)
    if matches:
        try:
            parsed: dict[str, str] = json.loads(matches[-1], strict=False)
            if "answer" in parsed:
                return True, parsed["answer"]
        except json.JSONDecodeError:
            pass
    return False, text


def run_generation(
    questions: Sequence[QuestionRecord],
    backend: BaseModelBackend,
    prompts_templates: PromptTemplates,
    input_type: str = "abstracts",
    selected_counts: Sequence[int] = (5,),
    selected_prompts: Sequence[str] = ("1",),
) -> dict[int, dict[str, dict[str, GeneratedAnswer]]]:
    """Run generation for all (question x num_support x prompt_id) combos.

    Returns
    -------
    ``{num_support: {prompt_id: {qid: GeneratedAnswer}}}``
    """
    prompt_list: list[str] = []
    prompt_info: list[tuple[str, int, str]] = []

    for q in questions:
        q_id: str = str(q["id"]) if "id" in q else ""
        q_body: str = str(q.get("body", ""))
        for n in selected_counts:
            context: str = build_context(q, input_type, n)
            for pid in selected_prompts:
                if pid not in prompts_templates:
                    msg: str = f"Prompt id '{pid}' not found."
                    raise ValueError(msg)
                template: str = prompts_templates[pid]["template"]
                prompt_list.append(
                    template.format(
                        d_type=input_type,
                        context=context,
                        question=q_body,
                    )
                )
                prompt_info.append((q_id, n, pid))

    responses: list[str] = backend.generate_batch(prompt_list)

    answer_dict: dict[int, dict[str, dict[str, GeneratedAnswer]]] = {
        n: {pid: {} for pid in selected_prompts} for n in selected_counts
    }

    for raw, (qid, n, pid) in zip(responses, prompt_info, strict=True):
        valid, text = parse_json_answer(raw)
        answer_dict[n][pid][qid] = GeneratedAnswer(text=text, valid=valid, raw=raw)

    return answer_dict
