"""Prompt builders for the quorum debate and final answer synthesis."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bioasq.phase_b.quorum._types import AgreementLevel, DebateTurn

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a rigorous biomedical scientist participating in a structured expert debate.
Your goal is to help reach the most accurate, evidence-grounded answer to a \
biomedical question.
You respond exclusively with valid JSON — no prose outside the JSON object.\
"""

# ---------------------------------------------------------------------------
# Debate turn template
# ---------------------------------------------------------------------------

_DEBATE_TURN_TEMPLATE = """\
## Biomedical Expert Debate

**Question:** {question}
**Question type:** {question_type}

---

### Context documents ({num_docs} injected so far)

{context}

---

### Conversation history

{history}

---

### Your thinking approach

{focus_description}

---

### Your task

1. Reason about the answer to the question using your thinking approach.
2. If other agents have already contributed, engage with their reasoning — \
reinforce what is sound, correct what is wrong, and fill gaps.
3. Decide whether you need additional documents to form a confident opinion.
4. Rate how much you agree with the current direction of the debate.

**Agreement scale:**
- `"strongly_agree"` — fully confident; the debate is converging on the correct, \
complete answer.
- `"agree"` — the direction is right, but some refinement remains.
- `"disagree"` — important aspects are missing or the direction needs correction.
- `"strongly_disagree"` — the current direction is wrong and needs significant revision.

**Respond with a single JSON object — no text outside it:**

```json
{{
  "opinion": "<your detailed reasoning and position>",
  "request_more_context": <true | false>,
  "agreement": "<strongly_disagree | disagree | agree | strongly_agree>"
}}
```\
"""

# ---------------------------------------------------------------------------
# Final answer templates per question type
# ---------------------------------------------------------------------------

_FINAL_ANSWER_BASE = """\
## Final Answer Synthesis

Based on a structured expert debate, produce the definitive answer to the \
following biomedical question.

**Question:** {question}
**Question type:** {question_type}

---

### Context documents

{context}

---

### Debate summary

{debate_summary}

---

### Instructions

- Answer the question directly and concisely.
- Do **not** reference the debate, the agents, or the deliberation process.
- Base your answer solely on the evidence in the context documents, \
informed by the reasoning developed in the debate.
{type_instructions}

**Respond with a single JSON object — no text outside it:**

```json
{format_spec}
```\
"""

_TYPE_INSTRUCTIONS: dict[str, str] = {
    "yesno": (
        '- `exact_answer` must be exactly `"yes"` or `"no"`.\n'
        "- `ideal_answer` is a concise explanatory paragraph (1–3 sentences)."
    ),
    "factoid": (
        "- `exact_answer` is a JSON array of candidate entity strings "
        '(most likely first), e.g. `["BRCA1", "BRCA2"]`.\n'
        "- `ideal_answer` is a concise explanatory sentence or two."
    ),
    "list": (
        "- `exact_answer` is a JSON array of arrays, one inner array per item, "
        'e.g. `[["interleukin-6"], ["TNF-alpha"]]`.\n'
        "- `ideal_answer` is a concise paragraph naming and briefly explaining the items."
    ),
    "summary": (
        "- There is no `exact_answer` for summary questions.\n"
        "- `ideal_answer` is a thorough, well-structured paragraph summarising the evidence."
    ),
}

_FORMAT_SPECS: dict[str, str] = {
    "yesno": '{\n  "ideal_answer": "<explanatory paragraph>",\n  "exact_answer": "yes"\n}',
    "factoid": '{\n  "ideal_answer": "<explanatory sentence>",\n  "exact_answer": ["entity1", "entity2"]\n}',
    "list": '{\n  "ideal_answer": "<explanatory paragraph>",\n  "exact_answer": [["item1"], ["item2"]]\n}',
    "summary": '{\n  "ideal_answer": "<comprehensive paragraph>"\n}',
}


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def _format_context(documents: list[str]) -> str:
    if not documents:
        return "(No context documents available yet.)"
    parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        parts.append(f"**Document {i}:**\n{doc.strip()}")
    return "\n\n".join(parts)


def _format_turn_full(turn: DebateTurn) -> str:
    return (
        f"**[Round {turn['round']}, Agent {turn['agent_id']} "
        f"— {turn['agent_focus']}]**\n"
        f"{turn['opinion']}\n"
        f"*Agreement: {turn['agreement']}*"
    )


def _format_turn_summary(turn: DebateTurn) -> str:
    opinion_preview = turn["opinion"][:120].rstrip()
    if len(turn["opinion"]) > 120:
        opinion_preview += "…"
    return (
        f"[Round {turn['round']}, Agent {turn['agent_id']} "
        f"— {turn['agent_focus']}] "
        f"{opinion_preview} (agreement: {turn['agreement']})"
    )


def _format_history(
    turns: list[DebateTurn],
    max_full_turns: int | None = None,
) -> str:
    """Format the conversation history.

    If *max_full_turns* is set and there are more turns than that, older
    turns beyond the window are compressed into a one-line summary each
    to keep the prompt within context-window bounds.
    """
    if not turns:
        return "(No contributions yet — you are the first to speak.)"

    if max_full_turns is None or len(turns) <= max_full_turns:
        return "\n\n---\n\n".join(_format_turn_full(t) for t in turns)

    cutoff = len(turns) - max_full_turns
    summarised = [_format_turn_summary(t) for t in turns[:cutoff]]
    full = [_format_turn_full(t) for t in turns[cutoff:]]
    header = "**Earlier turns (summarised):**\n" + "\n".join(summarised)
    return header + "\n\n---\n\n" + "\n\n---\n\n".join(full)


def build_debate_turn_messages(
    question: str,
    question_type: str,
    injected_docs: list[str],
    history: list[DebateTurn],
    focus_description: str,
    max_history_turns: int | None = None,
) -> list[dict[str, str]]:
    """Build the chat message list for one agent debate turn."""
    user_content = _DEBATE_TURN_TEMPLATE.format(
        question=question,
        question_type=question_type,
        num_docs=len(injected_docs),
        context=_format_context(injected_docs),
        history=_format_history(history, max_full_turns=max_history_turns),
        focus_description=focus_description,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_final_answer_messages(
    question: str,
    question_type: str,
    injected_docs: list[str],
    turns: list[DebateTurn],
) -> list[dict[str, str]]:
    """Build the chat message list for final answer synthesis."""
    debate_summary = _summarise_debate(turns)
    user_content = _FINAL_ANSWER_BASE.format(
        question=question,
        question_type=question_type,
        context=_format_context(injected_docs),
        debate_summary=debate_summary,
        type_instructions=_TYPE_INSTRUCTIONS.get(question_type, ""),
        format_spec=_FORMAT_SPECS.get(question_type, _FORMAT_SPECS["summary"]),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _summarise_debate(turns: list[DebateTurn]) -> str:
    """Produce a concise, structured summary of all debate turns."""
    if not turns:
        return "(No debate turns recorded.)"

    by_round: dict[int, list[DebateTurn]] = {}
    for turn in turns:
        by_round.setdefault(turn["round"], []).append(turn)

    parts: list[str] = []
    for rnd in sorted(by_round):
        round_parts: list[str] = [f"**Round {rnd}**"]
        for turn in by_round[rnd]:
            round_parts.append(
                f"- Agent {turn['agent_id']} ({turn['agent_focus']}): "
                f"{turn['opinion']}  \n"
                f"  *Agreement: {turn['agreement']}*"
            )
        parts.append("\n".join(round_parts))

    return "\n\n".join(parts)


def format_agreement_label(level: AgreementLevel) -> str:
    labels = {
        "strongly_disagree": "Strongly Disagree",
        "disagree": "Disagree",
        "agree": "Agree",
        "strongly_agree": "Strongly Agree",
    }
    return labels.get(level, level)
