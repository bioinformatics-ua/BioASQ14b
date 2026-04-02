"""Prompt builders for the quorum debate and final answer synthesis."""

from bioasq.phase_b.quorum._types import (
    AgreementLevel,
    DebateTurn,
    QuorumDocument,
)

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

### Your document sample ({num_shown} of {total_docs} total documents)

You are viewing a **subset** of the available evidence documents. Other agents \
may have seen different subsets. Documents are identified by a stable ID \
(e.g. Document 3) that is consistent across all agents and rounds.

Documents you list in `kept_documents` will be guaranteed to appear in your \
next round. Remaining slots will be filled with other documents from the pool \
(possibly ones you have seen before). If you keep **all** {num_shown} documents, \
you will receive one additional document slot next round.

{context}

---

### Conversation history

{history}

---

### Your thinking approach

{focus_description}

---

### Your task

1. Reason carefully about the answer using your thinking approach and the \
documents available to you.
2. **Engage critically with other agents.** If they cite evidence you have not \
seen, weigh their claims against your own documents. If you disagree, explain \
what specific evidence or reasoning undermines their position. If you agree, \
strengthen the argument with additional evidence from your documents.
3. Decide which documents are most relevant and should be retained for the next \
round. You may keep **zero or more** of the documents shown to you.
4. Rate your agreement with the emerging consensus.

**Agreement scale:**
- `"strongly_agree"` — fully confident; the answer is correct and complete.
- `"agree"` — the direction is right, but some refinement remains.
- `"disagree"` — important aspects are missing or the direction needs correction.
- `"strongly_disagree"` — the current direction is wrong and needs significant revision.

**Only output `"strongly_agree"` when you are genuinely confident the answer is \
correct, complete, and well-supported by evidence. Premature agreement weakens \
the final answer.**

**Respond with a single JSON object — no text outside it:**

```json
{{
  "opinion": "<your detailed reasoning and position>",
  "kept_documents": [<IDs of documents to retain, e.g. 3, 7>],
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
        "- `ideal_answer` is a concise explanatory paragraph (1-3 sentences)."
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
    "factoid": (
        '{\n  "ideal_answer": "<explanatory sentence>",\n'
        '  "exact_answer": ["entity1", "entity2"]\n}'
    ),
    "list": (
        '{\n  "ideal_answer": "<explanatory paragraph>",\n'
        '  "exact_answer": [["item1"], ["item2"]]\n}'
    ),
    "summary": '{\n  "ideal_answer": "<comprehensive paragraph>"\n}',
}


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def _format_context(documents: list[tuple[int, QuorumDocument]]) -> str:
    """Format documents as numbered blocks with stable 1-based IDs."""
    if not documents:
        return "(No documents in your current sample.)"

    parts: list[str] = []
    for doc_id, doc in documents:
        header = f"**Document {doc_id}:**"

        block_parts = [header, f"Text: {doc['text'].strip()}"]
        if doc["snippets"]:
            formatted_snippets = "\n".join(
                f"Snippet {index} - {snippet}"
                for index, snippet in enumerate(doc["snippets"], start=1)
            )
            block_parts.append(f"Attached snippets:\n{formatted_snippets}")

        parts.append("\n".join(block_parts))
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
    indexed_docs: list[tuple[int, QuorumDocument]],
    total_docs: int,
    history: list[DebateTurn],
    focus_description: str,
    max_history_turns: int | None = None,
) -> list[dict[str, str]]:
    """Build the chat message list for one agent debate turn."""
    user_content = _DEBATE_TURN_TEMPLATE.format(
        question=question,
        question_type=question_type,
        num_shown=len(indexed_docs),
        total_docs=total_docs,
        context=_format_context(indexed_docs),
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
    all_documents: list[QuorumDocument],
    turns: list[DebateTurn],
) -> list[dict[str, str]]:
    """Build the chat message list for final answer synthesis."""
    # Synthesis sees all documents with their stable 1-based IDs.
    indexed = [(i + 1, doc) for i, doc in enumerate(all_documents)]
    debate_summary = _summarise_debate(turns)
    user_content = _FINAL_ANSWER_BASE.format(
        question=question,
        question_type=question_type,
        context=_format_context(indexed),
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
