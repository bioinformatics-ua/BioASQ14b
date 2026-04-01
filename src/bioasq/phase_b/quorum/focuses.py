"""Agent thinking-focus definitions for the quorum debate.

Each focus represents a distinct cognitive lens — not a persona or character,
but a disciplined mode of reasoning applied to biomedical question answering.
"""

from __future__ import annotations

from typing import NamedTuple


class Focus(NamedTuple):
    name: str
    description: str


# Pool of available thinking focuses.
FOCUSES: list[Focus] = [
    Focus(
        name="analytical",
        description=(
            "Approach this by systematically decomposing the evidence. "
            "Identify premises, logical chains, and conclusions. "
            "Prioritize internal consistency and logical validity. "
            "Flag any unjustified inferential leaps."
        ),
    ),
    Focus(
        name="evidence-based",
        description=(
            "Ground every claim strictly in what the provided documents explicitly state. "
            "Avoid speculation beyond the text. "
            "Highlight which specific passages support or undermine the proposed answer. "
            "Note any direct contradictions between sources."
        ),
    ),
    Focus(
        name="skeptical",
        description=(
            "Challenge the direction the debate is taking. "
            "Seek gaps in the evidence, alternative interpretations, confounding factors, "
            "and methodological limitations. "
            "Ask: what assumptions are being made, and what would falsify this answer? "
            "However, don't be a negationist — if the evidence is strong, "
            "acknowledge and agree with it, but explain what would be needed to reach agreement."
        ),
    ),
    Focus(
        name="integrative",
        description=(
            "Synthesize information across all available sources. "
            "Seek converging evidence and reconcile apparent contradictions. "
            "Identify the most robust conclusion that is consistent with the full body of evidence."
        ),
    ),
    Focus(
        name="pragmatic",
        description=(
            "Focus on the most direct, actionable answer to the question as posed. "
            "Prioritize precision and clarity. "
            "Avoid over-qualification. "
            "What is the most defensible, concise answer given the evidence?"
        ),
    ),
    Focus(
        name="theoretical",
        description=(
            "Consider underlying biological mechanisms, biochemical pathways, and "
            "established scientific principles. "
            "How does the proposed answer fit with the broader theoretical framework of the field? "
            "Are there mechanistic reasons to accept or reject it?"
        ),
    ),
]

FOCUS_BY_NAME: dict[str, Focus] = {f.name: f for f in FOCUSES}


def assign_focuses(num_agents: int) -> list[Focus]:
    """Assign thinking focuses to ``num_agents`` agents.

    Focuses are assigned round-robin from the pool so that:
    - If ``num_agents <= len(FOCUSES)``, each agent gets a unique focus.
    - If ``num_agents > len(FOCUSES)``, focuses repeat cyclically.
    """
    return [FOCUSES[i % len(FOCUSES)] for i in range(num_agents)]
