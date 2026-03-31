"""Agent quorum debate system for BioASQ Phase B answer generation.

The quorum orchestrates multiple agents — each with a distinct cognitive focus —
through a structured debate until they converge on an answer.

Public API
----------
Debate
    The core debate orchestrator.  Instantiate with a question, documents,
    and a list of :class:`Agent` instances, then call :meth:`Debate.run`.
build_agents
    Factory that constructs agents from a list of models and loaded backends.
QuorumResult
    TypedDict describing the full output of a debate run.
DebateTurn
    TypedDict describing a single agent contribution.
FOCUSES
    The pool of available thinking focuses (analytical, evidence-based, …).
"""

from bioasq.phase_b.quorum._types import (
    AgreementLevel,
    DebateTurn,
    QuorumConfig,
    QuorumResult,
)
from bioasq.phase_b.quorum.agent import Agent, build_agents
from bioasq.phase_b.quorum.debate import Debate
from bioasq.phase_b.quorum.focuses import FOCUSES, Focus

__all__ = [
    "FOCUSES",
    "Agent",
    "AgreementLevel",
    "Debate",
    "DebateTurn",
    "Focus",
    "QuorumConfig",
    "QuorumResult",
    "build_agents",
]
