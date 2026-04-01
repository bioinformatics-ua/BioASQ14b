"""Type definitions for the agent quorum debate system."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

# Agreement levels — ordered from least to most consensus.
AgreementLevel = Literal["strongly_disagree", "disagree", "agree", "strongly_agree"]

AGREEMENT_RANK: dict[AgreementLevel, int] = {
    "strongly_disagree": 0,
    "disagree": 1,
    "agree": 2,
    "strongly_agree": 3,
}


class DebateTurn(TypedDict):
    """A single turn by one agent in the debate."""

    round: int
    agent_id: int
    agent_focus: str
    model: str
    opinion: str
    agreement: AgreementLevel
    documents_shown: list[int]
    kept_documents: list[int]


class QuorumResult(TypedDict):
    """Final output of a completed quorum debate."""

    ideal_answer: str
    exact_answer: str | list[str] | list[list[str]] | None
    debate: list[DebateTurn]
    rounds: int
    consensus_reached: bool
    total_docs: int
    docs_per_sample: int
    question_id: str
    question_type: str


class AgentConfig(TypedDict):
    """Configuration for a single agent."""

    agent_id: int
    focus: str
    model: str


class QuorumConfig(TypedDict, total=False):
    """Top-level configuration passed to the Debate runner."""

    num_agents: int
    models: list[str]
    max_rounds: int
    max_docs: int
    docs_per_sample: int
    request_delay: float
    max_tokens: int
    temperature: float


# Mapping from question type to the expected exact_answer shape.
ExactAnswerShape = Literal["yesno", "factoid", "list", "none"]

QUESTION_TYPE_SHAPE: dict[str, ExactAnswerShape] = {
    "yesno": "yesno",
    "factoid": "factoid",
    "list": "list",
    "summary": "none",
}


class ParsedAgentResponse(TypedDict):
    """Structured output parsed from an agent's raw LLM response."""

    opinion: str
    agreement: AgreementLevel
    kept_documents: list[int]
    raw: str


class ParsedFinalAnswer(TypedDict):
    """Structured output parsed from the final answer LLM response."""

    ideal_answer: str
    exact_answer: Any
    raw: str
