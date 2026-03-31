"""Agent definition and model-assignment logic for the quorum debate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bioasq.phase_b.quorum.focuses import Focus, assign_focuses
from bioasq.phase_b.quorum.parsing import extract_last_json
from bioasq.phase_b.quorum.types import (
    AGREEMENT_RANK,
    AgreementLevel,
    ParsedAgentResponse,
)

if TYPE_CHECKING:
    from bioasq.phase_b.backends.base import BaseModelBackend

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class Agent:
    """A single debate participant with a fixed thinking focus and LLM backend.

    Parameters
    ----------
    agent_id:
        Zero-based index used to identify this agent in the debate transcript.
    focus:
        The cognitive lens this agent applies throughout the debate.
    model:
        The OpenRouter (or other) model name used for generation.
    backend:
        A loaded :class:`BaseModelBackend` instance shared or dedicated.
    """

    agent_id: int
    focus: Focus
    model: str
    backend: BaseModelBackend
    max_retries: int = 3
    _participated: bool = field(default=False, repr=False)

    @property
    def has_participated(self) -> bool:
        return self._participated

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate with exponential-backoff retry on transient failures."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self.backend.generate_chat(messages)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = 2 ** attempt
                print(
                    f"  [retry] Agent {self.agent_id} attempt {attempt + 1}/{self.max_retries} "
                    f"failed ({exc!r}), waiting {wait}s…",
                    flush=True,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"Agent {self.agent_id} failed after {self.max_retries} attempts"
        ) from last_exc

    def mark_participated(self) -> None:
        self._participated = True


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_VALID_AGREEMENT: set[str] = set(AGREEMENT_RANK.keys())
_DEFAULT_AGREEMENT: AgreementLevel = "disagree"


def parse_agent_response(raw: str) -> ParsedAgentResponse:
    """Parse the JSON output from an agent's debate turn.

    Falls back to sane defaults when the model output is malformed so the
    debate loop is never broken by a single bad response.
    """
    parsed = extract_last_json(raw)

    if parsed is None:
        return ParsedAgentResponse(
            opinion=raw.strip() or "(no opinion provided)",
            agreement=_DEFAULT_AGREEMENT,
            request_more_context=False,
            raw=raw,
        )

    opinion: str = str(parsed.get("opinion", raw.strip() or "(no opinion provided)"))

    raw_agreement = str(parsed.get("agreement", _DEFAULT_AGREEMENT)).lower().strip()
    agreement: AgreementLevel = (
        raw_agreement if raw_agreement in _VALID_AGREEMENT else _DEFAULT_AGREEMENT  # type: ignore[assignment]
    )

    request_more_context: bool = bool(parsed.get("request_more_context", False))

    return ParsedAgentResponse(
        opinion=opinion,
        agreement=agreement,
        request_more_context=request_more_context,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Factory — build a list of agents from config
# ---------------------------------------------------------------------------


def build_agents(
    num_agents: int,
    models: list[str],
    backends: list[BaseModelBackend],
) -> list[Agent]:
    """Create ``num_agents`` agents, distributing models and focuses evenly.

    Model assignment:
      ``agent_i → models[ i % len(models) ]``

    This satisfies:
    - 4 agents, 1 model  → all agents share that model
    - 4 agents, 4 models → one agent per model
    - 4 agents, 2 models → two agents per model
    - 4 agents, 6 models → only the first 4 models are used

    The caller is responsible for constructing and loading the backends;
    ``backends[i]`` corresponds to ``models[i]``.

    Focus assignment:
      Round-robin from :data:`~bioasq.phase_b.quorum.focuses.FOCUSES`
      (up to 6 unique focuses, then repeating).
    """
    if len(backends) != len(models):
        msg = f"len(backends)={len(backends)} must equal len(models)={len(models)}"
        raise ValueError(msg)

    focuses = assign_focuses(num_agents)
    agents: list[Agent] = []

    for i in range(num_agents):
        model_idx = i % len(models)
        agents.append(
            Agent(
                agent_id=i,
                focus=focuses[i],
                model=models[model_idx],
                backend=backends[model_idx],
            )
        )

    return agents
