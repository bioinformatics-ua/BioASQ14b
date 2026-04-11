"""Agent definition and model-assignment logic for the quorum debate."""

import concurrent.futures
import contextlib
import time
from dataclasses import dataclass, field
from typing import cast

from bioasq.phase_b.backends.base import BaseModelBackend
from bioasq.phase_b.quorum._types import (
    AGREEMENT_RANK,
    AgreementLevel,
    ParsedAgentResponse,
)
from bioasq.phase_b.quorum.focuses import Focus, assign_focuses
from bioasq.phase_b.quorum.parsing import extract_last_json

# ---------------------------------------------------------------------------
# Hallucination-detection helpers
# ---------------------------------------------------------------------------


def _is_repetitive(
    text: str,
    ngram_size: int = 6,
    threshold: float = 0.15,
) -> bool:
    """Return True if *text* contains excessive repeated n-grams.

    A model that has entered a repetition loop (a common hallucination
    pattern) will produce the same word sequences over and over.  We
    detect this by computing word-level n-grams and checking whether
    the most frequent one occupies more than *threshold* of all n-gram
    positions.

    Parameters
    ----------
    text:
        The raw LLM response string.
    ngram_size:
        Number of consecutive words that form one n-gram (default 6).
    threshold:
        Fraction of n-gram positions that a single n-gram must exceed
        to be considered repetitive (default 0.15 = 15%).
    """
    words = text.split()
    min_words = ngram_size * 4
    if len(words) < min_words:
        return False

    ngrams = [tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
    if not ngrams:
        return False

    counts: dict[tuple, int] = {}
    for ng in ngrams:
        counts[ng] = counts.get(ng, 0) + 1

    max_count = max(counts.values())
    return (max_count / len(ngrams)) > threshold


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
    generation_timeout: float = 120.0
    """Seconds allowed per generation call.  0 disables the timeout."""
    repetition_threshold: float = 0.15
    """Max fraction of identical n-grams before a response is discarded."""
    _participated: bool = field(default=False, repr=False)

    @property
    def has_participated(self) -> bool:
        return self._participated

    def generate(self, messages: list[dict[str, str]]) -> str | None:
        """Generate with exponential-backoff retry on transient failures.

        Returns ``None`` (treated as invalid JSON by the debate loop) when:
        - all retries are exhausted,
        - the call exceeds *generation_timeout* seconds (stuck / hallucinating), or
        - the response is flagged as repetitive.
        """
        for attempt in range(self.max_retries):
            try:
                raw = self._call_with_timeout(messages)
            except concurrent.futures.TimeoutError:
                print(
                    f"  [timeout] Agent {self.agent_id} timed out after "
                    f"{self.generation_timeout:.0f}s — discarding as hallucination.",
                    flush=True,
                )
                return None  # do not retry; model is likely stuck
            except Exception as exc:
                wait = 2**attempt
                print(
                    f"  [retry] Agent {self.agent_id} attempt {attempt + 1}/{self.max_retries} "
                    f"failed ({exc!r}), waiting {wait}s…",
                    flush=True,
                )
                time.sleep(wait)
                continue

            if raw is not None and _is_repetitive(raw, threshold=self.repetition_threshold):
                print(
                    f"  [repetition] Agent {self.agent_id} response flagged as repetitive "
                    f"— discarding as hallucination.",
                    flush=True,
                )
                return None

            return raw
        return None

    def _call_with_timeout(self, messages: list[dict[str, str]]) -> str | None:
        """Invoke the backend with an optional wall-clock timeout.

        Raises ``concurrent.futures.TimeoutError`` if *generation_timeout* > 0
        and the call exceeds that many seconds.
        """
        if self.generation_timeout <= 0:
            return self.backend.generate_chat(messages)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.backend.generate_chat, messages)
            # Raises concurrent.futures.TimeoutError on expiry.
            return future.result(timeout=self.generation_timeout)

    def mark_participated(self) -> None:
        self._participated = True


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_VALID_AGREEMENT: set[str] = set(AGREEMENT_RANK.keys())
_DEFAULT_AGREEMENT: AgreementLevel = "disagree"


def parse_agent_response(raw: str | None) -> ParsedAgentResponse:
    """Parse the JSON output from an agent's debate turn.

    Falls back to sane defaults when the model output is malformed so the
    debate loop is never broken by a single bad response.
    """
    if raw is None or not raw.strip():
        return ParsedAgentResponse(
            opinion="(no opinion provided)",
            agreement=_DEFAULT_AGREEMENT,
            kept_documents=[],
            raw=raw or "",
        )

    parsed = extract_last_json(raw)

    if parsed is None:
        print(
            "  [parse warning] Failed to parse JSON from agent response, "
            "falling back to defaults. Raw response was:\n"
            f"{raw}",
            flush=True,
        )
        return ParsedAgentResponse(
            opinion=raw.strip() or "(no opinion provided)",
            agreement=_DEFAULT_AGREEMENT,
            kept_documents=[],
            raw=raw,
        )

    opinion: str = str(parsed.get("opinion", raw.strip() or "(no opinion provided)"))

    raw_agreement: AgreementLevel = cast(
        "AgreementLevel", str(parsed.get("agreement", _DEFAULT_AGREEMENT)).lower().strip()
    )
    agreement = raw_agreement if raw_agreement in _VALID_AGREEMENT else _DEFAULT_AGREEMENT

    raw_kept = parsed.get("kept_documents", [])
    kept_documents: list[int] = []
    if isinstance(raw_kept, list):
        for item in raw_kept:
            with contextlib.suppress(TypeError, ValueError):
                kept_documents.append(int(item))

    return ParsedAgentResponse(
        opinion=opinion,
        agreement=agreement,
        kept_documents=kept_documents,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Factory — build a list of agents from config
# ---------------------------------------------------------------------------


def build_agents(
    num_agents: int,
    models: list[str],
    backends: list[BaseModelBackend],
    generation_timeout: float = 120.0,
    repetition_threshold: float = 0.15,
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
                generation_timeout=generation_timeout,
                repetition_threshold=repetition_threshold,
            )
        )

    return agents
