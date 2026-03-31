"""Debate orchestrator for the agent quorum system.

Flow
----
1. One document is injected into the context to start the debate.
2. Agents participate sequentially in a randomly shuffled order within each round.
3. Each agent outputs an opinion, an agreement level, and optionally requests
   more context (triggering immediate injection of the next available document).
4. Each agent must participate at least once before the termination check applies.
5. The debate ends when **all** agents output ``"strongly_agree"`` in the same round.
6. A separate synthesis call produces the final clean answer with no reference
   to the debate itself.

The ``Debate`` class is a pure orchestrator — it does not own the backends and
assumes they are already loaded.
"""

import random
from typing import Any

from bioasq.phase_b.quorum import AgreementLevel
from bioasq.phase_b.quorum._types import (
    AGREEMENT_RANK,
    DebateTurn,
    ParsedFinalAnswer,
    QuorumResult,
)
from bioasq.phase_b.quorum.agent import Agent, parse_agent_response
from bioasq.phase_b.quorum.parsing import extract_last_json
from bioasq.phase_b.quorum.prompts import (
    build_debate_turn_messages,
    build_final_answer_messages,
)


class Debate:
    """Runs a multi-agent quorum debate for a single BioASQ question.

    Parameters
    ----------
    question_id:
        BioASQ question identifier.
    question_body:
        The question text.
    question_type:
        One of ``"yesno"``, ``"factoid"``, ``"list"``, ``"summary"``.
    documents:
        Ordered list of document texts (abstracts).  The first document is
        injected at the start; additional documents are added on agent request.
    agents:
        List of :class:`~bioasq.phase_b.quorum.agent.Agent` instances.
    max_rounds:
        Hard upper bound on the number of debate rounds (safety limit).
    max_docs:
        Maximum number of documents that may be injected over the course of the
        debate.  Defaults to all available documents.
    verbose:
        Print progress to stdout.
    rng:
        Optional :class:`random.Random` instance for reproducible shuffling.
    """

    def __init__(
        self,
        question_id: str,
        question_body: str,
        question_type: str,
        documents: list[str],
        agents: list[Agent],
        max_rounds: int = 10,
        max_docs: int | None = None,
        max_history_turns: int | None = None,
        stagnation_rounds: int = 3,
        verbose: bool = True,
        rng: random.Random | None = None,
    ) -> None:
        if not agents:
            msg = "At least one agent is required."
            raise ValueError(msg)
        if not documents:
            msg = "At least one document is required."
            raise ValueError(msg)

        self.question_id = question_id
        self.question_body = question_body
        self.question_type = question_type
        self._all_documents = documents
        self.agents = agents
        self.max_rounds = max_rounds
        self.max_docs = max_docs if max_docs is not None else len(documents)
        self.max_history_turns = max_history_turns
        self.stagnation_rounds = stagnation_rounds
        self.verbose = verbose
        self._rng = rng or random.Random()

        self._turns: list[DebateTurn] = []
        self._injected_docs: list[str] = []
        self._round_agreement_history: list[tuple[str, ...]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> QuorumResult:
        """Execute the debate and return the final answer."""
        self._inject_next_doc()  # Seed with the first document.

        consensus_reached = False
        for round_num in range(1, self.max_rounds + 1):
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"Round {round_num}  |  docs injected: {len(self._injected_docs)}")
                print("=" * 60)

            order = self._shuffled_agents()
            round_agreements: list[AgreementLevel] = []

            for agent in order:
                turn = self._run_agent_turn(agent, round_num)
                self._turns.append(turn)
                round_agreements.append(turn["agreement"])

                if turn["request_more_context"]:
                    injected = self._inject_next_doc()
                    if self.verbose and injected:
                        print(
                            f"  → Agent {agent.agent_id} requested more context "
                            f"(doc {len(self._injected_docs)} injected)."
                        )

            all_participated = all(a.has_participated for a in self.agents)
            all_strongly_agree = all(
                AGREEMENT_RANK[a] == AGREEMENT_RANK["strongly_agree"]  # type: ignore[literal-required]
                for a in round_agreements
            )

            sorted_agreements = tuple(sorted(round_agreements))
            self._round_agreement_history.append(sorted_agreements)
            stagnated = self._detect_stagnation()

            if self.verbose:
                print(
                    f"\n  Round {round_num} summary — "
                    f"agreements: {round_agreements}  "
                    f"all_participated={all_participated}  "
                    f"all_strongly_agree={all_strongly_agree}"
                    f"{'  STAGNATED' if stagnated else ''}"
                )

            if all_participated and all_strongly_agree:
                consensus_reached = True
                if self.verbose:
                    print("\nConsensus reached — terminating debate.")
                break

            if all_participated and stagnated:
                if self.verbose:
                    print(
                        f"\nAgreement distribution unchanged for "
                        f"{self.stagnation_rounds} rounds — terminating."
                    )
                break

        final = self._synthesise_answer()

        return QuorumResult(
            ideal_answer=final["ideal_answer"],
            exact_answer=final["exact_answer"],
            debate=self._turns,
            rounds=self._current_round(),
            consensus_reached=consensus_reached,
            docs_injected=len(self._injected_docs),
            question_id=self.question_id,
            question_type=self.question_type,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shuffled_agents(self) -> list[Agent]:
        order = list(self.agents)
        self._rng.shuffle(order)
        return order

    def _detect_stagnation(self) -> bool:
        """Return True if the agreement distribution hasn't changed for N rounds."""
        n = self.stagnation_rounds
        if len(self._round_agreement_history) < n:
            return False
        recent = self._round_agreement_history[-n:]
        return all(r == recent[0] for r in recent)

    def _inject_next_doc(self) -> bool:
        """Inject the next available document.  Returns True if one was added."""
        next_idx = len(self._injected_docs)
        if next_idx >= len(self._all_documents) or next_idx >= self.max_docs:
            return False
        self._injected_docs.append(self._all_documents[next_idx])
        return True

    def _run_agent_turn(self, agent: Agent, round_num: int) -> DebateTurn:
        messages = build_debate_turn_messages(
            question=self.question_body,
            question_type=self.question_type,
            injected_docs=list(self._injected_docs),
            history=list(self._turns),
            focus_description=agent.focus.description,
            max_history_turns=self.max_history_turns,
        )

        if self.verbose:
            print(
                f"\n  Agent {agent.agent_id} ({agent.focus.name}, {agent.model}) …",
                end="",
                flush=True,
            )

        raw = agent.generate(messages)
        parsed = parse_agent_response(raw)
        agent.mark_participated()

        if self.verbose:
            print(f" {parsed['agreement']}")

        return DebateTurn(
            round=round_num,
            agent_id=agent.agent_id,
            agent_focus=agent.focus.name,
            model=agent.model,
            opinion=parsed["opinion"],
            agreement=parsed["agreement"],
            request_more_context=parsed["request_more_context"],
        )

    def _current_round(self) -> int:
        return self._turns[-1]["round"] if self._turns else 0

    def _synthesise_answer(self) -> ParsedFinalAnswer:
        """Call an agent to produce the final clean answer."""
        if self.verbose:
            print("\nSynthesising final answer …")

        # Use the first agent's backend for synthesis (arbitrary but consistent).
        synthesiser = self.agents[0]
        messages = build_final_answer_messages(
            question=self.question_body,
            question_type=self.question_type,
            injected_docs=list(self._injected_docs),
            turns=list(self._turns),
        )
        raw = synthesiser.generate(messages)
        return _parse_final_answer(raw, self.question_type)


# ---------------------------------------------------------------------------
# Final answer parser
# ---------------------------------------------------------------------------


def _parse_final_answer(raw: str, question_type: str) -> ParsedFinalAnswer:
    parsed = extract_last_json(raw)
    if parsed is None:
        return ParsedFinalAnswer(
            ideal_answer=raw.strip(),
            exact_answer=None,
            raw=raw,
        )

    ideal: str = str(parsed.get("ideal_answer", raw.strip()))
    exact: Any = parsed.get("exact_answer", None)

    if question_type == "yesno" and isinstance(exact, str):
        exact = exact.lower().strip()
        if exact not in ("yes", "no"):
            exact = None

    elif question_type == "factoid":
        if isinstance(exact, str):
            exact = [exact]
        elif not isinstance(exact, list):
            exact = None

    elif question_type == "list":
        if isinstance(exact, list):
            normalised: list[list[str]] = []
            for item in exact:
                if isinstance(item, list):
                    normalised.append([str(x) for x in item])
                else:
                    normalised.append([str(item)])
            exact = normalised
        else:
            exact = None

    else:
        exact = None  # summary

    return ParsedFinalAnswer(
        ideal_answer=ideal,
        exact_answer=exact,
        raw=raw,
    )
