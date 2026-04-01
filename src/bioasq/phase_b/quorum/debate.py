"""Debate orchestrator for the agent quorum system.

Flow
----
1. Each agent is fed a randomly sampled subset of *n* documents per round.
2. Agents participate sequentially in a randomly shuffled order within each round.
3. Each agent outputs an opinion, an agreement level, and a list of document IDs
   it wants to **keep** for the next round.
4. Kept documents are guaranteed to appear in the next round's sample for that
   agent; remaining slots are filled by random sampling from the full pool.
5. If an agent keeps *all n* documents it was shown, its personal *n* increases
   by 1 for the next round (expanding its evidence window).
6. Each agent must participate at least once before the termination check applies.
7. The debate ends when **all** agents output ``"strongly_agree"`` in the same
   round, or when agreement stagnates for several rounds.
8. A separate synthesis call (seeing **all** documents) produces the final answer.

The ``Debate`` class is a pure orchestrator — it does not own the backends and
assumes they are already loaded.
"""

import random
from typing import TYPE_CHECKING, Any

from bioasq.phase_b.backends.base import BaseModelBackend
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

if TYPE_CHECKING:
    from bioasq.phase_b.quorum import AgreementLevel


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
        Ordered list of document texts (abstracts).  Each document is assigned
        a stable 1-based ID corresponding to its position in this list.
    agents:
        List of :class:`~bioasq.phase_b.quorum.agent.Agent` instances.
    docs_per_sample:
        Number of documents each agent sees per round (initial *n*).  If an
        agent keeps all *n* documents, its personal *n* grows by 1.
    max_rounds:
        Hard upper bound on the number of debate rounds (safety limit).
    max_history_turns:
        Optional limit on the number of full history turns shown in the prompt.
    stagnation_rounds:
        Terminate if the agreement distribution is unchanged for this many rounds.
    verbose:
        Print progress to stdout.
    rng:
        Optional :class:`random.Random` instance for reproducible shuffling.
    synthesizer_backend:
        Optional backend to use for the final synthesis step.  When ``None``
        (the default), the first agent's backend is used.
    """

    def __init__(
        self,
        question_id: str,
        question_body: str,
        question_type: str,
        documents: list[str],
        agents: list[Agent],
        docs_per_sample: int = 3,
        max_rounds: int = 10,
        max_history_turns: int | None = None,
        stagnation_rounds: int = 3,
        verbose: bool = True,
        rng: random.Random | None = None,
        synthesizer_backend: BaseModelBackend | None = None,
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
        self.docs_per_sample = docs_per_sample
        self.max_rounds = max_rounds
        self.max_history_turns = max_history_turns
        self.stagnation_rounds = stagnation_rounds
        self.verbose = verbose
        self._rng = rng or random.Random()
        self._synthesizer_backend = synthesizer_backend

        self._turns: list[DebateTurn] = []
        self._round_agreement_history: list[tuple[str, ...]] = []

        # Per-agent document memory.
        # _agent_kept[agent_id] : 0-based indices the agent wants to keep.
        # _agent_n[agent_id]    : current sample size for that agent.
        self._agent_kept: dict[int, list[int]] = {a.agent_id: [] for a in agents}
        self._agent_n: dict[int, int] = {
            a.agent_id: min(docs_per_sample, len(documents)) for a in agents
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> QuorumResult:
        """Execute the debate and return the final answer."""
        consensus_reached = False
        for round_num in range(1, self.max_rounds + 1):
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"Round {round_num}")
                print("=" * 60)

            order = self._shuffled_agents()
            round_agreements: list[AgreementLevel] = []

            for agent in order:
                # 1. Sample documents for this agent.
                indexed_docs = self._sample_docs_for_agent(agent)

                # 2. Run the agent's turn.
                turn = self._run_agent_turn(agent, round_num, indexed_docs)
                self._turns.append(turn)
                round_agreements.append(turn["agreement"])

                # 3. Update the agent's document memory.
                self._update_agent_memory(agent, turn, indexed_docs)

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
            total_docs=len(self._all_documents),
            docs_per_sample=self.docs_per_sample,
            question_id=self.question_id,
            question_type=self.question_type,
        )

    # ------------------------------------------------------------------
    # Document sampling
    # ------------------------------------------------------------------

    def _sample_docs_for_agent(self, agent: Agent) -> list[tuple[int, str]]:
        """Build the document sample for *agent* this round.

        Kept documents are included first; remaining slots are filled by
        random sampling from the full pool (excluding already-selected
        indices to avoid duplicates within one sample).
        """
        agent_id = agent.agent_id
        n = self._agent_n[agent_id]
        kept = list(self._agent_kept[agent_id])

        # Start with the kept documents.
        selected: list[int] = list(kept)

        # Fill remaining slots from the full pool (excluding already selected).
        remaining = n - len(selected)
        if remaining > 0:
            pool = [i for i in range(len(self._all_documents)) if i not in set(selected)]
            sample_size = min(remaining, len(pool))
            if sample_size > 0:
                sampled = self._rng.sample(pool, sample_size)
                selected.extend(sampled)

        # Return as (1-based doc ID, text) pairs.
        return [(i + 1, self._all_documents[i]) for i in selected]

    def _update_agent_memory(
        self,
        agent: Agent,
        turn: DebateTurn,
        shown_docs: list[tuple[int, str]],
    ) -> None:
        """Update the agent's kept documents and sample size."""
        agent_id = agent.agent_id
        n = self._agent_n[agent_id]

        # Only keep document IDs that were actually shown this round.
        shown_ids = {doc_id for doc_id, _ in shown_docs}
        valid_kept_1based = [d for d in turn["kept_documents"] if d in shown_ids]
        kept_0based = [d - 1 for d in valid_kept_1based]

        # If agent kept all n documents, expand its window by 1.
        if len(kept_0based) >= n:
            self._agent_n[agent_id] = min(n + 1, len(self._all_documents))
            if self.verbose:
                print(
                    f"  → Agent {agent_id} kept all {n} docs "
                    f"— sample size now {self._agent_n[agent_id]}"
                )

        self._agent_kept[agent_id] = kept_0based

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

    def _run_agent_turn(
        self,
        agent: Agent,
        round_num: int,
        indexed_docs: list[tuple[int, str]],
    ) -> DebateTurn:
        messages = build_debate_turn_messages(
            question=self.question_body,
            question_type=self.question_type,
            indexed_docs=indexed_docs,
            total_docs=len(self._all_documents),
            history=list(self._turns),
            focus_description=agent.focus.description,
            max_history_turns=self.max_history_turns,
        )

        doc_ids_shown = [doc_id for doc_id, _ in indexed_docs]

        if self.verbose:
            print(
                f"\n  Agent {agent.agent_id} ({agent.focus.name}, {agent.model}) "
                f"docs={doc_ids_shown} …",
                end="",
                flush=True,
            )

        raw = agent.generate(messages)
        parsed = parse_agent_response(raw)
        agent.mark_participated()

        if self.verbose:
            kept = [d for d in parsed["kept_documents"] if d in set(doc_ids_shown)]
            print(f" {parsed['agreement']}  kept={kept}")

        return DebateTurn(
            round=round_num,
            agent_id=agent.agent_id,
            agent_focus=agent.focus.name,
            model=agent.model,
            opinion=parsed["opinion"],
            agreement=parsed["agreement"],
            documents_shown=doc_ids_shown,
            kept_documents=parsed["kept_documents"],
        )

    def _current_round(self) -> int:
        return self._turns[-1]["round"] if self._turns else 0

    def _synthesise_answer(self) -> ParsedFinalAnswer:
        """Call an agent to produce the final clean answer."""
        if self.verbose:
            print("\nSynthesising final answer …")

        # Use the dedicated synthesizer backend if provided, otherwise fall
        # back to the first agent (arbitrary but consistent).
        messages = build_final_answer_messages(
            question=self.question_body,
            question_type=self.question_type,
            all_documents=list(self._all_documents),
            turns=list(self._turns),
        )
        if self._synthesizer_backend is not None:
            raw = self._synthesizer_backend.generate_chat(messages)
        else:
            raw = self.agents[0].generate(messages)
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
