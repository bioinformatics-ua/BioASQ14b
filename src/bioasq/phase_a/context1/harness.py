"""Context-1 style multi-step retrieval harness."""

import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from bioasq.phase_a.context1.reranker import Context1Reranker
from bioasq.phase_a.context1.store import Context1CorpusStore
from bioasq.phase_a.context1.toolbox import Context1FastMCPToolbox
from bioasq.phase_a.context1.types import (
    AgentConfig,
    CorpusDocument,
    FinalSelection,
    RetrievedDocument,
    RolloutResult,
    ToolCallSpec,
    ToolResultEnvelope,
)
from bioasq.phase_a.context1.vllm_backend import Context1VLLMOpenAIBackend

_FINAL_DOC_RE = re.compile(
    r"<Document\s+id=\{?['\"]?([^>\}\"']+)['\"]?\}?\s*>\s*<Justification>(.*?)</Justification>\s*</Document>",
    re.DOTALL,
)

_SYSTEM_PROMPT = """
You are Context-1, a retrieval specialist operating over a biomedical literature corpus.

Your job is to identify the most relevant PubMed articles for the user's question.
You are not answering the question itself. You are ranking documents.

Available tools:
- search_corpus(query): hybrid semantic and lexical search over PMID-level articles, then rerank.
- grep_corpus(pattern): exact or regex search over whole-article text.
- read_document(document_id): inspect the content of one promising PMID.
- prune_chunks(chunk_ids): remove visible PMIDs from context if they are no longer useful.

Rules:
- Prefer search_corpus before reading full documents.
- Use grep_corpus when exact names, genes, proteins, compounds, or abbreviations matter.
- Use read_document when a PMID already looks promising and you need more detail.
- Keep only evidence that helps rank the final PMIDs; prune aggressively when context is crowded.
- If the tool messages say the hard cutoff is reached, only prune_chunks or conclude.
- Final output must contain only document tags in this exact form:
<Document id={PMID}><Justification>brief evidence-grounded justification</Justification></Document>
- Output up to 10 documents.
""".strip()


@dataclass(slots=True)
class ConversationTurn:
    assistant_content: str
    tool_calls: list[ToolCallSpec] = field(default_factory=list)
    tool_results: list[ToolResultEnvelope] = field(default_factory=list)


@dataclass(slots=True)
class AgentState:
    query: str
    turns: list[ConversationTurn] = field(default_factory=list)
    active_documents: dict[str, CorpusDocument] = field(default_factory=dict)
    encountered_documents: dict[str, CorpusDocument] = field(default_factory=dict)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    final_text: str = ""


class Context1Agent:
    """Runs the Context-1 inference loop against local BioASQ retrieval tools."""

    def __init__(
        self,
        *,
        backend: Context1VLLMOpenAIBackend,
        store: Context1CorpusStore,
        reranker: Context1Reranker,
        token_counter: Callable[[str], int],
        config: AgentConfig,
    ) -> None:
        self.backend = backend
        self.store = store
        self.reranker = reranker
        self.token_counter = token_counter
        self.config = config

    async def run_rollouts(self, query: str) -> list[RolloutResult]:
        """Run one or more independent retrieval rollouts for a query."""

        results: list[RolloutResult] = []
        for rollout_index in range(self.config.num_rollouts):
            seed = None
            if self.config.rollout_seed is not None:
                seed = self.config.rollout_seed + rollout_index
            results.append(await self._run_single_rollout(query, seed=seed))
        return results

    async def aggregate_rollouts(self, rollouts: list[RolloutResult]) -> list[RetrievedDocument]:
        """Fuse final document rankings across multiple rollouts."""

        if not rollouts:
            return []
        if len(rollouts) == 1:
            return rollouts[0].documents

        scores: dict[str, float] = defaultdict(float)
        exemplar: dict[str, RetrievedDocument] = {}
        for rollout in rollouts:
            for rank, document in enumerate(rollout.documents, start=1):
                scores[document.pmid] += 1.0 / (60 + rank)
                exemplar.setdefault(document.pmid, document)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            : self.config.final_topk
        ]
        return [
            RetrievedDocument(
                pmid=pmid,
                full_text=exemplar[pmid].full_text,
                score=score,
                justification=exemplar[pmid].justification,
            )
            for pmid, score in ranked
        ]

    async def _run_single_rollout(self, query: str, *, seed: int | None) -> RolloutResult:
        state = AgentState(query=query)
        toolbox = Context1FastMCPToolbox(agent=self, state=state)
        tools = await toolbox.openai_tools()

        for turn_index in range(self.config.max_turns):
            messages = self._build_messages(state)
            if self._remaining_budget(state) <= 0 and state.active_documents:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The visible document budget is exhausted. "
                            "Only call prune_chunks or conclude with final ranked documents."
                        ),
                    }
                )

            model_turn = await self.backend.complete(messages=messages, tools=tools, seed=seed)
            state.trajectory.append(
                {
                    "type": "assistant",
                    "turn": turn_index,
                    "content": model_turn.content,
                    "tool_calls": [asdict(call) for call in model_turn.tool_calls],
                    "visible_pmids": sorted(state.active_documents),
                    "visible_tokens": self._visible_tokens(state),
                }
            )

            if not model_turn.tool_calls:
                state.final_text = model_turn.content.strip()
                selections = self._parse_final_documents(state.final_text)
                if selections:
                    documents = await self._hydrate_documents(state, selections)
                    return RolloutResult(
                        selections=selections,
                        documents=documents,
                        final_text=state.final_text,
                        trajectory=state.trajectory,
                    )
                break

            turn = ConversationTurn(
                assistant_content=model_turn.content.strip(),
                tool_calls=model_turn.tool_calls,
            )
            tool_results: list[ToolResultEnvelope] = []
            for call in model_turn.tool_calls:
                tool_results.append(await self._execute_tool_call(toolbox, call))
            turn.tool_results = tool_results
            state.turns.append(turn)
            state.trajectory.append(
                {
                    "type": "tools",
                    "turn": turn_index,
                    "results": [asdict(result) for result in tool_results],
                    "visible_pmids": sorted(state.active_documents),
                    "visible_tokens": self._visible_tokens(state),
                }
            )

        selections = self._fallback_selections(state)
        documents = await self._hydrate_documents(state, selections)
        final_text = state.final_text or self._render_fallback_output(selections)
        return RolloutResult(
            selections=selections,
            documents=documents,
            final_text=final_text,
            trajectory=state.trajectory,
        )

    async def _execute_tool_call(
        self,
        toolbox: Context1FastMCPToolbox,
        call: ToolCallSpec,
    ) -> ToolResultEnvelope:
        return await toolbox.execute(call)

    def _register_documents(
        self,
        state: AgentState,
        documents: list[CorpusDocument],
    ) -> list[CorpusDocument]:
        newly_visible: list[CorpusDocument] = []
        for document in documents:
            encountered = state.encountered_documents.get(document.pmid)
            if encountered is None or self._should_replace_document(encountered, document):
                state.encountered_documents[document.pmid] = document

            visible = state.active_documents.get(document.pmid)
            if visible is None:
                state.active_documents[document.pmid] = document
                newly_visible.append(document)
                continue
            if self._should_replace_document(visible, document):
                state.active_documents[document.pmid] = document
                newly_visible.append(document)
        return newly_visible

    def _should_replace_document(
        self,
        current: CorpusDocument,
        candidate: CorpusDocument,
    ) -> bool:
        if candidate.is_expanded and not current.is_expanded:
            return True
        if candidate.token_count > current.token_count:
            return True
        return candidate.score > current.score

    def _cap_documents(
        self,
        documents: list[CorpusDocument],
        *,
        budget: int,
    ) -> list[CorpusDocument]:
        if budget <= 0:
            return []
        chosen: list[CorpusDocument] = []
        running = 0
        for document in documents:
            if document.token_count <= 0:
                continue
            if chosen and running + document.token_count > budget:
                break
            if not chosen and document.token_count > budget:
                continue
            chosen.append(document)
            running += document.token_count
            if running >= budget:
                break
        return chosen

    def _visible_tokens(self, state: AgentState) -> int:
        return sum(document.token_count for document in state.active_documents.values())

    def _hard_cutoff(self) -> int:
        return (
            int(self.config.context_window_tokens * self.config.hard_cutoff_ratio)
            - self.config.assistant_reserve_tokens
        )

    def _remaining_budget(self, state: AgentState) -> int:
        return max(0, self._hard_cutoff() - self._visible_tokens(state))

    def _budget_line(self, state: AgentState) -> str:
        visible = self._visible_tokens(state)
        hard_cutoff = self._hard_cutoff()
        return f"Visible document tokens: {visible}/{hard_cutoff}."

    def _build_messages(self, state: AgentState) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._user_prompt(state.query)},
        ]

        for turn in state.turns:
            assistant_message: dict[str, Any] = {"role": "assistant"}
            if turn.tool_calls:
                assistant_message["content"] = turn.assistant_content or None
                assistant_message["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name.value,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in turn.tool_calls
                ]
            else:
                assistant_message["content"] = turn.assistant_content
            messages.append(assistant_message)

            for result in turn.tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": self._render_tool_result(result, state),
                    }
                )
        return messages

    def _render_tool_result(self, result: ToolResultEnvelope, state: AgentState) -> str:
        if not result.returned_documents:
            return result.content
        visible = [
            document
            for document in result.returned_documents
            if document.pmid in state.active_documents
        ]
        if visible:
            return "\n".join(
                [
                    result.content.split("\n\n", 1)[0],
                    self._format_documents(visible),
                    self._budget_line(state),
                ]
            )
        return f"All documents returned by this tool call were pruned.\n{self._budget_line(state)}"

    def _user_prompt(self, query: str) -> str:
        return (
            "Retrieve the PubMed articles most relevant to this biomedical question.\n"
            f"Question: {query}\n"
            "Search iteratively, keep only useful evidence, and end with final document tags only."
        )

    def _format_documents(self, documents: list[CorpusDocument]) -> str:
        rendered: list[str] = []
        for document in documents:
            mode = "expanded" if document.is_expanded else "preview"
            score_text = f" score={document.score:.4f}" if document.score else ""
            rendered.append(
                f"<DocumentContext pmid={document.pmid} mode={mode}{score_text}>\n"
                f"{document.text}\n"
                "</DocumentContext>"
            )
        return "\n".join(rendered)

    def _parse_final_documents(self, text: str) -> list[FinalSelection]:
        selections: list[FinalSelection] = []
        seen: set[str] = set()
        for match in _FINAL_DOC_RE.finditer(text):
            pmid = match.group(1).strip().strip("{}\"'")
            if not pmid or pmid in seen:
                continue
            seen.add(pmid)
            selections.append(
                FinalSelection(
                    pmid=pmid,
                    justification=match.group(2).strip(),
                )
            )
            if len(selections) >= self.config.final_topk:
                break
        return selections

    def _fallback_selections(self, state: AgentState) -> list[FinalSelection]:
        ranked_pmids = sorted(
            state.encountered_documents.items(),
            key=lambda item: item[1].score,
            reverse=True,
        )[: self.config.final_topk]
        return [
            FinalSelection(
                pmid=pmid,
                justification=(document.text[:240] + "...")
                if len(document.text) > 240
                else document.text,
            )
            for pmid, document in ranked_pmids
        ]

    def _render_fallback_output(self, selections: list[FinalSelection]) -> str:
        return "\n".join(
            f"<Document id={{{selection.pmid}}}><Justification>{selection.justification}</Justification></Document>"
            for selection in selections
        )

    async def _hydrate_documents(
        self,
        state: AgentState,
        selections: list[FinalSelection],
    ) -> list[RetrievedDocument]:
        documents: list[RetrievedDocument] = []
        for selection in selections[: self.config.final_topk]:
            full_text = await self.store.get_document_text(selection.pmid)
            if not full_text:
                continue
            score = state.encountered_documents.get(
                selection.pmid,
                CorpusDocument(
                    pmid=selection.pmid,
                    text="",
                    token_count=0,
                ),
            ).score
            documents.append(
                RetrievedDocument(
                    pmid=selection.pmid,
                    full_text=full_text,
                    score=score,
                    justification=selection.justification,
                )
            )
        return documents
