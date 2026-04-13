"""FastMCP-backed retrieval tools for the Context-1 agent."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from bioasq.phase_a.context1.types import CorpusDocument, ToolCallSpec, ToolName, ToolResultEnvelope

if TYPE_CHECKING:
    from bioasq.phase_a.context1.harness import AgentState, Context1Agent

_TOOL_DESCRIPTIONS: dict[ToolName, str] = {
    ToolName.SEARCH_CORPUS: (
        "Hybrid PMID-level retrieval followed by reranking. "
        "Use for semantic search over the corpus."
    ),
    ToolName.GREP_CORPUS: (
        "Regex search over whole-article text. "
        "Use for exact strings, gene symbols, acronyms, or trial identifiers."
    ),
    ToolName.READ_DOCUMENT: "Read one promising PMID in document form.",
    ToolName.PRUNE_CHUNKS: (
        "Remove visible PMIDs from context while preserving encountered-history "
        "deduplication. The tool name is historical; the ids are PMIDs."
    ),
}


class Context1FastMCPToolbox:
    """Owns the FastMCP tool registry used by one retrieval rollout."""

    def __init__(self, *, agent: Context1Agent, state: AgentState) -> None:
        self._agent = agent
        self._state = state
        self._app = FastMCP("bioasq-context1")
        self._openai_tools: list[dict[str, Any]] | None = None
        self._register_tools()

    async def openai_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schemas derived from FastMCP metadata."""

        if self._openai_tools is None:
            tools = await self._app.list_tools(run_middleware=False)
            self._openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": mcp_tool.name,
                        "description": mcp_tool.description or "",
                        "parameters": mcp_tool.inputSchema,
                    },
                }
                for tool in tools
                for mcp_tool in [tool.to_mcp_tool()]
            ]
        return self._openai_tools

    async def execute(self, call: ToolCallSpec) -> ToolResultEnvelope:
        """Execute one model-emitted tool call through FastMCP."""

        try:
            result = await self._app.call_tool(
                call.name.value,
                call.arguments,
                run_middleware=False,
            )
        except Exception as exc:
            return ToolResultEnvelope(
                call_id=call.call_id,
                name=call.name,
                content=f"Tool failed: {exc}\n{self._agent._budget_line(self._state)}",
                error=True,
            )

        payload = result.structured_content if isinstance(result.structured_content, dict) else {}
        content = payload.get("content")
        if not isinstance(content, str):
            content = self._coalesce_tool_result_content(result)
        return ToolResultEnvelope(
            call_id=call.call_id,
            name=call.name,
            content=content,
            returned_documents=self._coerce_documents(payload.get("returned_documents")),
            error=bool(payload.get("error")),
        )

    def _register_tools(self) -> None:
        @self._app.tool(
            name=ToolName.SEARCH_CORPUS.value,
            description=_TOOL_DESCRIPTIONS[ToolName.SEARCH_CORPUS],
        )
        async def search_corpus(
            query: Annotated[
                str,
                Field(description="A biomedical retrieval query describing the evidence to find."),
            ],
        ) -> dict[str, Any]:
            return await self._search_corpus(query)

        @self._app.tool(
            name=ToolName.GREP_CORPUS.value,
            description=_TOOL_DESCRIPTIONS[ToolName.GREP_CORPUS],
        )
        async def grep_corpus(
            pattern: Annotated[
                str,
                Field(
                    description="Case-insensitive regular expression to search within article text."
                ),
            ],
        ) -> dict[str, Any]:
            return await self._grep_corpus(pattern)

        @self._app.tool(
            name=ToolName.READ_DOCUMENT.value,
            description=_TOOL_DESCRIPTIONS[ToolName.READ_DOCUMENT],
        )
        async def read_document(
            document_id: Annotated[
                str,
                Field(description="PMID of the document to inspect."),
            ],
        ) -> dict[str, Any]:
            return await self._read_document(document_id)

        @self._app.tool(
            name=ToolName.PRUNE_CHUNKS.value,
            description=_TOOL_DESCRIPTIONS[ToolName.PRUNE_CHUNKS],
        )
        def prune_chunks(
            chunk_ids: Annotated[
                list[int],
                Field(
                    description=(
                        "PMIDs to remove from currently visible context. "
                        "The parameter name is historical."
                    )
                ),
            ],
        ) -> dict[str, Any]:
            return self._prune_chunks(chunk_ids)

    async def _search_corpus(self, query: str) -> dict[str, Any]:
        if not query.strip():
            return self._payload(
                content=f"Missing query argument.\n{self._agent._budget_line(self._state)}",
                error=True,
            )
        remaining_budget = self._agent._remaining_budget(self._state)
        if remaining_budget <= 0:
            return self._hard_cutoff_payload()

        fused = await self._agent.store.hybrid_search_candidates(
            query,
            bm25_topk=self._agent.config.bm25_topk,
            dense_topk=self._agent.config.dense_topk,
            year=self._agent.config.year,
            exclude_pmids=set(self._state.encountered_documents),
        )
        reranked = self._agent.reranker.score(
            query,
            fused[: self._agent.config.search_candidate_pool_size],
        )
        previews = self._agent.store.preview_documents(
            reranked,
            max_tokens=min(self._agent.config.search_preview_tokens, remaining_budget),
        )
        chosen = self._agent._cap_documents(
            previews,
            budget=min(self._agent.config.search_tool_token_budget, remaining_budget),
        )
        newly_visible = self._agent._register_documents(self._state, chosen)
        summary = [
            f"search_corpus query: {query}",
            (
                "Hybrid candidates: "
                f"{len(fused)}; reranked candidates: "
                f"{min(len(fused), self._agent.config.search_candidate_pool_size)}."
            ),
            f"Returned {len(newly_visible)} new document previews.",
            self._agent._budget_line(self._state),
        ]
        if newly_visible:
            summary.append(self._agent._format_documents(newly_visible))
        else:
            summary.append(
                "No new documents fit the remaining budget or all results were already encountered."
            )
        return self._payload(
            content="\n".join(summary),
            returned_documents=newly_visible,
        )

    async def _grep_corpus(self, pattern: str) -> dict[str, Any]:
        if not pattern.strip():
            return self._payload(
                content=f"Missing pattern argument.\n{self._agent._budget_line(self._state)}",
                error=True,
            )
        remaining_budget = self._agent._remaining_budget(self._state)
        if remaining_budget <= 0:
            return self._hard_cutoff_payload()

        matches = await self._agent.store.grep_search(
            pattern,
            topk=self._agent.config.grep_topk,
            year=self._agent.config.year,
            exclude_pmids=set(self._state.encountered_documents),
            preview_tokens=min(self._agent.config.grep_preview_tokens, remaining_budget),
        )
        chosen = self._agent._cap_documents(
            matches,
            budget=min(self._agent.config.search_tool_token_budget, remaining_budget),
        )
        newly_visible = self._agent._register_documents(self._state, chosen)
        summary = [
            f"grep_corpus pattern: {pattern}",
            f"Returned {len(newly_visible)} new regex-matched documents.",
            self._agent._budget_line(self._state),
        ]
        if newly_visible:
            summary.append(self._agent._format_documents(newly_visible))
        else:
            summary.append("No unseen grep matches fit the current budget.")
        return self._payload(
            content="\n".join(summary),
            returned_documents=newly_visible,
        )

    async def _read_document(self, document_id: str) -> dict[str, Any]:
        if not document_id.strip():
            return self._payload(
                content=f"Missing document_id argument.\n{self._agent._budget_line(self._state)}",
                error=True,
            )
        remaining_budget = self._agent._remaining_budget(self._state)
        if remaining_budget <= 0:
            return self._hard_cutoff_payload()

        existing = self._state.active_documents.get(document_id)
        if existing is not None and existing.is_expanded:
            return self._payload(
                content=(
                    f"read_document PMID: {document_id}\n"
                    "This PMID is already expanded in visible context.\n"
                    f"{self._agent._budget_line(self._state)}"
                )
            )

        document = await self._agent.store.read_document(
            document_id,
            max_tokens=min(self._agent.config.read_tool_token_budget, remaining_budget),
        )
        if document is None or document.token_count <= 0:
            return self._payload(
                content=(
                    f"read_document PMID: {document_id}\n"
                    "Unable to load document text for this PMID.\n"
                    f"{self._agent._budget_line(self._state)}"
                ),
                error=True,
            )

        newly_visible = self._agent._register_documents(self._state, [document])
        summary = [
            f"read_document PMID: {document_id}",
            f"Returned {len(newly_visible)} expanded document entries.",
            self._agent._budget_line(self._state),
        ]
        if newly_visible:
            summary.append(self._agent._format_documents(newly_visible))
        else:
            summary.append("This PMID was already visible at equal or greater detail.")
        return self._payload(
            content="\n".join(summary),
            returned_documents=newly_visible,
        )

    def _prune_chunks(self, chunk_ids: list[int]) -> dict[str, Any]:
        removed = 0
        for chunk_id in chunk_ids:
            if self._state.active_documents.pop(str(chunk_id), None) is not None:
                removed += 1
        return self._payload(
            content=(
                f"prune_chunks removed {removed} visible PMIDs. "
                f"{len(self._state.active_documents)} PMIDs remain visible.\n"
                f"{self._agent._budget_line(self._state)}"
            )
        )

    def _hard_cutoff_payload(self) -> dict[str, Any]:
        return self._payload(
            content=(
                "Visible context is at the hard cutoff. "
                "Only call prune_chunks or conclude with final documents.\n"
                f"{self._agent._budget_line(self._state)}"
            ),
            error=True,
        )

    def _payload(
        self,
        *,
        content: str,
        returned_documents: list[CorpusDocument] | None = None,
        error: bool = False,
    ) -> dict[str, Any]:
        return {
            "content": content,
            "returned_documents": [asdict(document) for document in returned_documents or []],
            "error": error,
        }

    def _coerce_documents(self, payload: Any) -> list[CorpusDocument]:
        if not isinstance(payload, list):
            return []
        documents: list[CorpusDocument] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                documents.append(CorpusDocument(**item))
            except TypeError:
                continue
        return documents

    def _coalesce_tool_result_content(self, result: Any) -> str:
        parts: list[str] = []
        for block in getattr(result, "content", []):
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        return "\n".join(parts)
