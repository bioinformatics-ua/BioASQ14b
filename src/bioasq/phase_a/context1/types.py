"""Typed state containers for the Context-1 retrieval harness."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ToolName(StrEnum):
    """Available retrieval tools exposed to the Context-1 model."""

    SEARCH_CORPUS = "search_corpus"
    GREP_CORPUS = "grep_corpus"
    READ_DOCUMENT = "read_document"
    PRUNE_CHUNKS = "prune_chunks"
    INVALID = "__invalid_tool__"


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One PMID-level retrieval candidate shown to the agent."""

    pmid: str
    text: str
    token_count: int
    score: float = 0.0
    is_expanded: bool = False


@dataclass(frozen=True, slots=True)
class ToolCallSpec:
    """Structured function call emitted by the model."""

    call_id: str
    name: ToolName
    arguments: dict[str, Any]


@dataclass(slots=True)
class AgentStateSnapshot:
    """Frozen view of model-visible documents for one tool execution."""

    active_documents: dict[str, CorpusDocument] = field(default_factory=dict)
    encountered_documents: dict[str, CorpusDocument] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResultEnvelope:
    """Recorded result for a tool call, used to rebuild model-visible context."""

    call_id: str
    name: ToolName
    content: str
    returned_documents: list[CorpusDocument] = field(default_factory=list)
    pruned_pmids: list[str] = field(default_factory=list)
    error: bool = False


@dataclass(slots=True)
class FinalSelection:
    """One final document selected by the agent."""

    pmid: str
    justification: str = ""


@dataclass(slots=True)
class RetrievedDocument:
    """Final hydrated document output for a query."""

    pmid: str
    full_text: str
    score: float
    justification: str = ""


@dataclass(slots=True)
class AgentConfig:
    """Runtime settings for the Context-1 harness and retrieval tools."""

    model_name: str = "chromadb/context-1"
    base_url: str = "http://127.0.0.1:8000"
    api_key: str = "EMPTY"
    max_turns: int = 12
    context_window_tokens: int = 32_768
    search_tool_token_budget: int = 4_096
    read_tool_token_budget: int = 4_096
    assistant_reserve_tokens: int = 2_048
    soft_warning_ratio: float = 0.5
    hard_cutoff_ratio: float = 0.85
    search_candidate_pool_size: int = 50
    bm25_topk: int = 50
    dense_topk: int = 50
    grep_topk: int = 5
    final_topk: int = 10
    search_preview_tokens: int = 256
    grep_preview_tokens: int = 256
    year: int | None = None
    temperature: float = 0.2
    max_completion_tokens: int = 2_048
    num_rollouts: int = 1
    rollout_seed: int | None = None
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_batch_size: int = 16
    reranker_max_length: int = 1_024
    reranker_device: str = "cuda"
    reranker_invert_scores: bool = False
    tei_embed_url: str | None = None
    qdrant_collection: str = "articles"
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = True


@dataclass(slots=True)
class RolloutResult:
    """Complete result for a single Context-1 rollout."""

    selections: list[FinalSelection]
    documents: list[RetrievedDocument]
    final_text: str
    trajectory: list[dict[str, Any]]
