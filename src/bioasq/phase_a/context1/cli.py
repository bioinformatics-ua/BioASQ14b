"""CLI for Context-1 article-level retrieval and agentic inference."""

from __future__ import annotations

import asyncio
import pathlib
import re
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Annotated, Any, Protocol, cast

import orjson
import typer
from tqdm.auto import tqdm

from bioasq.common import PROJECT_DATA_DIR
from bioasq.common.utils import typer_async
from bioasq.phase_a.context1.harness import Context1Agent
from bioasq.phase_a.context1.reranker import Context1Reranker
from bioasq.phase_a.context1.store import Context1CorpusStore
from bioasq.phase_a.context1.tokenizer import Context1Tokenizer
from bioasq.phase_a.context1.types import AgentConfig
from bioasq.phase_a.context1.vllm_backend import Context1VLLMOpenAIBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bioasq.phase_a.context1.types import CorpusDocument

PathType = pathlib.Path
_PUBMED_ID_RE = re.compile(r"(\d+)(?:/)?$")

app = typer.Typer(help="Context-1 style PMID retrieval and tool-calling inference.")


def _load_questions(path: PathType) -> list[dict[str, Any]]:
    return [orjson.loads(line) for line in path.open("rb") if line.strip()]


def _split_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


class _RerankerLike(Protocol):
    def score(self, query: str, documents: list[CorpusDocument]) -> list[CorpusDocument]: ...


def _question_id(question: dict[str, Any]) -> str:
    value = question.get("id", question.get("qid", ""))
    return "" if value is None else str(value)


def _question_body(question: dict[str, Any]) -> str:
    for key in ("body", "query_text", "query"):
        value = question.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _resolve_document_id(value: object) -> str | None:
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    match = _PUBMED_ID_RE.search(stripped)
    if match is not None:
        return match.group(1)
    return stripped


def _extract_document_text(document: dict[str, Any]) -> str:
    for key in ("text", "full_text"):
        value = document.get(key)
        if isinstance(value, str):
            return value

    title = document.get("title")
    abstract = document.get("abstract")
    parts = [part.strip() for part in (title, abstract) if isinstance(part, str) and part.strip()]
    return "  ".join(parts)


def _positive_source(question: dict[str, Any]) -> list[object]:
    source = question.get("pos_docs")
    if isinstance(source, list):
        return source
    source = question.get("documents")
    return source if isinstance(source, list) else []


def _question_year(question: dict[str, Any], *, year_override: int | None) -> int | None:
    if year_override is not None:
        return year_override

    baseline = question.get("baseline")
    if isinstance(baseline, int):
        return baseline
    if isinstance(baseline, str):
        match = re.search(r"(20\d{2})", baseline)
        if match is not None:
            return int(match.group(1))
    return None


async def _normalize_positive_docs(
    question: dict[str, Any],
    *,
    store: Context1CorpusStore,
) -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    order: list[str] = []

    for raw_document in _positive_source(question):
        doc_id: str | None = None
        text = ""
        if isinstance(raw_document, dict):
            typed_document = cast("dict[str, Any]", raw_document)
            doc_id = _resolve_document_id(
                typed_document.get("id")
                or typed_document.get("pmid")
                or typed_document.get("document")
            )
            text = _extract_document_text(typed_document)
        else:
            doc_id = _resolve_document_id(raw_document)

        if doc_id is None:
            continue

        existing = by_id.get(doc_id)
        if existing is None:
            by_id[doc_id] = {"id": doc_id, "text": text}
            order.append(doc_id)
            continue
        if not existing["text"] and text:
            existing["text"] = text

    missing_text_ids = [doc_id for doc_id in order if not by_id[doc_id]["text"]]
    if missing_text_ids:
        hydrated_texts = await asyncio.gather(
            *(store.get_document_text(doc_id) for doc_id in missing_text_ids)
        )
        for doc_id, text in zip(missing_text_ids, hydrated_texts, strict=True):
            by_id[doc_id]["text"] = text

    return [by_id[doc_id] for doc_id in order]


def _normalize_ensemble_scores(documents: list[CorpusDocument]) -> dict[str, float]:
    if not documents:
        return {}

    raw_scores = [document.score for document in documents]
    score_min = min(raw_scores)
    score_max = max(raw_scores)
    if score_max <= score_min:
        return {document.pmid: 0.0 for document in documents}

    scale = score_max - score_min
    return {document.pmid: (document.score - score_min) / scale for document in documents}


def _ensemble_reranked_documents(
    query: str,
    documents: list[CorpusDocument],
    rerankers: Sequence[_RerankerLike],
) -> list[CorpusDocument]:
    if not rerankers:
        raise ValueError("At least one reranker is required for negatives mining.")
    if not documents:
        return []

    ensembled_scores = {document.pmid: 0.0 for document in documents}

    for reranker in rerankers:
        scored = reranker.score(query, documents)
        normalized_scores = _normalize_ensemble_scores(scored)
        for pmid, normalized_score in normalized_scores.items():
            ensembled_scores[pmid] = ensembled_scores.get(pmid, 0.0) + normalized_score

    reranker_count = float(len(rerankers))
    return sorted(
        [
            replace(document, score=ensembled_scores[document.pmid] / reranker_count)
            for document in documents
        ],
        key=lambda document: (-document.score, document.pmid),
    )


async def _mine_negatives_for_question(
    *,
    question: dict[str, Any],
    store: Context1CorpusStore,
    rerankers: Sequence[_RerankerLike],
    num_negatives: int,
    candidate_buffer: int,
    bm25_topk: int,
    dense_topk: int,
    search_candidate_pool_size: int,
    year_override: int | None,
) -> dict[str, Any]:
    qid = _question_id(question)
    if not qid:
        raise ValueError("Each training row must contain an 'id' field.")

    body = _question_body(question)
    if not body:
        raise ValueError(f"Question '{qid}' is missing a non-empty body.")

    pos_docs = await _normalize_positive_docs(question, store=store)
    positive_pmids = {document["id"] for document in pos_docs}
    effective_depth = num_negatives + candidate_buffer
    question_year = _question_year(question, year_override=year_override)

    fused = await store.hybrid_search_candidates(
        body,
        bm25_topk=max(bm25_topk, effective_depth),
        dense_topk=max(dense_topk, effective_depth),
        year=question_year,
        exclude_pmids=positive_pmids,
    )
    reranked = _ensemble_reranked_documents(
        body,
        fused[: max(search_candidate_pool_size, effective_depth)],
        rerankers,
    )
    neg_docs = [
        {
            "id": document.pmid,
            "text": document.text,
            "score": document.score,
        }
        for document in reranked
        if document.pmid not in positive_pmids
    ][:num_negatives]

    if len(neg_docs) < num_negatives:
        raise RuntimeError(
            f"Question '{qid}' produced only {len(neg_docs)} negatives after filtering. "
            "Increase candidate depth or reduce --num-negatives."
        )

    return {
        "id": qid,
        "body": body,
        "pos_docs": pos_docs,
        "neg_docs": neg_docs,
    }


@app.command("retrieve")
@typer_async
async def retrieve(
    testset_file: Annotated[
        PathType,
        typer.Argument(..., exists=True, help="JSONL with question id and body fields."),
    ],
    output_file: Annotated[
        PathType,
        typer.Option(..., "-o", "--output", help="Output JSONL with final document rankings."),
    ],
    trajectory_file: Annotated[
        PathType | None,
        typer.Option(help="Optional JSONL file with full rollout trajectories."),
    ] = None,
    model_name: Annotated[
        str, typer.Option(help="Served model name on the vLLM server.")
    ] = "chromadb/context-1",
    vllm_base_url: Annotated[
        str, typer.Option(help="Base URL of the vLLM OpenAI-compatible server.")
    ] = "http://127.0.0.1:8000",
    api_key: Annotated[
        str, typer.Option(help="OpenAI-compatible API key for the vLLM server.")
    ] = "EMPTY",
    temperature: Annotated[float, typer.Option(help="Sampling temperature.")] = 0.2,
    max_completion_tokens: Annotated[
        int, typer.Option(help="Maximum completion tokens per model turn.")
    ] = 2_048,
    max_turns: Annotated[int, typer.Option(help="Maximum tool-calling turns per rollout.")] = 12,
    num_rollouts: Annotated[
        int, typer.Option(help="Number of independent rollouts per query.")
    ] = 1,
    rollout_seed: Annotated[
        int | None, typer.Option(help="Optional base random seed for rollouts.")
    ] = None,
    year: Annotated[int | None, typer.Option(help="Optional BioASQ baseline year filter.")] = None,
    final_topk: Annotated[
        int, typer.Option(help="Maximum final PMIDs returned per question.")
    ] = 10,
    bm25_topk: Annotated[int, typer.Option(help="BM25 article candidate depth.")] = 50,
    dense_topk: Annotated[int, typer.Option(help="Dense article candidate depth.")] = 50,
    grep_topk: Annotated[int, typer.Option(help="Maximum regex article matches returned.")] = 5,
    search_candidate_pool_size: Annotated[
        int, typer.Option(help="Number of fused article candidates to rerank.")
    ] = 50,
    search_preview_tokens: Annotated[
        int, typer.Option(help="Maximum tokens shown per search_corpus document preview.")
    ] = 256,
    grep_preview_tokens: Annotated[
        int, typer.Option(help="Maximum tokens shown per grep_corpus document preview.")
    ] = 256,
    search_tool_token_budget: Annotated[
        int, typer.Option(help="Token budget for one search tool response.")
    ] = 4_096,
    read_tool_token_budget: Annotated[
        int, typer.Option(help="Token budget for one read_document response.")
    ] = 4_096,
    context_window_tokens: Annotated[
        int, typer.Option(help="Approximate model context budget used for chunk visibility.")
    ] = 32_768,
    soft_warning_ratio: Annotated[
        float, typer.Option(help="Soft warning ratio for visible document tokens.")
    ] = 0.5,
    hard_cutoff_ratio: Annotated[
        float, typer.Option(help="Hard cutoff ratio for visible chunk tokens.")
    ] = 0.85,
    assistant_reserve_tokens: Annotated[
        int, typer.Option(help="Tokens reserved for the assistant response.")
    ] = 2_048,
    reranker_model_name: Annotated[
        str, typer.Option(help="Cross-encoder reranker model.")
    ] = "BAAI/bge-reranker-v2-m3",
    reranker_batch_size: Annotated[int, typer.Option(help="Reranker batch size.")] = 16,
    reranker_max_length: Annotated[int, typer.Option(help="Reranker max sequence length.")] = 1_024,
    reranker_device: Annotated[str, typer.Option(help="Torch device for the reranker.")] = "cuda",
    reranker_invert_scores: Annotated[
        bool, typer.Option(help="Invert reranker logits before sorting.")
    ] = False,
    tei_embed_url: Annotated[str | None, typer.Option(help="TEI embeddings endpoint.")] = None,
    collection_name: Annotated[
        str, typer.Option(help="Qdrant collection with existing PMID vectors.")
    ] = "articles",
) -> None:
    """Run Context-1 retrieval over a BioASQ question set using existing PMID embeddings."""

    config = AgentConfig(
        model_name=model_name,
        vllm_base_url=vllm_base_url,
        api_key=api_key,
        max_turns=max_turns,
        context_window_tokens=context_window_tokens,
        search_tool_token_budget=search_tool_token_budget,
        read_tool_token_budget=read_tool_token_budget,
        assistant_reserve_tokens=assistant_reserve_tokens,
        soft_warning_ratio=soft_warning_ratio,
        hard_cutoff_ratio=hard_cutoff_ratio,
        search_candidate_pool_size=search_candidate_pool_size,
        bm25_topk=bm25_topk,
        dense_topk=dense_topk,
        grep_topk=grep_topk,
        final_topk=final_topk,
        search_preview_tokens=search_preview_tokens,
        grep_preview_tokens=grep_preview_tokens,
        year=year,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        num_rollouts=num_rollouts,
        rollout_seed=rollout_seed,
        reranker_model_name=reranker_model_name,
        reranker_batch_size=reranker_batch_size,
        reranker_max_length=reranker_max_length,
        reranker_device=reranker_device,
        reranker_invert_scores=reranker_invert_scores,
        tei_embed_url=tei_embed_url,
        qdrant_collection=collection_name,
    )

    tokenizer = Context1Tokenizer(model_name=config.model_name)
    store = Context1CorpusStore(
        token_counter=tokenizer.count_tokens,
        text_truncator=tokenizer.truncate,
        tei_embed_url=config.tei_embed_url,
        collection_name=config.qdrant_collection,
    )
    reranker = Context1Reranker(
        config.reranker_model_name,
        batch_size=config.reranker_batch_size,
        max_length=config.reranker_max_length,
        device=config.reranker_device,
        invert_scores=config.reranker_invert_scores,
    )
    backend = Context1VLLMOpenAIBackend(
        model_name=config.model_name,
        base_url=config.vllm_base_url,
        api_key=config.api_key,
        temperature=config.temperature,
        max_completion_tokens=config.max_completion_tokens,
    )
    agent = Context1Agent(
        backend=backend,
        store=store,
        reranker=reranker,
        token_counter=tokenizer.count_tokens,
        config=config,
    )

    questions = _load_questions(testset_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if trajectory_file is not None:
        trajectory_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        status = await store.prepare_existing_corpus(year=year, ensure_bm25=False)
        if not status["qdrant_collection_exists"]:
            raise RuntimeError(
                f"Qdrant collection '{config.qdrant_collection}' is missing. "
                "Point Context-1 at the existing PMID embedding collection."
            )
        with output_file.open("wb") as out_f:
            trajectory_handle = trajectory_file.open("wb") if trajectory_file is not None else None
            try:
                for question in tqdm(questions, desc="Context-1 retrieve", unit="q"):
                    qid = str(question["id"])
                    body = str(question["body"])
                    rollouts = await agent.run_rollouts(body)
                    documents = await agent.aggregate_rollouts(rollouts)
                    output_row = {
                        "qid": qid,
                        "results": [
                            {
                                "pmid": document.pmid,
                                "full_text": document.full_text,
                                "score": document.score,
                                "justification": document.justification,
                            }
                            for document in documents
                        ],
                    }
                    out_f.write(orjson.dumps(output_row) + b"\n")

                    if trajectory_handle is not None:
                        trajectory_row = {
                            "qid": qid,
                            "rollouts": [
                                {
                                    "selections": [
                                        asdict(selection) for selection in rollout.selections
                                    ],
                                    "documents": [
                                        asdict(document) for document in rollout.documents
                                    ],
                                    "final_text": rollout.final_text,
                                    "trajectory": rollout.trajectory,
                                }
                                for rollout in rollouts
                            ],
                        }
                        trajectory_handle.write(orjson.dumps(trajectory_row) + b"\n")
            finally:
                if trajectory_handle is not None:
                    trajectory_handle.close()
    finally:
        await backend.close()
        await store.close()


@app.command("negatives")
@typer_async
async def negatives(
    training_file: Annotated[
        PathType,
        typer.Argument(
            ..., exists=True, help="JSONL training data with positives in documents or pos_docs."
        ),
    ],
    output_file: Annotated[
        PathType,
        typer.Option(..., "-o", "--output", help="Output JSONL with pos_docs and neg_docs."),
    ] = PROJECT_DATA_DIR / "negatives_fixed.jsonl",
    num_negatives: Annotated[
        int,
        typer.Option("--num-negatives", "-n", help="Final negatives per question."),
    ] = 100,
    candidate_buffer: Annotated[
        int,
        typer.Option(
            help=(
                "Extra fused candidates fetched beyond the final negative count to absorb "
                "positive exclusion and deduplication."
            )
        ),
    ] = 100,
    year: Annotated[
        int | None,
        typer.Option(
            help="Optional year override. When omitted, use each row's baseline year if present."
        ),
    ] = None,
    bm25_topk: Annotated[int, typer.Option(help="BM25 candidate depth before fusion.")] = 200,
    dense_topk: Annotated[int, typer.Option(help="Dense candidate depth before fusion.")] = 200,
    search_candidate_pool_size: Annotated[
        int,
        typer.Option(help="Number of fused candidates reranked per question."),
    ] = 200,
    reranker_models_raw: Annotated[
        list[str],
        typer.Option(
            "--reranker-model",
            help="Cross-encoder reranker models to ensemble.",
        ),
    ] = ["BAAI/bge-reranker-v2-m3"],
    reranker_batch_size: Annotated[int, typer.Option(help="Reranker batch size.")] = 16,
    reranker_max_length: Annotated[int, typer.Option(help="Reranker max sequence length.")] = 1_024,
    reranker_device: Annotated[str, typer.Option(help="Torch device for the reranker.")] = "cuda",
    reranker_invert_scores: Annotated[
        bool, typer.Option(help="Invert reranker logits before sorting.")
    ] = False,
    tei_embed_url: Annotated[str | None, typer.Option(help="TEI embeddings endpoint.")] = None,
    collection_name: Annotated[
        str, typer.Option(help="Qdrant collection with existing PMID vectors.")
    ] = "articles",
) -> None:
    """Mine hard negatives with the Context-1 hybrid retrieval and reranker stack."""

    if num_negatives <= 0:
        raise typer.BadParameter("--num-negatives must be positive.")
    if candidate_buffer < 0:
        raise typer.BadParameter("--candidate-buffer cannot be negative.")
    reranker_model_names = _split_csv(reranker_models_raw)
    if not reranker_model_names:
        raise typer.BadParameter("At least one reranker model must be provided.")

    store = Context1CorpusStore(
        token_counter=lambda _text: 0,
        text_truncator=lambda text, _max_tokens: text,
        tei_embed_url=tei_embed_url,
        collection_name=collection_name,
    )
    rerankers = [
        Context1Reranker(
            model_name,
            batch_size=reranker_batch_size,
            max_length=reranker_max_length,
            device=reranker_device,
            invert_scores=reranker_invert_scores,
        )
        for model_name in reranker_model_names
    ]

    questions = _load_questions(training_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        status = await store.prepare_existing_corpus(year=year, ensure_bm25=True)
        if not status["qdrant_collection_exists"]:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' is missing. "
                "Point Context-1 at the existing PMID embedding collection."
            )

        with output_file.open("wb") as out_f:
            for question in tqdm(questions, desc="Context-1 negatives", unit="q"):
                row = await _mine_negatives_for_question(
                    question=question,
                    store=store,
                    rerankers=rerankers,
                    num_negatives=num_negatives,
                    candidate_buffer=candidate_buffer,
                    bm25_topk=bm25_topk,
                    dense_topk=dense_topk,
                    search_candidate_pool_size=search_candidate_pool_size,
                    year_override=year,
                )
                out_f.write(orjson.dumps(row) + b"\n")
    finally:
        await store.close()


if __name__ == "__main__":
    app()
