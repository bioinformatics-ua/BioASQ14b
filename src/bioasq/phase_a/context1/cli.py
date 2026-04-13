"""CLI for Context-1 article-level retrieval and agentic inference."""

from __future__ import annotations

import pathlib
from dataclasses import asdict
from typing import Annotated, Any

import orjson
import typer
from tqdm.auto import tqdm

from bioasq.common.utils import typer_async
from bioasq.phase_a.context1.harness import Context1Agent
from bioasq.phase_a.context1.reranker import Context1Reranker
from bioasq.phase_a.context1.store import Context1CorpusStore
from bioasq.phase_a.context1.tokenizer import Context1Tokenizer
from bioasq.phase_a.context1.types import AgentConfig
from bioasq.phase_a.context1.vllm_backend import Context1VLLMOpenAIBackend

PathType = pathlib.Path

app = typer.Typer(help="Context-1 style PMID retrieval and tool-calling inference.")


def _load_questions(path: PathType) -> list[dict[str, Any]]:
    return [orjson.loads(line) for line in path.open("rb") if line.strip()]


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


if __name__ == "__main__":
    app()
