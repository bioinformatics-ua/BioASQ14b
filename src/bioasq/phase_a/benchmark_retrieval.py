"""Benchmark Phase A retrieval methods against BioASQ gold documents."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import orjson
import typer
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

from bioasq.common.metrics import DEFAULT_RETRIEVAL_METRICS, evaluate_retrieval_run
from bioasq.common.utils import typer_async
from bioasq.data.database import bm25_search, close_pool, close_qdrant_client, semantic_search
from bioasq.data.dataloader import BioASQDataLoader
from bioasq.phase_a.context1.harness import Context1Agent
from bioasq.phase_a.context1.reranker import Context1Reranker
from bioasq.phase_a.context1.store import Context1CorpusStore
from bioasq.phase_a.context1.tokenizer import Context1Tokenizer
from bioasq.phase_a.context1.types import AgentConfig, CorpusDocument
from bioasq.phase_a.context1.vllm_backend import Context1VLLMOpenAIBackend
from bioasq.phase_a.reranker.model import short_model_name
from bioasq.phase_a.retrieval.fusion import fuse_retrieval_lists_rrf, fuse_retrieval_lists_wsum
from bioasq.phase_a.retrieval.query_encoder import embed_queries_tei

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bioasq.common.types import DocumentWithScore, Question

PathType = Path
_PMID_RE = re.compile(r"(\d+)$")

app = typer.Typer(help="Benchmark Phase A retrieval methods against BioASQ gold documents.")


@dataclass(frozen=True, slots=True)
class RetrievedItem:
    """Normalized retrieval item stored by the benchmark."""

    pmid: str
    full_text: str
    score: float
    justification: str = ""


def _split_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_weights(raw: str) -> tuple[float, float]:
    parts = _split_csv(raw)
    if len(parts) != 2:
        raise typer.BadParameter("wsum-weights must contain exactly two comma-separated values")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise typer.BadParameter("wsum-weights must contain valid floats") from exc


def _extract_pmid(document_ref: str) -> str | None:
    match = _PMID_RE.search(document_ref.strip())
    if match is None:
        return None
    return match.group(1)


def _load_gold_questions(
    gold_file: PathType,
    *,
    limit: int | None = None,
) -> tuple[list[Question], dict[str, dict[str, int]]]:
    loader = BioASQDataLoader(gold_file)
    questions: list[Question] = []
    qrels: dict[str, dict[str, int]] = {}

    for question in loader:
        relevant_pmids: dict[str, int] = {}
        for document_ref in question.documents:
            pmid = _extract_pmid(document_ref)
            if pmid is not None:
                relevant_pmids[pmid] = 1
        if not relevant_pmids:
            continue
        questions.append(question)
        qrels[question.id] = relevant_pmids
        if limit is not None and len(questions) >= limit:
            break

    if not questions:
        raise typer.BadParameter(
            "The input file does not contain any questions with gold document PMIDs.",
        )

    return questions, qrels


def _to_items(documents: list[DocumentWithScore]) -> list[RetrievedItem]:
    return [
        RetrievedItem(
            pmid=document.pmid,
            full_text=document.full_text,
            score=float(document.score),
        )
        for document in documents
    ]


def _normalize_items(
    primary: list[RetrievedItem],
    *,
    top_k: int,
    fallback_lists: list[list[RetrievedItem]] | None = None,
) -> list[RetrievedItem]:
    seen: set[str] = set()
    normalized: list[RetrievedItem] = []

    for item in primary:
        if item.pmid in seen:
            continue
        seen.add(item.pmid)
        normalized.append(item)
        if len(normalized) >= top_k:
            return normalized

    for fallback in fallback_lists or []:
        for item in fallback:
            if item.pmid in seen:
                continue
            seen.add(item.pmid)
            normalized.append(item)
            if len(normalized) >= top_k:
                return normalized

    return normalized


def _limit_top_k(documents: list[RetrievedItem], *, top_k: int) -> list[RetrievedItem]:
    """Clip retrieval output without failing on underfilled runs.

    The shared retrieval pipeline already tolerates methods returning fewer than
    ``top_k`` documents. The benchmark tracks underfilled runs explicitly via the
    rough metrics instead of aborting the whole evaluation.
    """

    return documents[:top_k]


def _build_run_dict(results_by_qid: dict[str, list[RetrievedItem]]) -> dict[str, dict[str, float]]:
    return {
        qid: {item.pmid: float(item.score) for item in documents}
        for qid, documents in results_by_qid.items()
    }


def _compute_rough_metrics(
    results_by_qid: dict[str, list[RetrievedItem]],
    qrels: dict[str, dict[str, int]],
    *,
    top_k: int,
) -> dict[str, float | int | None]:
    doc_counts: list[int] = []
    matches_at_10: list[int] = []
    matches_at_top_k: list[int] = []
    first_match_ranks: list[int] = []
    queries_with_any_match = 0
    queries_with_all_relevant = 0
    queries_exact_top_k = 0

    for qid, relevant_dict in qrels.items():
        relevant_pmids = set(relevant_dict)
        documents = results_by_qid.get(qid, [])
        ranked_pmids = [item.pmid for item in documents[:top_k]]

        doc_count = len(ranked_pmids)
        doc_counts.append(doc_count)
        if doc_count == top_k:
            queries_exact_top_k += 1

        matched_at_10 = sum(1 for pmid in ranked_pmids[:10] if pmid in relevant_pmids)
        matched_at_top_k_value = sum(1 for pmid in ranked_pmids if pmid in relevant_pmids)
        matches_at_10.append(matched_at_10)
        matches_at_top_k.append(matched_at_top_k_value)

        if matched_at_top_k_value > 0:
            queries_with_any_match += 1
        if matched_at_top_k_value >= len(relevant_pmids):
            queries_with_all_relevant += 1

        for rank, pmid in enumerate(ranked_pmids, start=1):
            if pmid in relevant_pmids:
                first_match_ranks.append(rank)
                break

    total_queries = len(qrels)
    avg_docs = sum(doc_counts) / total_queries if total_queries else 0.0
    avg_matches_at_10 = sum(matches_at_10) / total_queries if total_queries else 0.0
    avg_matches_at_top_k = sum(matches_at_top_k) / total_queries if total_queries else 0.0

    return {
        "queries": total_queries,
        "avg_docs_returned": avg_docs,
        "min_docs_returned": min(doc_counts) if doc_counts else 0,
        "max_docs_returned": max(doc_counts) if doc_counts else 0,
        "queries_exact_topk": queries_exact_top_k,
        "queries_exact_topk_rate": queries_exact_top_k / total_queries if total_queries else 0.0,
        "avg_matches@10": avg_matches_at_10,
        f"avg_matches@{top_k}": avg_matches_at_top_k,
        f"hit_rate@{top_k}": queries_with_any_match / total_queries if total_queries else 0.0,
        f"all_relevant_rate@{top_k}": (
            queries_with_all_relevant / total_queries if total_queries else 0.0
        ),
        "avg_first_match_rank": (
            sum(first_match_ranks) / len(first_match_ranks) if first_match_ranks else None
        ),
        "median_first_match_rank": median(first_match_ranks) if first_match_ranks else None,
    }


async def _embed_question_map(
    questions: list[Question],
    *,
    batch_size: int,
    embed_url: str | None,
) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    for start in tqdm(range(0, len(questions), batch_size), desc="Embedding queries", unit="batch"):
        batch = questions[start : start + batch_size]
        matrix = await embed_queries_tei([question.body for question in batch], embed_url=embed_url)
        for question, embedding in zip(batch, matrix, strict=True):
            embeddings[question.id] = embedding
    return embeddings


def _rerank_items(
    query: str,
    documents: list[RetrievedItem],
    *,
    reranker: Context1Reranker,
    top_k: int,
) -> list[RetrievedItem]:
    corpus_docs = [
        CorpusDocument(
            pmid=document.pmid,
            text=document.full_text,
            token_count=0,
            score=document.score,
        )
        for document in documents
    ]
    reranked = reranker.score(query, corpus_docs)
    normalized = _normalize_items(
        [
            RetrievedItem(
                pmid=document.pmid,
                full_text=document.text,
                score=float(document.score),
            )
            for document in reranked
        ],
        top_k=top_k,
        fallback_lists=[documents],
    )
    return normalized[:top_k]


def _sanitize_method_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _serialize_run_row(qid: str, documents: list[RetrievedItem]) -> bytes:
    results_payload: list[dict[str, Any]] = []
    for document in documents:
        row: dict[str, Any] = {
            "pmid": document.pmid,
            "full_text": document.full_text,
            "score": document.score,
        }
        if document.justification:
            row["justification"] = document.justification
        results_payload.append(row)
    return orjson.dumps({"qid": qid, "results": results_payload})


def _serialize_run(results_by_qid: dict[str, list[RetrievedItem]]) -> bytes:
    rows = [_serialize_run_row(qid, documents) for qid, documents in results_by_qid.items()]
    return b"\n".join(rows) + (b"\n" if rows else b"")


def _deserialize_run(serialized: bytes) -> dict[str, list[RetrievedItem]]:
    if not serialized.strip():
        return {}

    results_by_qid: dict[str, list[RetrievedItem]] = {}
    for raw_line in serialized.splitlines():
        if not raw_line.strip():
            continue
        payload = orjson.loads(raw_line)
        qid = str(payload["qid"])
        documents: list[RetrievedItem] = []
        for document_payload in payload.get("results", []):
            documents.append(
                RetrievedItem(
                    pmid=str(document_payload["pmid"]),
                    full_text=str(document_payload.get("full_text", "")),
                    score=float(document_payload.get("score", 0.0)),
                    justification=str(document_payload.get("justification", "")),
                )
            )
        results_by_qid[qid] = documents
    return results_by_qid


def _load_run_file(path: PathType) -> dict[str, list[RetrievedItem]]:
    if not path.exists():
        return {}
    return _deserialize_run(path.read_bytes())


def _append_run_row(path: PathType, qid: str, documents: list[RetrievedItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_serialize_run_row(qid, documents))
        handle.write(b"\n")


def _checkpoint_file_path(checkpoint_dir: PathType, method_name: str) -> PathType:
    return checkpoint_dir / f"{_sanitize_method_name(method_name)}.jsonl"


def _checkpoint_manifest_path(checkpoint_dir: PathType) -> PathType:
    return checkpoint_dir / "manifest.json"


def _resolve_checkpoint_dir(
    *,
    checkpoint_dir: PathType | None,
    output_file: PathType | None,
    runs_dir: PathType | None,
) -> PathType | None:
    if checkpoint_dir is not None:
        return checkpoint_dir
    if runs_dir is not None:
        return runs_dir
    if output_file is not None:
        return output_file.parent / f".{output_file.stem}.benchmark-checkpoints"
    return None


def _collect_pending_question_ids(
    question_ids: list[str],
    results_by_method: Mapping[str, dict[str, list[RetrievedItem]]],
    *,
    required_methods: list[str],
) -> list[str]:
    pending: list[str] = []
    for qid in question_ids:
        if any(qid not in results_by_method[method_name] for method_name in required_methods):
            pending.append(qid)
    return pending


def _build_checkpoint_manifest(
    *,
    gold_file: PathType,
    question_ids: list[str],
    top_k: int,
    year: int | None,
    bm25_topk: int,
    semantic_topk: int,
    rrf_k: int,
    wsum_weights: tuple[float, float],
    tei_embed_url: str | None,
    reranker_model_names: list[str],
    reranker_batch_size: int,
    reranker_max_length: int,
    reranker_device: str,
    reranker_dtype: str,
    reranker_invert_scores: bool,
    include_context1: bool,
    context1_model_name: str,
    context1_vllm_base_url: str,
    context1_temperature: float,
    context1_max_completion_tokens: int,
    context1_max_turns: int,
    context1_num_rollouts: int,
    context1_rollout_seed: int | None,
    context1_reranker_model_name: str,
    context1_collection_name: str,
    method_names: list[str],
) -> dict[str, Any]:
    return {
        "gold_file": str(gold_file.resolve()),
        "question_ids": question_ids,
        "top_k": top_k,
        "year": year,
        "bm25_topk": bm25_topk,
        "semantic_topk": semantic_topk,
        "rrf_k": rrf_k,
        "wsum_weights": list(wsum_weights),
        "tei_embed_url": tei_embed_url,
        "reranker_model_names": reranker_model_names,
        "reranker_batch_size": reranker_batch_size,
        "reranker_max_length": reranker_max_length,
        "reranker_device": reranker_device,
        "reranker_dtype": reranker_dtype,
        "reranker_invert_scores": reranker_invert_scores,
        "include_context1": include_context1,
        "context1": {
            "model_name": context1_model_name,
            "vllm_base_url": context1_vllm_base_url,
            "temperature": context1_temperature,
            "max_completion_tokens": context1_max_completion_tokens,
            "max_turns": context1_max_turns,
            "num_rollouts": context1_num_rollouts,
            "rollout_seed": context1_rollout_seed,
            "reranker_model_name": context1_reranker_model_name,
            "collection_name": context1_collection_name,
        },
        "method_names": method_names,
    }


def _prepare_checkpoint_dir(
    checkpoint_dir: PathType,
    *,
    manifest: dict[str, Any],
    method_names: list[str],
    resume: bool,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _checkpoint_manifest_path(checkpoint_dir)
    checkpoint_files = [
        _checkpoint_file_path(checkpoint_dir, method_name) for method_name in method_names
    ]

    if resume:
        if manifest_path.exists():
            existing_manifest = orjson.loads(manifest_path.read_bytes())
            if existing_manifest != manifest:
                raise typer.BadParameter(
                    "Existing benchmark checkpoint does not match the current configuration. "
                    f"Use a different checkpoint directory or disable resume: {checkpoint_dir}"
                )
        elif any(path.exists() for path in checkpoint_files):
            raise typer.BadParameter(
                "Checkpoint files exist without a manifest, so resume is unsafe. "
                f"Remove {checkpoint_dir} or disable resume."
            )
    else:
        for path in checkpoint_files:
            if path.exists():
                path.unlink()

    manifest_path.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))


type JsonMetricValue = int | float | None


def _to_json_metric_value(value: JsonMetricValue | np.generic) -> JsonMetricValue:
    if isinstance(value, np.generic):
        scalar_value = value.item()
        if scalar_value is None or isinstance(scalar_value, int | float):
            return scalar_value
        raise TypeError(f"Unsupported numpy scalar metric value: {scalar_value!r}")
    return value


def _metric_to_float(value: JsonMetricValue) -> float:
    return 0.0 if value is None else float(value)


def _json_ready_metric_dict(
    metrics_dict: Mapping[str, JsonMetricValue | np.generic],
) -> dict[str, JsonMetricValue]:
    normalized: dict[str, JsonMetricValue] = {}
    for key, value in metrics_dict.items():
        normalized[key] = _to_json_metric_value(value)
    return normalized


def _print_summary(
    report: dict[str, dict[str, dict[str, float | int | None]]],
    *,
    top_k: int,
) -> None:
    console = Console()
    table = Table(title="Phase A Retrieval Benchmark")
    table.add_column("Method")
    table.add_column("Docs", justify="right")
    table.add_column(f"Match@{top_k}", justify="right")
    table.add_column(f"Hit@{top_k}", justify="right")
    table.add_column(f"AllRel@{top_k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Recall@100", justify="right")
    table.add_column("MAP-BioASQ@10", justify="right")

    for method_name, method_report in report.items():
        rough = method_report["rough_metrics"]
        retrieval = method_report["retrieval_metrics"]
        table.add_row(
            method_name,
            f"{_metric_to_float(rough['avg_docs_returned']):.1f}",
            f"{_metric_to_float(rough[f'avg_matches@{top_k}']):.2f}",
            f"{_metric_to_float(rough[f'hit_rate@{top_k}']):.3f}",
            f"{_metric_to_float(rough[f'all_relevant_rate@{top_k}']):.3f}",
            f"{_metric_to_float(retrieval.get('mrr', 0.0)):.3f}",
            f"{_metric_to_float(retrieval.get('recall@100', 0.0)):.3f}",
            f"{_metric_to_float(retrieval.get('map-bioasq@10', 0.0)):.3f}",
        )
    console.print(table)


async def _run_context1_benchmark(
    questions: list[Question],
    *,
    top_k: int,
    year: int | None,
    bm25_topk: int,
    semantic_topk: int,
    model_name: str,
    vllm_base_url: str,
    api_key: str,
    temperature: float,
    max_completion_tokens: int,
    max_turns: int,
    num_rollouts: int,
    rollout_seed: int | None,
    reranker_model_name: str,
    reranker_batch_size: int,
    reranker_max_length: int,
    reranker_device: str,
    reranker_invert_scores: bool,
    tei_embed_url: str | None,
    collection_name: str,
    checkpoint_path: PathType | None = None,
    checkpointed_qids: set[str] | None = None,
) -> dict[str, list[RetrievedItem]]:
    config = AgentConfig(
        model_name=model_name,
        vllm_base_url=vllm_base_url,
        api_key=api_key,
        max_turns=max_turns,
        bm25_topk=bm25_topk,
        dense_topk=semantic_topk,
        final_topk=top_k,
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
        search_candidate_pool_size=max(top_k, 100),
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

    try:
        status = await store.prepare_existing_corpus(year=year, ensure_bm25=False)
        if not status["qdrant_collection_exists"]:
            raise RuntimeError(
                "Qdrant collection "
                f"'{config.qdrant_collection}' is missing for Context1 benchmarking."
            )

        results_by_qid: dict[str, list[RetrievedItem]] = {}
        for question in tqdm(questions, desc="Running Context1", unit="q"):
            rollouts = await agent.run_rollouts(question.body)
            documents = await agent.aggregate_rollouts(rollouts)
            primary_items = [
                RetrievedItem(
                    pmid=document.pmid,
                    full_text=document.full_text,
                    score=float(document.score),
                    justification=document.justification,
                )
                for document in documents
            ]

            fallback_items: list[RetrievedItem] = []
            if len(primary_items) < top_k:
                hybrid_candidates = await store.hybrid_search_candidates(
                    question.body,
                    bm25_topk=max(bm25_topk, top_k),
                    dense_topk=max(semantic_topk, top_k),
                    year=year,
                )
                reranked_candidates = reranker.score(
                    question.body,
                    hybrid_candidates[: max(top_k, config.search_candidate_pool_size)],
                )
                fallback_items = [
                    RetrievedItem(
                        pmid=document.pmid,
                        full_text=document.text,
                        score=float(document.score),
                    )
                    for document in reranked_candidates
                ]

            normalized = _limit_top_k(
                _normalize_items(
                    primary_items,
                    top_k=top_k,
                    fallback_lists=[fallback_items],
                ),
                top_k=top_k,
            )
            results_by_qid[question.id] = normalized
            if (
                checkpoint_path is not None
                and checkpointed_qids is not None
                and question.id not in checkpointed_qids
            ):
                _append_run_row(checkpoint_path, question.id, normalized)
                checkpointed_qids.add(question.id)
        return results_by_qid
    finally:
        await backend.close()
        await store.close()


async def _benchmark(
    gold_file: PathType,
    *,
    output_file: PathType | None,
    runs_dir: PathType | None,
    checkpoint_dir: PathType | None,
    resume: bool,
    year: int | None,
    limit: int | None,
    top_k: int,
    bm25_topk: int,
    semantic_topk: int,
    rrf_k: int,
    wsum_weights: tuple[float, float],
    metrics: list[str],
    embed_batch_size: int,
    tei_embed_url: str | None,
    reranker_model_names: list[str],
    reranker_batch_size: int,
    reranker_max_length: int,
    reranker_device: str,
    reranker_dtype: str,
    reranker_invert_scores: bool,
    include_context1: bool,
    context1_model_name: str,
    context1_vllm_base_url: str,
    context1_api_key: str,
    context1_temperature: float,
    context1_max_completion_tokens: int,
    context1_max_turns: int,
    context1_num_rollouts: int,
    context1_rollout_seed: int | None,
    context1_reranker_model_name: str,
    context1_collection_name: str,
) -> None:
    if top_k != 100:
        raise typer.BadParameter("This benchmark is currently fixed to top-k=100 for all methods.")
    if bm25_topk < top_k:
        raise typer.BadParameter("bm25-topk must be at least 100.")
    if semantic_topk < top_k:
        raise typer.BadParameter("semantic-topk must be at least 100.")
    if embed_batch_size <= 0:
        raise typer.BadParameter("embed-batch-size must be greater than zero.")

    questions, qrels = _load_gold_questions(gold_file, limit=limit)

    rerankers: list[tuple[str, Context1Reranker]] = [
        (
            short_model_name(model_name).replace("/", "-"),
            Context1Reranker(
                model_name,
                batch_size=reranker_batch_size,
                max_length=reranker_max_length,
                device=reranker_device,
                dtype=reranker_dtype,
                invert_scores=reranker_invert_scores,
            ),
        )
        for model_name in reranker_model_names
    ]

    results_by_method: dict[str, dict[str, list[RetrievedItem]]] = {
        "bm25": {},
        "hybrid_rrf": {},
        "hybrid_wsum": {},
    }
    for reranker_name, _ in rerankers:
        results_by_method[f"hybrid_rrf_rerank::{reranker_name}"] = {}
        results_by_method[f"hybrid_wsum_rerank::{reranker_name}"] = {}
    if include_context1:
        results_by_method["context1"] = {}

    all_method_names = list(results_by_method)
    baseline_method_names = [
        method_name for method_name in all_method_names if method_name != "context1"
    ]
    question_ids = [question.id for question in questions]

    resolved_checkpoint_dir = _resolve_checkpoint_dir(
        checkpoint_dir=checkpoint_dir,
        output_file=output_file,
        runs_dir=runs_dir,
    )
    checkpoint_paths: dict[str, PathType] = {}
    checkpointed_qids: dict[str, set[str]] = {
        method_name: set() for method_name in all_method_names
    }
    if resolved_checkpoint_dir is not None:
        manifest = _build_checkpoint_manifest(
            gold_file=gold_file,
            question_ids=question_ids,
            top_k=top_k,
            year=year,
            bm25_topk=bm25_topk,
            semantic_topk=semantic_topk,
            rrf_k=rrf_k,
            wsum_weights=wsum_weights,
            tei_embed_url=tei_embed_url,
            reranker_model_names=reranker_model_names,
            reranker_batch_size=reranker_batch_size,
            reranker_max_length=reranker_max_length,
            reranker_device=reranker_device,
            reranker_dtype=reranker_dtype,
            reranker_invert_scores=reranker_invert_scores,
            include_context1=include_context1,
            context1_model_name=context1_model_name,
            context1_vllm_base_url=context1_vllm_base_url,
            context1_temperature=context1_temperature,
            context1_max_completion_tokens=context1_max_completion_tokens,
            context1_max_turns=context1_max_turns,
            context1_num_rollouts=context1_num_rollouts,
            context1_rollout_seed=context1_rollout_seed,
            context1_reranker_model_name=context1_reranker_model_name,
            context1_collection_name=context1_collection_name,
            method_names=all_method_names,
        )
        _prepare_checkpoint_dir(
            resolved_checkpoint_dir,
            manifest=manifest,
            method_names=all_method_names,
            resume=resume,
        )
        for method_name in all_method_names:
            checkpoint_path = _checkpoint_file_path(resolved_checkpoint_dir, method_name)
            checkpoint_paths[method_name] = checkpoint_path
            loaded_results = _load_run_file(checkpoint_path)
            results_by_method[method_name].update(loaded_results)
            checkpointed_qids[method_name] = set(loaded_results)

    pending_baseline_qids = set(
        _collect_pending_question_ids(
            question_ids,
            results_by_method,
            required_methods=baseline_method_names,
        )
    )
    pending_baseline_questions = [
        question for question in questions if question.id in pending_baseline_qids
    ]
    if resolved_checkpoint_dir is not None and len(pending_baseline_questions) != len(questions):
        completed_baseline_questions = len(questions) - len(pending_baseline_questions)
        typer.echo(
            "Resuming retrieval baselines from "
            f"{resolved_checkpoint_dir} "
            f"({completed_baseline_questions}/{len(questions)} questions already saved)."
        )

    embeddings: dict[str, np.ndarray] = {}
    if pending_baseline_questions:
        embeddings = await _embed_question_map(
            pending_baseline_questions,
            batch_size=embed_batch_size,
            embed_url=tei_embed_url,
        )

    try:
        for question in tqdm(
            pending_baseline_questions,
            desc="Running retrieval baselines",
            unit="q",
        ):
            qid = question.id
            query = question.body
            query_embedding = embeddings[qid]

            bm25_docs, dense_docs = await asyncio.gather(
                bm25_search(query, topk=bm25_topk, year=year),
                semantic_search(query_embedding, topk=semantic_topk, year=year),
            )

            bm25_items = _limit_top_k(
                _normalize_items(_to_items(bm25_docs), top_k=top_k),
                top_k=top_k,
            )
            dense_items = _limit_top_k(
                _normalize_items(_to_items(dense_docs), top_k=top_k),
                top_k=top_k,
            )

            rrf_items = _limit_top_k(
                _normalize_items(
                    _to_items(
                        fuse_retrieval_lists_rrf(
                            qid,
                            [("bm25", bm25_docs), ("dense", dense_docs)],
                            rrf_k=rrf_k,
                        )
                    ),
                    top_k=top_k,
                    fallback_lists=[bm25_items, dense_items],
                ),
                top_k=top_k,
            )
            wsum_items = _limit_top_k(
                _normalize_items(
                    _to_items(
                        fuse_retrieval_lists_wsum(
                            qid,
                            [("bm25", bm25_docs), ("dense", dense_docs)],
                            weights=wsum_weights,
                        )
                    ),
                    top_k=top_k,
                    fallback_lists=[bm25_items, dense_items],
                ),
                top_k=top_k,
            )

            results_by_method["bm25"][qid] = bm25_items
            results_by_method["hybrid_rrf"][qid] = rrf_items
            results_by_method["hybrid_wsum"][qid] = wsum_items

            for reranker_name, reranker in rerankers:
                results_by_method[f"hybrid_rrf_rerank::{reranker_name}"][qid] = _limit_top_k(
                    _rerank_items(query, rrf_items, reranker=reranker, top_k=top_k),
                    top_k=top_k,
                )
                results_by_method[f"hybrid_wsum_rerank::{reranker_name}"][qid] = _limit_top_k(
                    _rerank_items(query, wsum_items, reranker=reranker, top_k=top_k),
                    top_k=top_k,
                )

            if resolved_checkpoint_dir is not None:
                for method_name in baseline_method_names:
                    if qid in checkpointed_qids[method_name]:
                        continue
                    _append_run_row(
                        checkpoint_paths[method_name],
                        qid,
                        results_by_method[method_name][qid],
                    )
                    checkpointed_qids[method_name].add(qid)

        if include_context1:
            pending_context1_qids = set(
                _collect_pending_question_ids(
                    question_ids,
                    results_by_method,
                    required_methods=["context1"],
                )
            )
            pending_context1_questions = [
                question for question in questions if question.id in pending_context1_qids
            ]
            if resolved_checkpoint_dir is not None and len(pending_context1_questions) != len(
                questions
            ):
                completed_context1_questions = len(questions) - len(pending_context1_questions)
                typer.echo(
                    "Resuming Context1 from "
                    f"{resolved_checkpoint_dir} "
                    f"({completed_context1_questions}/{len(questions)} questions already saved)."
                )
            if pending_context1_questions:
                results_by_method["context1"].update(
                    await _run_context1_benchmark(
                        pending_context1_questions,
                        top_k=top_k,
                        year=year,
                        bm25_topk=bm25_topk,
                        semantic_topk=semantic_topk,
                        model_name=context1_model_name,
                        vllm_base_url=context1_vllm_base_url,
                        api_key=context1_api_key,
                        temperature=context1_temperature,
                        max_completion_tokens=context1_max_completion_tokens,
                        max_turns=context1_max_turns,
                        num_rollouts=context1_num_rollouts,
                        rollout_seed=context1_rollout_seed,
                        reranker_model_name=context1_reranker_model_name,
                        reranker_batch_size=reranker_batch_size,
                        reranker_max_length=reranker_max_length,
                        reranker_device=reranker_device,
                        reranker_invert_scores=reranker_invert_scores,
                        tei_embed_url=tei_embed_url,
                        collection_name=context1_collection_name,
                        checkpoint_path=checkpoint_paths.get("context1"),
                        checkpointed_qids=checkpointed_qids["context1"],
                    )
                )
    finally:
        await close_pool()
        await close_qdrant_client()

    report_methods: dict[str, dict[str, dict[str, float | int | None]]] = {}
    saved_run_files: dict[str, str] = {}
    for method_name, method_results in results_by_method.items():
        run_dict = _build_run_dict(method_results)
        report_methods[method_name] = {
            "retrieval_metrics": _json_ready_metric_dict(
                evaluate_retrieval_run(run_dict, qrels, metrics=metrics)["total"],
            ),
            "rough_metrics": _json_ready_metric_dict(
                _compute_rough_metrics(method_results, qrels, top_k=top_k),
            ),
        }

        if runs_dir is not None:
            runs_dir.mkdir(parents=True, exist_ok=True)
            run_path = runs_dir / f"{_sanitize_method_name(method_name)}.jsonl"
            run_path.write_bytes(_serialize_run(method_results))
            saved_run_files[method_name] = str(run_path)

    _print_summary(report_methods, top_k=top_k)

    report: dict[str, Any] = {
        "metadata": {
            "gold_file": str(gold_file),
            "questions": len(questions),
            "top_k": top_k,
            "year": year,
            "bm25_topk": bm25_topk,
            "semantic_topk": semantic_topk,
            "rrf_k": rrf_k,
            "wsum_weights": list(wsum_weights),
            "metrics": metrics,
            "reranker_models": reranker_model_names,
            "include_context1": include_context1,
            "checkpoint_dir": str(resolved_checkpoint_dir) if resolved_checkpoint_dir else None,
            "resume": resume,
        },
        "methods": report_methods,
        "run_files": saved_run_files,
    }

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
        typer.echo(f"Saved benchmark report to {output_file}")


@app.command("benchmark-retrieval")
@typer_async
async def benchmark_retrieval_command(
    gold_file: Annotated[
        PathType,
        typer.Option(
            ..., "--gold-file", exists=True, help="BioASQ JSON or JSONL with gold documents."
        ),
    ],
    output_file: Annotated[
        PathType | None,
        typer.Option("--output", "-o", help="Optional JSON report output path."),
    ] = None,
    runs_dir: Annotated[
        PathType | None,
        typer.Option(help="Optional directory to save each method's retrieved documents as JSONL."),
    ] = None,
    checkpoint_dir: Annotated[
        PathType | None,
        typer.Option(
            help=(
                "Optional directory for incremental checkpoint JSONL files. "
                "Defaults to --runs-dir, "
                "or to a hidden sibling directory next to --output."
            )
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Resume from matching checkpoint files when available.",
        ),
    ] = True,
    year: Annotated[int | None, typer.Option(help="Optional BioASQ baseline year filter.")] = None,
    limit: Annotated[
        int | None, typer.Option(help="Optional limit on evaluated questions.")
    ] = None,
    top_k: Annotated[int, typer.Option("--top-k", help="Final documents per method.")] = 100,
    bm25_topk: Annotated[int, typer.Option(help="BM25 retrieval depth.")] = 200,
    semantic_topk: Annotated[int, typer.Option(help="Dense retrieval depth.")] = 200,
    rrf_k: Annotated[int, typer.Option(help="RRF k parameter.")] = 60,
    wsum_weights_raw: Annotated[
        str,
        typer.Option(
            "--wsum-weights", help="Comma-separated BM25,Dense weights for weighted sum fusion."
        ),
    ] = "0.6,0.4",
    metrics_raw: Annotated[
        str,
        typer.Option("--metrics", help="Comma-separated ranx metrics to report."),
    ] = ",".join(DEFAULT_RETRIEVAL_METRICS),
    embed_batch_size: Annotated[
        int, typer.Option(help="Batch size for query embedding requests.")
    ] = 32,
    tei_embed_url: Annotated[
        str | None, typer.Option(help="Optional TEI embeddings endpoint.")
    ] = None,
    reranker_models_raw: Annotated[
        str | None,
        typer.Option(
            "--reranker-models",
            help="Optional comma-separated reranker model names or local paths.",
        ),
    ] = None,
    reranker_batch_size: Annotated[int, typer.Option(help="Reranker batch size.")] = 16,
    reranker_max_length: Annotated[int, typer.Option(help="Reranker max sequence length.")] = 512,
    reranker_device: Annotated[str, typer.Option(help="Reranker device.")] = "cuda",
    reranker_dtype: Annotated[
        str, typer.Option(help="Reranker dtype: float32, bfloat16, or float16.")
    ] = "bfloat16",
    reranker_invert_scores: Annotated[
        bool,
        typer.Option(help="Invert reranker scores before sorting."),
    ] = False,
    include_context1: Annotated[
        bool,
        typer.Option("--with-context1", help="Also benchmark the Context1 retrieval harness."),
    ] = False,
    context1_model_name: Annotated[
        str,
        typer.Option(help="Served model name on the Context1 vLLM server."),
    ] = "chromadb/context-1",
    context1_vllm_base_url: Annotated[
        str,
        typer.Option(help="Base URL of the Context1 vLLM OpenAI-compatible server."),
    ] = "http://127.0.0.1:8000",
    context1_api_key: Annotated[
        str,
        typer.Option(help="API key for the Context1 vLLM server."),
    ] = "EMPTY",
    context1_temperature: Annotated[
        float, typer.Option(help="Context1 sampling temperature.")
    ] = 0.2,
    context1_max_completion_tokens: Annotated[
        int,
        typer.Option(help="Maximum completion tokens per Context1 model turn."),
    ] = 4096,
    context1_max_turns: Annotated[
        int, typer.Option(help="Maximum Context1 tool-calling turns.")
    ] = 12,
    context1_num_rollouts: Annotated[
        int, typer.Option(help="Number of Context1 rollouts per question.")
    ] = 1,
    context1_rollout_seed: Annotated[
        int | None,
        typer.Option(help="Optional base seed for Context1 rollouts."),
    ] = None,
    context1_reranker_model_name: Annotated[
        str,
        typer.Option(help="Cross-encoder reranker used inside Context1."),
    ] = "BAAI/bge-reranker-v2-m3",
    context1_collection_name: Annotated[
        str,
        typer.Option(help="Qdrant collection used by Context1."),
    ] = "articles",
) -> None:
    """Benchmark BM25, hybrid fusion, optional reranking, and Context1 at top-100."""

    await _benchmark(
        gold_file,
        output_file=output_file,
        runs_dir=runs_dir,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        year=year,
        limit=limit,
        top_k=top_k,
        bm25_topk=bm25_topk,
        semantic_topk=semantic_topk,
        rrf_k=rrf_k,
        wsum_weights=_parse_weights(wsum_weights_raw),
        metrics=_split_csv(metrics_raw) or list(DEFAULT_RETRIEVAL_METRICS),
        embed_batch_size=embed_batch_size,
        tei_embed_url=tei_embed_url,
        reranker_model_names=_split_csv(reranker_models_raw),
        reranker_batch_size=reranker_batch_size,
        reranker_max_length=reranker_max_length,
        reranker_device=reranker_device,
        reranker_dtype=reranker_dtype,
        reranker_invert_scores=reranker_invert_scores,
        include_context1=include_context1,
        context1_model_name=context1_model_name,
        context1_vllm_base_url=context1_vllm_base_url,
        context1_api_key=context1_api_key,
        context1_temperature=context1_temperature,
        context1_max_completion_tokens=context1_max_completion_tokens,
        context1_max_turns=context1_max_turns,
        context1_num_rollouts=context1_num_rollouts,
        context1_rollout_seed=context1_rollout_seed,
        context1_reranker_model_name=context1_reranker_model_name,
        context1_collection_name=context1_collection_name,
    )


if __name__ == "__main__":
    app()
