"""Evaluate BM25, Dense (Qdrant), and Hybrid (BM25+Dense RRF) retrieval on 13B golden data.

Generates predictions for each method and evaluates using ranx metrics (nDCG, MRR, Recall, MAP).

Usage::

    python scripts/evaluate_retrieval.py evaluate                      # all 13B batches
    python scripts/evaluate_retrieval.py evaluate --batches 1          # just batch 1
    python scripts/evaluate_retrieval.py evaluate --batches 1 2 --topk 200
    python scripts/evaluate_retrieval.py evaluate --bm25-weight 0.7    # weighted fusion
    python scripts/evaluate_retrieval.py sweep                         # grid search
    python scripts/evaluate_retrieval.py sweep --output sweep.json     # save sweep results
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from tqdm.asyncio import tqdm

from bioasq.common import PROJECT_DATA_DIR
from bioasq.common.metrics import DEFAULT_RETRIEVAL_METRICS, evaluate_retrieval_run
from bioasq.common.types import DocumentWithScore
from bioasq.common.utils import typer_async
from bioasq.data.database import bm25_search, close_pool, close_qdrant_client, semantic_search
from bioasq.phase_a.retrieval.fusion import fuse_retrieval_lists_rrf, fuse_retrieval_lists_wsum
from bioasq.phase_a.retrieval.query_encoder import embed_queries_tei

app = typer.Typer(no_args_is_help=True)

VAL_DATA_DIR = PROJECT_DATA_DIR / "val_data"

_PUBMED_URL_RE = re.compile(r"(?:https?://www\.ncbi\.nlm\.nih\.gov/pubmed/)(\d+)")


def _extract_pmid(url_or_id: str) -> str:
    """Extract bare PMID from a BioASQ document URL or return as-is."""
    m = _PUBMED_URL_RE.search(url_or_id)
    return m.group(1) if m else url_or_id


def _load_golden_files(
    batches: list[int],
) -> tuple[
    dict[str, str],  # qid -> body
    dict[str, dict[str, int]],  # qrels: qid -> {pmid: 1}
    dict[str, list[str]],  # per_file: filename -> [qid, ...]
]:
    """Load golden BioASQ 13B JSON files and build qrels."""
    all_questions: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    per_file: dict[str, list[str]] = {}

    for batch_num in batches:
        filename = f"13B{batch_num}_golden.json"
        path = VAL_DATA_DIR / filename
        if not path.exists():
            typer.echo(f"Warning: {path} not found, skipping.", err=True)
            continue

        data = json.loads(path.read_text())
        batch_qids: list[str] = []
        for q in data["questions"]:
            qid = str(q["id"])
            body = str(q["body"])
            docs = q.get("documents", [])
            if not docs:
                continue
            all_questions[qid] = body
            qrels[qid] = {_extract_pmid(d): 1 for d in docs}
            batch_qids.append(qid)

        per_file[filename] = batch_qids

    return all_questions, qrels, per_file


def _run_dict_from_docs(docs: list[DocumentWithScore]) -> dict[str, float]:
    """Convert list of DocumentWithScore to {pmid: score}."""
    return {d.pmid: float(d.score) for d in docs}


async def _retrieve_raw(
    questions: dict[str, str],
    bm25_topk: int,
    dense_topk: int,
    embed_url: str | None,
) -> tuple[
    list[str],  # qids
    dict[str, list[DocumentWithScore]],  # bm25 raw
    dict[str, list[DocumentWithScore]],  # dense raw
]:
    """Fetch raw BM25 and Dense results (no fusion). Reusable for sweep."""
    qids = list(questions.keys())
    bodies = [questions[qid] for qid in qids]
    typer.echo(f"Encoding {len(bodies)} queries via TEI (batch size 32)...")
    embedding_chunks = []
    for start in range(0, len(bodies), 32):
        chunk = bodies[start : start + 32]
        embedding_chunks.append(await embed_queries_tei(chunk, embed_url=embed_url))
    embeddings = np.concatenate(embedding_chunks, axis=0)

    bm25_raw: dict[str, list[DocumentWithScore]] = {}
    dense_raw: dict[str, list[DocumentWithScore]] = {}

    for i, qid in enumerate(tqdm(qids, desc="Retrieving", unit="q")):
        body = bodies[i]
        query_emb = embeddings[i]
        bm25_docs, dense_docs = await asyncio.gather(
            bm25_search(body, topk=bm25_topk),
            semantic_search(query_emb, topk=dense_topk),
        )
        bm25_raw[qid] = bm25_docs
        dense_raw[qid] = dense_docs

    return qids, bm25_raw, dense_raw


def _fuse_runs(
    qids: list[str],
    bm25_raw: dict[str, list[DocumentWithScore]],
    dense_raw: dict[str, list[DocumentWithScore]],
    *,
    rrf_k: int = 60,
    bm25_weight: float | None = None,
) -> tuple[
    dict[str, dict[str, float]],  # bm25 run
    dict[str, dict[str, float]],  # dense run
    dict[str, dict[str, float]],  # hybrid run
]:
    """Fuse pre-fetched raw results. Cheap to re-run with different params."""
    bm25_run: dict[str, dict[str, float]] = {}
    dense_run: dict[str, dict[str, float]] = {}
    hybrid_run: dict[str, dict[str, float]] = {}

    for qid in qids:
        bm25_docs = bm25_raw[qid]
        dense_docs = dense_raw[qid]

        bm25_run[qid] = _run_dict_from_docs(bm25_docs)
        dense_run[qid] = _run_dict_from_docs(dense_docs)

        named = [("bm25", bm25_docs), ("dense", dense_docs)]
        if bm25_weight is not None:
            fused = fuse_retrieval_lists_wsum(
                qid,
                named,
                weights=[bm25_weight, 1.0 - bm25_weight],
            )
        else:
            fused = fuse_retrieval_lists_rrf(qid, named, rrf_k=rrf_k)
        hybrid_run[qid] = _run_dict_from_docs(fused)

    return bm25_run, dense_run, hybrid_run


def _print_results(
    all_results: dict[str, dict[str, dict[str, float]]],
    per_file: dict[str, list[str]],
) -> None:
    """Pretty-print a comparison table using Rich."""
    console = Console()

    # Total results table
    table = Table(title="Retrieval Evaluation — Total", show_lines=True)
    table.add_column("Metric", style="bold")
    for method_name in all_results:
        table.add_column(method_name, justify="right")

    metrics = list(next(iter(all_results.values()))["total"].keys())
    for metric in metrics:
        row = [metric]
        for method_name in all_results:
            val = all_results[method_name]["total"].get(metric, 0.0)
            row.append(f"{val:.4f}")
        table.add_row(*row)

    console.print(table)

    # Per-file breakdown
    if per_file and len(per_file) > 1:
        for filename in per_file:
            sub_table = Table(title=f"Per-file: {filename}", show_lines=True)
            sub_table.add_column("Metric", style="bold")
            for method_name in all_results:
                sub_table.add_column(method_name, justify="right")

            for metric in metrics:
                row = [metric]
                for method_name in all_results:
                    val = all_results[method_name].get(filename, {}).get(metric, 0.0)
                    row.append(f"{val:.4f}")
                sub_table.add_row(*row)

            console.print(sub_table)


@app.command()
@typer_async
async def evaluate(
    batches: Annotated[
        list[int] | None,
        typer.Option("--batches", "-b", help="13B batch numbers to evaluate (default: 1 2 3 4)."),
    ] = None,
    topk: Annotated[
        int,
        typer.Option("--topk", "-k", help="Number of documents to retrieve per method."),
    ] = 100,
    bm25_topk: Annotated[
        int | None,
        typer.Option("--bm25-topk", help="BM25 candidate depth (overrides --topk for BM25)."),
    ] = None,
    dense_topk: Annotated[
        int | None,
        typer.Option("--dense-topk", help="Dense candidate depth (overrides --topk for dense)."),
    ] = None,
    rrf_k: Annotated[
        int,
        typer.Option("--rrf-k", help="RRF smoothing constant."),
    ] = 60,
    bm25_weight: Annotated[
        float | None,
        typer.Option(
            "--bm25-weight",
            "-w",
            help="BM25 weight for weighted-sum fusion (0-1). Dense gets 1-w. Omit to use RRF.",
        ),
    ] = None,
    tei_url: Annotated[
        str | None,
        typer.Option("--tei-url", help="TEI embed endpoint URL."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Save detailed JSON results to file."),
    ] = "eval.json",
    metrics: Annotated[
        list[str] | None,
        typer.Option("--metrics", "-m", help="Metrics to compute (default: standard set)."),
    ] = None,
) -> None:
    """Evaluate BM25, Dense, and Hybrid retrieval against 13B golden data."""
    used_batches = batches or [1, 2, 3, 4]
    used_metrics = metrics or DEFAULT_RETRIEVAL_METRICS
    eff_bm25_topk = bm25_topk or topk
    eff_dense_topk = dense_topk or topk

    typer.echo(f"Loading golden data for batches: {used_batches}")
    questions, qrels, per_file = _load_golden_files(used_batches)

    if not questions:
        typer.echo("No questions found. Check that golden files exist.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(questions)} questions with ground-truth documents.")
    if bm25_weight is not None:
        typer.echo(f"Fusion: weighted-sum  bm25_weight={bm25_weight}")
    else:
        typer.echo(f"Fusion: RRF  k={rrf_k}")

    try:
        qids, bm25_raw, dense_raw = await _retrieve_raw(
            questions,
            bm25_topk=eff_bm25_topk,
            dense_topk=eff_dense_topk,
            embed_url=tei_url,
        )

        bm25_run, dense_run, hybrid_run = _fuse_runs(
            qids,
            bm25_raw,
            dense_raw,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
        )

        fusion_label = (
            f"Hybrid (wsum w={bm25_weight})"
            if bm25_weight is not None
            else "Hybrid (BM25+Dense RRF)"
        )
        runs = {
            "BM25": bm25_run,
            "Dense (Qdrant)": dense_run,
            fusion_label: hybrid_run,
        }

        typer.echo("\nEvaluating with ranx...")
        all_results: dict[str, dict[str, dict[str, float]]] = {}
        for method_name, run_dict in runs.items():
            all_results[method_name] = evaluate_retrieval_run(
                run_dict,
                qrels,
                metrics=used_metrics,
                per_file_results=per_file if len(per_file) > 1 else None,
            )

        _print_results(all_results, per_file)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(all_results, indent=2))
            typer.echo(f"\nResults saved to {output}")

    finally:
        await close_pool()
        await close_qdrant_client()


@app.command()
@typer_async
async def sweep(
    batches: Annotated[
        list[int] | None,
        typer.Option("--batches", "-b", help="13B batch numbers (default: 1 2 3 4)."),
    ] = None,
    topks: Annotated[
        list[int] | None,
        typer.Option("--topk", "-k", help="Topk values to sweep (retrieves once at max)."),
    ] = None,
    rrf_ks: Annotated[
        list[int] | None,
        typer.Option("--rrf-k", help="RRF k values to sweep."),
    ] = None,
    bm25_weights: Annotated[
        list[float] | None,
        typer.Option("--bm25-weight", "-w", help="BM25 weights (0-1) to sweep with wsum fusion."),
    ] = None,
    optimize: Annotated[
        str,
        typer.Option("--optimize", help="Metric to rank configs by."),
    ] = "map-bioasq@10",
    tei_url: Annotated[
        str | None,
        typer.Option("--tei-url", help="TEI embed endpoint URL."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Save sweep results JSON."),
    ] = "sweep.json",
) -> None:
    """Grid-search over topk, RRF k, and/or BM25 weight to find the best fusion config."""
    used_batches = batches or [1, 2, 3, 4]
    used_metrics = DEFAULT_RETRIEVAL_METRICS
    used_topks = topks or [100]
    max_topk = max(used_topks)

    # Build grid: list of (label, topk, rrf_k | None, bm25_weight | None)
    grid: list[tuple[str, int, int | None, float | None]] = []
    for tk in used_topks:
        for k in rrf_ks or [20, 40, 60, 80, 100]:
            grid.append((f"topk={tk} RRF k={k}", tk, k, None))
        for w in bm25_weights or [0.5, 0.6, 0.7, 0.8, 0.9]:
            grid.append((f"topk={tk} wsum w={w}", tk, None, w))

    typer.echo(f"Loading golden data for batches: {used_batches}")
    questions, qrels, _per_file = _load_golden_files(used_batches)
    if not questions:
        typer.echo("No questions found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(questions)} questions. Retrieving once at max topk={max_topk}...")

    try:
        qids, bm25_raw, dense_raw = await _retrieve_raw(
            questions,
            bm25_topk=max_topk,
            dense_topk=max_topk,
            embed_url=tei_url,
        )

        console = Console()
        sweep_results: list[dict] = []

        for label, tk, k, w in grid:
            # Truncate raw lists to current topk
            bm25_trunc = {q: docs[:tk] for q, docs in bm25_raw.items()}
            dense_trunc = {q: docs[:tk] for q, docs in dense_raw.items()}

            _, _, hybrid_run = _fuse_runs(
                qids,
                bm25_trunc,
                dense_trunc,
                rrf_k=k or 60,
                bm25_weight=w,
            )
            scores = evaluate_retrieval_run(
                hybrid_run,
                qrels,
                metrics=used_metrics,
            )["total"]
            sweep_results.append(
                {
                    "config": label,
                    "topk": tk,
                    "rrf_k": k,
                    "bm25_weight": w,
                    **scores,
                }
            )

        # Sort by optimize metric
        sweep_results.sort(key=lambda r: r.get(optimize, 0.0), reverse=True)

        # Print table
        table = Table(title=f"Sweep Results (sorted by {optimize})", show_lines=True)
        table.add_column("Rank", style="bold", width=4)
        table.add_column("Config", style="bold")
        for m in used_metrics:
            table.add_column(m, justify="right")

        for rank, row in enumerate(sweep_results, 1):
            cells = [str(rank), row["config"]]
            cells.extend(f"{row.get(m, 0.0):.4f}" for m in used_metrics)
            table.add_row(*cells)

        console.print(table)

        best = sweep_results[0]
        typer.echo(f"\nBest config: {best['config']}  {optimize}={best[optimize]:.4f}")

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(sweep_results, indent=2))
            typer.echo(f"Sweep results saved to {output}")

    finally:
        await close_pool()
        await close_qdrant_client()


def _oracle_union_run(
    qids: list[str],
    bm25_raw: dict[str, list[DocumentWithScore]],
    dense_raw: dict[str, list[DocumentWithScore]],
) -> dict[str, dict[str, float]]:
    """Build an 'oracle union' run: union of both sets, scored by max rank position."""
    union_run: dict[str, dict[str, float]] = {}
    for qid in qids:
        merged: dict[str, float] = {}
        # Give each doc a score = 1/(rank+1) so higher-ranked docs score more;
        # take the max across the two lists.
        for docs in (bm25_raw[qid], dense_raw[qid]):
            for rank, d in enumerate(docs):
                score = 1.0 / (rank + 1)
                if d.pmid not in merged or score > merged[d.pmid]:
                    merged[d.pmid] = score
        union_run[qid] = merged
    return union_run


@app.command("recall-analysis")
@typer_async
async def recall_analysis(
    batches: Annotated[
        list[int] | None,
        typer.Option("--batches", "-b", help="13B batch numbers (default: 1 2 3 4)."),
    ] = None,
    topk: Annotated[
        int,
        typer.Option("--topk", "-k", help="Retrieval depth per method (must be >= max cutoff)."),
    ] = 1000,
    cutoffs: Annotated[
        list[int] | None,
        typer.Option("--cutoff", "-c", help="Recall cutoffs to evaluate."),
    ] = None,
    tei_url: Annotated[
        str | None,
        typer.Option("--tei-url", help="TEI embed endpoint URL."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Save results JSON."),
    ] = "recall_analysis.json",
) -> None:
    """Recall at multiple cutoffs for BM25, Dense, Oracle Union, and Hybrid RRF."""
    used_batches = batches or [1, 2, 3, 4]
    used_cutoffs = cutoffs or [10, 50, 100, 200, 500, 1000]
    recall_metrics = [f"recall@{c}" for c in used_cutoffs]

    typer.echo(f"Loading golden data for batches: {used_batches}")
    questions, qrels, _per_file = _load_golden_files(used_batches)
    if not questions:
        typer.echo("No questions found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(questions)} questions. Retrieving topk={topk}...")

    try:
        qids, bm25_raw, dense_raw = await _retrieve_raw(
            questions,
            bm25_topk=topk,
            dense_topk=topk,
            embed_url=tei_url,
        )

        bm25_run, dense_run, hybrid_run = _fuse_runs(
            qids,
            bm25_raw,
            dense_raw,
            rrf_k=60,
        )
        union_run = _oracle_union_run(qids, bm25_raw, dense_raw)

        runs = {
            "BM25": bm25_run,
            "Dense": dense_run,
            "Hybrid (RRF k=60)": hybrid_run,
            "Oracle Union": union_run,
        }

        typer.echo("\nEvaluating recall...")
        all_results: dict[str, dict[str, float]] = {}
        for name, run_dict in runs.items():
            scores = evaluate_retrieval_run(run_dict, qrels, metrics=recall_metrics)["total"]
            all_results[name] = scores

        # Print table
        console = Console()
        table = Table(title="Recall Analysis", show_lines=True)
        table.add_column("Cutoff", style="bold")
        for name in runs:
            table.add_column(name, justify="right")

        for m in recall_metrics:
            row = [m]
            for name in runs:
                val = all_results[name].get(m, 0.0)
                row.append(f"{val:.4f}")
            table.add_row(*row)

        # Also show pool sizes
        table2 = Table(title="Pool Statistics", show_lines=True)
        table2.add_column("Stat", style="bold")
        table2.add_column("Value", justify="right")

        bm25_unique = set()
        dense_unique = set()
        for qid in qids:
            bm25_unique.update(bm25_run[qid].keys())
            dense_unique.update(dense_run[qid].keys())
        union_unique = bm25_unique | dense_unique
        overlap = bm25_unique & dense_unique

        table2.add_row("Total unique BM25 docs", str(len(bm25_unique)))
        table2.add_row("Total unique Dense docs", str(len(dense_unique)))
        table2.add_row("Union (BM25 | Dense)", str(len(union_unique)))
        table2.add_row("Overlap (BM25 & Dense)", str(len(overlap)))
        table2.add_row("Overlap %", f"{100 * len(overlap) / len(union_unique):.1f}%")

        avg_bm25 = np.mean([len(bm25_raw[q]) for q in qids])
        avg_dense = np.mean([len(dense_raw[q]) for q in qids])
        avg_union = np.mean(
            [len({d.pmid for d in bm25_raw[q]} | {d.pmid for d in dense_raw[q]}) for q in qids]
        )
        table2.add_row("Avg docs/query (BM25)", f"{avg_bm25:.0f}")
        table2.add_row("Avg docs/query (Dense)", f"{avg_dense:.0f}")
        table2.add_row("Avg docs/query (Union)", f"{avg_union:.0f}")

        console.print(table)
        console.print(table2)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            out_data = {
                "recall": all_results,
                "pool_stats": {
                    "bm25_unique": len(bm25_unique),
                    "dense_unique": len(dense_unique),
                    "union": len(union_unique),
                    "overlap": len(overlap),
                },
            }
            output.write_text(json.dumps(out_data, indent=2))
            typer.echo(f"\nResults saved to {output}")

    finally:
        await close_pool()
        await close_qdrant_client()


@app.command("pool-compare")
@typer_async
async def pool_compare(
    batches: Annotated[
        list[int] | None,
        typer.Option("--batches", "-b", help="13B batch numbers (default: 1 2 3 4)."),
    ] = [1, 2, 3, 4],
    topk: Annotated[
        int,
        typer.Option("--topk", "-k", help="Docs to retrieve per method."),
    ] = 500,
    tei_url: Annotated[
        str | None,
        typer.Option("--tei-url", help="TEI embed endpoint URL."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Save results JSON."),
    ] = "pool_compare.json",
) -> None:
    """Compare BM25 alone vs union pool truncated to the same doc budget.

    Retrieves topk docs from each method, merges the two sets per query
    (up to 2*topk unique docs), then truncates to exactly topk using
    RRF-based ranking. This answers: "with the same number of documents
    as BM25, does adding Dense docs improve performance?"
    """
    used_batches = batches or [1, 2, 3, 4]
    used_metrics = DEFAULT_RETRIEVAL_METRICS + ["recall@200", "recall@500"]

    typer.echo(f"Loading golden data for batches: {used_batches}")
    questions, qrels, per_file = _load_golden_files(used_batches)
    if not questions:
        typer.echo("No questions found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(questions)} questions. Retrieving topk={topk} per method...")

    try:
        qids, bm25_raw, dense_raw = await _retrieve_raw(
            questions,
            bm25_topk=topk,
            dense_topk=topk,
            embed_url=tei_url,
        )

        # BM25-only run (already topk docs)
        bm25_run: dict[str, dict[str, float]] = {}
        # Dense-only run
        dense_run: dict[str, dict[str, float]] = {}
        # Union pool truncated to topk via RRF ranking
        pool_rrf_run: dict[str, dict[str, float]] = {}
        # Union pool WITHOUT truncation (all fused docs via RRF)
        pool_full_run: dict[str, dict[str, float]] = {}
        # Raw union: every doc from either set, scored by best retriever score
        raw_union_run: dict[str, dict[str, float]] = {}

        for qid in qids:
            bm25_docs = bm25_raw[qid]
            dense_docs = dense_raw[qid]

            bm25_run[qid] = _run_dict_from_docs(bm25_docs)
            dense_run[qid] = _run_dict_from_docs(dense_docs)

            # Raw union: merge both sets, keep max score per doc
            merged: dict[str, float] = {}
            for d in bm25_docs:
                merged[d.pmid] = float(d.score)
            for d in dense_docs:
                if d.pmid not in merged or float(d.score) > merged[d.pmid]:
                    merged[d.pmid] = float(d.score)
            raw_union_run[qid] = merged

            # Fuse both sets with RRF
            fused = fuse_retrieval_lists_rrf(
                qid,
                [("bm25", bm25_docs), ("dense", dense_docs)],
                rrf_k=60,
            )
            fused_dict = _run_dict_from_docs(fused)
            # Full pool (no truncation)
            pool_full_run[qid] = fused_dict
            # Truncate to same budget as BM25
            top_items = sorted(
                fused_dict.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:topk]
            pool_rrf_run[qid] = dict(top_items)

        runs = {
            "BM25": bm25_run,
            "Dense": dense_run,
            f"Pool RRF (top {topk})": pool_rrf_run,
            "Pool RRF (full)": pool_full_run,
            "Raw Union": raw_union_run,
        }

        typer.echo("\nEvaluating...")
        console = Console()
        all_results: dict[str, dict[str, dict[str, float]]] = {}
        for name, run_dict in runs.items():
            all_results[name] = evaluate_retrieval_run(
                run_dict,
                qrels,
                metrics=used_metrics,
                per_file_results=per_file if len(per_file) > 1 else None,
            )

        _print_results(all_results, per_file)

        # Show per-query stats
        overlaps = []
        bm25_only_relevant = 0
        dense_only_relevant = 0
        both_relevant = 0
        total_gold = 0
        missed = 0
        for qid in qids:
            bm25_set = set(bm25_run[qid].keys())
            dense_set = set(dense_run[qid].keys())
            overlaps.append(len(bm25_set & dense_set))
            rel = set(qrels.get(qid, {}).keys())
            total_gold += len(rel)
            bm25_rel = bm25_set & rel
            dense_rel = dense_set & rel
            bm25_only_relevant += len(bm25_rel - dense_rel)
            dense_only_relevant += len(dense_rel - bm25_rel)
            both_relevant += len(bm25_rel & dense_rel)
            missed += len(rel - bm25_rel - dense_rel)

        table2 = Table(title="Pool Statistics", show_lines=True)
        table2.add_column("Stat", style="bold")
        table2.add_column("Value", justify="right")
        table2.add_row("Docs per method", str(topk))
        table2.add_row(
            "Avg overlap/query",
            f"{np.mean(overlaps):.1f}",
        )
        table2.add_row(
            "Avg unique in pool/query",
            f"{np.mean([2 * topk - o for o in overlaps]):.0f}",
        )

        table3 = Table(
            title="Gold Document Breakdown",
            show_lines=True,
        )
        table3.add_column("Category", style="bold")
        table3.add_column("Count", justify="right")
        table3.add_column("%", justify="right")
        table3.add_row(
            "Total gold docs",
            str(total_gold),
            "100.0%",
        )
        table3.add_row(
            "Found by BM25 only",
            str(bm25_only_relevant),
            f"{100 * bm25_only_relevant / total_gold:.1f}%",
        )
        table3.add_row(
            "Found by Dense only",
            str(dense_only_relevant),
            f"{100 * dense_only_relevant / total_gold:.1f}%",
        )
        table3.add_row(
            "Found by both",
            str(both_relevant),
            f"{100 * both_relevant / total_gold:.1f}%",
        )
        found_total = bm25_only_relevant + dense_only_relevant + both_relevant
        table3.add_row(
            "Found by either (union)",
            str(found_total),
            f"{100 * found_total / total_gold:.1f}%",
        )
        table3.add_row(
            "Missed by both",
            str(missed),
            f"{100 * missed / total_gold:.1f}%",
        )

        console.print(table2)
        console.print(table3)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(all_results, indent=2))
            typer.echo(f"\nResults saved to {output}")

    finally:
        await close_pool()
        await close_qdrant_client()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
