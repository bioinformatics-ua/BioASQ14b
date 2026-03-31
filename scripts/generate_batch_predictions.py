#!/usr/bin/env python3
"""
End-to-end pipeline: Hybrid retrieval → Reranker inference for all local models.

Steps:
  1. Convert BioASQ testset JSON to retrieval input JSONL  ({id, body})
  2. Run hybrid retrieval (BM25 + Dense → RRF fusion)
  3. Reformat retrieval output to reranker candidate format ({id, query_text, bm25: [{id, text}]})
  4. Run reranker inference on each checkpoint

Usage:
  uv run python scripts/generate_batch_predictions.py \
    --testset data/BioASQ-task14bPhaseA-testset1.json \
    --output-dir phaseA-reranker/refactored-trainer/inference_batch1 \
    --models-dir phaseA-reranker/refactored-trainer/outputs

  # Skip retrieval if candidates already exist:
  uv run python scripts/generate_batch_predictions.py \
    --candidates data/bm25/batch1_hybrid.jsonl \
    --output-dir phaseA-reranker/refactored-trainer/inference_batch1 \
    --models-dir phaseA-reranker/refactored-trainer/outputs
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import orjson
import typer
from tqdm import tqdm

app = typer.Typer(no_args_is_help=True)


# ── Step 1+2: Retrieval ─────────────────────────────────────────────────────


def _convert_testset_to_jsonl(testset_path: Path) -> list[dict]:
    """Convert BioASQ testset JSON to list of {id, body} dicts."""
    with testset_path.open("rb") as f:
        data = orjson.loads(f.read())
    return [{"id": q["id"], "body": q["body"]} for q in data["questions"]]


async def _run_hybrid_retrieval(
    questions: list[dict],
    *,
    bm25_topk: int = 100,
    semantic_topk: int = 100,
    rrf_k: int = 60,
    tei_embed_url: str | None = None,
) -> list[dict]:
    """Run hybrid retrieval and return reranker-ready candidates."""

    from bioasq.phase_a.retrieval.pipeline import hybrid_retrieve_rrf

    candidates = []
    for question in tqdm(questions, desc="Hybrid retrieve", unit="q"):
        qid = str(question["id"])
        body = str(question["body"])
        fused = await hybrid_retrieve_rrf(
            qid,
            body,
            bm25_topk=bm25_topk,
            semantic_topk=semantic_topk,
            embed_url=tei_embed_url,
            rrf_k=rrf_k,
        )
        # Convert to reranker input format: {id, query_text, bm25: [{id, text}]}
        candidates.append({
            "id": qid,
            "query_text": body,
            "bm25": [
                {"id": d.pmid, "text": d.full_text}
                for d in fused
            ],
        })
    return candidates


def _write_candidates(candidates: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for row in candidates:
            f.write(orjson.dumps(row) + b"\n")


def _load_candidates(path: Path) -> list[dict]:
    with path.open("rb") as f:
        return [orjson.loads(line) for line in f if line.strip()]


# ── Step 3: Discover checkpoints ────────────────────────────────────────────


def _find_checkpoints(models_dir: Path) -> list[Path]:
    """Find all checkpoint-* directories under models_dir."""
    checkpoints = sorted(models_dir.rglob("checkpoint-*"))
    return [cp for cp in checkpoints if cp.is_dir()]


# ── Step 4: Reranker inference ──────────────────────────────────────────────


def _run_inference_on_checkpoint(
    checkpoint: Path,
    candidates_path: Path,
    output_path: Path,
    *,
    batch_size: int = 64,
    max_length: int = 512,
    max_docs: int = 100,
    inference_dtype: str = "bfloat16",
) -> None:
    """Run reranker inference for a single checkpoint."""
    from bioasq.phase_a.reranker.cli import inference_command

    inference_command(
        model_name=str(checkpoint),
        questions_path=candidates_path,
        output_path=output_path,
        batch_size=batch_size,
        max_length=max_length,
        max_docs=max_docs,
        inference_dtype=inference_dtype,
    )


def _checkpoint_output_name(checkpoint: Path, models_dir: Path) -> str:
    """Build output directory name from checkpoint relative path."""
    rel = checkpoint.relative_to(models_dir)
    return str(rel).replace("/", "__")


# ── Main command ────────────────────────────────────────────────────────────


@app.command()
def main(
    testset: Annotated[
        Path | None,
        typer.Option(help="BioASQ testset JSON ({questions: [{id, body, type}]})."),
    ] = None,
    candidates: Annotated[
        Path | None,
        typer.Option(help="Pre-built candidates JSONL (skip retrieval)."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(help="Base directory for inference outputs."),
    ] = Path("phaseA-reranker/refactored-trainer/inference_batch1"),
    models_dir: Annotated[
        Path,
        typer.Option(help="Directory containing trained model checkpoints."),
    ] = Path("phaseA-reranker/refactored-trainer/outputs"),
    bm25_topk: Annotated[int, typer.Option(help="BM25 candidate depth.")] = 100,
    semantic_topk: Annotated[int, typer.Option(help="Dense candidate depth.")] = 100,
    rrf_k: Annotated[int, typer.Option(help="RRF smoothing constant k.")] = 60,
    batch_size: Annotated[int, typer.Option(help="Reranker batch size.")] = 64,
    max_length: Annotated[int, typer.Option(help="Max token length.")] = 512,
    max_docs: Annotated[int, typer.Option(help="Max candidates per question.")] = 100,
    inference_dtype: Annotated[str, typer.Option(help="Inference dtype.")] = "bfloat16",
) -> None:
    """Generate reranker predictions for all local model checkpoints."""
    if testset is None and candidates is None:
        typer.echo("Error: provide either --testset (run retrieval) or --candidates (skip retrieval).")
        raise typer.Exit(1)

    # ── Retrieval or load candidates ─────────────────────────────────────
    if candidates and candidates.exists():
        typer.echo(f"Loading pre-built candidates from {candidates}")
        candidates_path = candidates
    elif testset:
        typer.echo(f"Running hybrid retrieval on {testset}...")
        questions = _convert_testset_to_jsonl(testset)
        typer.echo(f"  {len(questions)} questions loaded.")

        candidates_list = asyncio.run(
            _run_hybrid_retrieval(
                questions,
                bm25_topk=bm25_topk,
                semantic_topk=semantic_topk,
                rrf_k=rrf_k,
            )
        )

        # Save candidates alongside the output dir
        candidates_path = output_dir / "candidates_hybrid.jsonl"
        _write_candidates(candidates_list, candidates_path)
        typer.echo(f"  Candidates saved to {candidates_path}")
    else:
        typer.echo(f"Error: candidates file not found: {candidates}")
        raise typer.Exit(1)

    # ── Discover checkpoints ─────────────────────────────────────────────
    checkpoints = _find_checkpoints(models_dir)
    if not checkpoints:
        typer.echo(f"No checkpoints found under {models_dir}")
        raise typer.Exit(1)

    typer.echo(f"\nFound {len(checkpoints)} checkpoints under {models_dir}")
    typer.echo(f"Output base: {output_dir}")
    typer.echo("=" * 60)

    # ── Run inference per checkpoint ─────────────────────────────────────
    for i, checkpoint in enumerate(checkpoints, 1):
        name = _checkpoint_output_name(checkpoint, models_dir)
        pred_path = output_dir / name / "predictions.json"

        if pred_path.exists():
            typer.echo(f"[{i}/{len(checkpoints)}] SKIP (exists): {name}")
            continue

        typer.echo(f"\n[{i}/{len(checkpoints)}] {name}")
        _run_inference_on_checkpoint(
            checkpoint,
            candidates_path,
            pred_path,
            batch_size=batch_size,
            max_length=max_length,
            max_docs=max_docs,
            inference_dtype=inference_dtype,
        )
        typer.echo(f"[{i}/{len(checkpoints)}] Done: {pred_path}")

    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"All {len(checkpoints)} checkpoints processed.")


if __name__ == "__main__":
    app()
