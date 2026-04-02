"""Unified CLI entry point for the BioASQ pipeline.

All subcommands are registered under a single :class:`typer.Typer` app.

Usage::

    bioasq phase-a train --model-name bert-base-uncased …
    bioasq phase-a evaluate --model-name outputs/checkpoint …
    bioasq phase-a inference --model-name outputs/checkpoint …
    bioasq phase-a hybrid-retrieve run-jsonl questions.jsonl -o hybrid.jsonl
    bioasq phase-b generate --data-path data/batch1.jsonl …
    bioasq phase-b synthesize runs/*.json …
    bioasq bm25 index --baseline data/baseline.jsonl …
    bioasq bm25 negatives training.jsonl …
"""

from pathlib import Path

import typer

# Standalone evaluation script (BM25 / Dense / Hybrid comparison)
# from scripts.evaluate_retrieval import app as eval_retrieval_app
from bioasq.data.qdrant_store import upload_embeddings_command
from bioasq.phase_a.bm25.negatives import app as negatives_app
from bioasq.phase_a.reranker.cli import evaluate_command, inference_command, train_command
from bioasq.phase_a.reranker.experiments import (
    run_experiments_command,
    run_llama_experiments_command,
)
from bioasq.phase_b.quorum.run import app as quorum_app

app: typer.Typer = typer.Typer(
    name="bioasq",
    help="BioASQ 14b — Unified biomedical question-answering pipeline.",
    no_args_is_help=True,
)

data_app: typer.Typer = typer.Typer(
    name="data",
    help="Data management: embeddings, vector stores, etc.",
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# Sub-apps
# ---------------------------------------------------------------------------

phase_a_app: typer.Typer = typer.Typer(
    name="phase-a",
    help="Phase A: document retrieval and reranking.",
    no_args_is_help=True,
)

phase_b_app: typer.Typer = typer.Typer(
    name="phase-b",
    help="Phase B: answer generation and synthesis.",
    no_args_is_help=True,
)

bm25_app: typer.Typer = typer.Typer(
    name="bm25",
    help="BM25 index management and negative mining.",
    no_args_is_help=True,
)

bm25_app.add_typer(negatives_app, name="negatives")

app.add_typer(phase_a_app, name="phase-a")
app.add_typer(phase_b_app, name="phase-b")
app.add_typer(bm25_app, name="bm25")
app.add_typer(data_app, name="data")
# app.add_typer(eval_retrieval_app, name="evaluate-retrieval")

# ---------------------------------------------------------------------------
# Phase-A Reranker commands
# ---------------------------------------------------------------------------

phase_a_app.command(name="train")(train_command)
phase_a_app.command(name="evaluate")(evaluate_command)
phase_a_app.command(name="inference")(inference_command)
phase_a_app.command(name="run-experiments")(run_experiments_command)
phase_a_app.command(name="run-llama-experiments")(run_llama_experiments_command)

# ---------------------------------------------------------------------------
# Phase-B commands
# ---------------------------------------------------------------------------
# phase_b_app.command(name="generate")(generate_command)
phase_b_app.add_typer(quorum_app, name="quorum")

# ---------------------------------------------------------------------------
# Data commands
# ---------------------------------------------------------------------------

data_app.command(name="upload-to-qdrant")(upload_embeddings_command)

# ---------------------------------------------------------------------------
# BM25 commands
# ---------------------------------------------------------------------------


@bm25_app.command()
def index(
    baseline: str = typer.Option(..., help="Path to baseline JSONL file."),
    output_dir: str = typer.Option(..., "-o", "--out", help="Index output directory."),
) -> None:
    """Create a PISA BM25 index from a PubMed baseline."""

    from bioasq.phase_a.bm25.index import create_index

    create_index(Path(baseline), Path(output_dir))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
