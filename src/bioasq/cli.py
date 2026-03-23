"""Unified CLI entry point for the BioASQ pipeline.

All subcommands are registered under a single :class:`typer.Typer` app.

Usage::

    bioasq phase-a train --model-name bert-base-uncased …
    bioasq phase-a evaluate --model-name outputs/checkpoint …
    bioasq phase-a inference --model-name outputs/checkpoint …
    bioasq phase-b generate --data-path data/batch1.jsonl …
    bioasq phase-b synthesize runs/*.json …
    bioasq bm25 index --baseline data/baseline.jsonl …
    bioasq bm25 negatives training.jsonl …
"""

from __future__ import annotations

import typer

app: typer.Typer = typer.Typer(
    name="bioasq",
    help="BioASQ 14b — Unified biomedical question-answering pipeline.",
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

app.add_typer(phase_a_app, name="phase-a")
app.add_typer(phase_b_app, name="phase-b")
app.add_typer(bm25_app, name="bm25")


# ---------------------------------------------------------------------------
# BM25 commands
# ---------------------------------------------------------------------------


@bm25_app.command()
def index(
    baseline: str = typer.Option(..., help="Path to baseline JSONL file."),
    output_dir: str = typer.Option(..., "-o", "--out", help="Index output directory."),
) -> None:
    """Create a PISA BM25 index from a PubMed baseline."""
    from pathlib import Path

    from bioasq.phase_a.bm25.index import create_index

    create_index(Path(baseline), Path(output_dir))


@bm25_app.command()
def negatives(
    training_file: str = typer.Argument(..., help="Training JSONL file."),
    indexes_dir: str = typer.Option("../data/indexes", "-i", help="Indexes directory."),
    output_file: str = typer.Option("../data/negatives.jsonl", "-o", help="Output JSONL."),
    baselines_dir: str = typer.Option("../data/baselines", help="Baselines directory."),
    ids_per_baseline: str = typer.Option("../data/ids_per_baseline.json", "-p", help="IDs per baseline JSON."),
    k1: float = typer.Option(0.4, help="BM25 k1 parameter."),
    b: float = typer.Option(0.3, help="BM25 b parameter."),
    num_results: int = typer.Option(100, "-n", help="Negatives per question."),
) -> None:
    """Mine BM25 negatives for training questions."""
    from pathlib import Path

    from bioasq.phase_a.bm25.negatives import mine_negatives

    mine_negatives(
        Path(training_file),
        Path(indexes_dir),
        Path(output_file),
        Path(baselines_dir),
        Path(ids_per_baseline),
        k1=k1,
        b=b,
        num_results=num_results,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
