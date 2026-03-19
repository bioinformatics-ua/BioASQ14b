"""
evaluation/evaluate.py

CLI wrapper that scores a predictions file against ground truth.

Loads predictions from run.py output, loads ground truth via the dataloader,
calls metrics.py, and prints a formatted report. Optionally saves the report
as JSON for tracking results across prompt experiments.

Usage:
    python evaluation/evaluate.py \
        --predictions dev/outputs/run_p1.json \
        --ground-truth data/training14b/training14b.json \
        --output dev/results/eval_p1.json
"""

import argparse
import json
import sys
from pathlib import Path

# Allow imports from project root regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loaders.dataloader import BioASQDataLoader
from evaluation.metrics import evaluate_all


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(results: dict, n_total: int, n_valid: int) -> None:
    """
    Print a human-readable evaluation report to stdout.

    Shows per-type metrics with the official competition measure highlighted,
    plus ROUGE-2 for ideal answers across all types.
    """
    sep = "-" * 52

    print()
    print("=" * 52)
    print("  BioASQ Phase B — Evaluation Report")
    print("=" * 52)

    # Coverage — how many predictions had valid JSON output
    pct_valid = 100 * n_valid / n_total if n_total else 0
    print(f"\n  Questions scored : {n_total}")
    print(f"  Valid JSON output: {n_valid} ({pct_valid:.1f}%)")

    # -------------------------------------------------------------------
    # ROUGE-2 ideal answers (all types)
    # -------------------------------------------------------------------
    r2 = results["rouge2_ideal"]
    print(f"\n{sep}")
    print("  ROUGE-2 (ideal answers — all types)")
    print(sep)
    print(f"  Mean ROUGE-2 F1 : {r2['mean']:.4f}")

    # Per-type ROUGE-2 breakdown if per_question scores exist
    if r2.get("per_question"):
        by_type: dict[str, list[float]] = {}
        # We don't have type info here — it comes from ground truth
        # This is filled in by evaluate_all_with_types below if available
        pass

    # -------------------------------------------------------------------
    # Yes/No — official measure: macro F1
    # -------------------------------------------------------------------
    yn = results["yesno"]
    print(f"\n{sep}")
    print("  Yes/No questions")
    print(sep)
    print(f"  Macro F1 (official) : {yn['macro_f1']:.4f}")
    print(f"  F1 yes              : {yn['f1_yes']:.4f}")
    print(f"  F1 no               : {yn['f1_no']:.4f}")
    print(f"  Questions scored    : {yn['n_scored']}")

    # -------------------------------------------------------------------
    # Factoid — official measure: MRR
    # -------------------------------------------------------------------
    fa = results["factoid"]
    print(f"\n{sep}")
    print("  Factoid questions")
    print(sep)
    print(f"  MRR (official)   : {fa['mrr']:.4f}")
    print(f"  Strict acc (SAcc): {fa['strict_acc']:.4f}")
    print(f"  Lenient acc (LAcc): {fa['lenient_acc']:.4f}")
    print(f"  Questions scored : {fa['n_scored']}")

    # -------------------------------------------------------------------
    # List — official measure: mean F1
    # -------------------------------------------------------------------
    li = results["list"]
    print(f"\n{sep}")
    print("  List questions")
    print(sep)
    print(f"  Mean F1 (official): {li['mean_f1']:.4f}")
    print(f"  Questions scored  : {li['n_scored']}")

    print(f"\n{'=' * 52}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score BioASQ Phase B predictions against ground truth"
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions JSON (output of inference/run.py)"
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to BioASQ JSON with gold answers (e.g. training14b.json)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the report as JSON"
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=["yesno", "factoid", "list", "summary"],
        help="Question types to evaluate (default: all four)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of ground truth questions (for quick checks)"
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load predictions
    # -----------------------------------------------------------------------
    print(f"Loading predictions from: {args.predictions}")
    with open(args.predictions) as f:
        predictions: dict = json.load(f)

    # Count how many predictions had valid JSON output from the model
    n_valid = sum(1 for v in predictions.values() if v.get("valid", False))
    n_total = len(predictions)
    print(f"Predictions loaded: {n_total} total, {n_valid} with valid JSON")

    # -----------------------------------------------------------------------
    # Load ground truth via dataloader
    #
    # We filter to the same question types requested and only score questions
    # that actually have gold answers (ideal_answer is not None).
    # -----------------------------------------------------------------------
    print(f"Loading ground truth from: {args.ground_truth}")
    loader = BioASQDataLoader(
        path=args.ground_truth,
        limit=args.limit,
        question_types=args.question_types,
    )

    # Keep only questions that have gold answers and appear in predictions
    ground_truth = [
        q for q in loader
        if q.get("ideal_answer") is not None
        and q["id"] in predictions
    ]
    print(f"Ground truth questions available for scoring: {len(ground_truth)}")

    # -----------------------------------------------------------------------
    # Run all metrics
    # -----------------------------------------------------------------------
    print("Computing metrics...")
    results = evaluate_all(predictions, ground_truth)

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    print_report(results, n_total, n_valid)

    # -----------------------------------------------------------------------
    # Optionally save report as JSON
    # -----------------------------------------------------------------------
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "predictions_file": args.predictions,
            "ground_truth_file": args.ground_truth,
            "n_total": n_total,
            "n_valid": n_valid,
            "metrics": results,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
