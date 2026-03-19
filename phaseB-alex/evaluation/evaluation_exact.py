"""
evaluation/evaluation_exact.py

CLI: score a run_exact.py predictions file against a gold JSONL file.

Usage:
    python evaluation/evaluation_exact.py \
        --predictions dev/outputs/exact/model_p1_abstracts_yesno_factoid_list.json \
        --ground-truth ../data/val_data/13B1_golden_documents.jsonl \
        [--output dev/results/exact/report.json]
        [--types yesno factoid list]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loaders.dataloader import BioASQDataLoader
from evaluation.metrics_exact import score_yesno, score_factoid, score_list


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: dict) -> None:
    W = 52
    sep = "-" * W

    print()
    print("=" * W)
    print("  BioASQ Phase B — Exact Answer Evaluation")
    print("=" * W)

    # ---- Yes/No -------------------------------------------------------
    yn = results.get("yesno")
    if yn:
        print(f"\n{sep}")
        print("  Yes/No questions")
        print(sep)
        print(f"  Macro F1   [official] : {yn['macro_f1']:.4f}")
        print(f"  F1 yes                : {yn['f1_yes']:.4f}")
        print(f"  F1 no                 : {yn['f1_no']:.4f}")
        print(f"  Accuracy              : {yn['accuracy']:.4f}")
        print(f"  Questions scored      : {yn['n_scored']}")

    # ---- Factoid -------------------------------------------------------
    fa = results.get("factoid")
    if fa:
        print(f"\n{sep}")
        print("  Factoid questions")
        print(sep)
        print(f"  MRR        [official] : {fa['mrr']:.4f}")
        print(f"  Strict acc  (SAcc)    : {fa['strict_acc']:.4f}")
        print(f"  Lenient acc (LAcc)    : {fa['lenient_acc']:.4f}")
        print(f"  Questions scored      : {fa['n_scored']}")

    # ---- List ----------------------------------------------------------
    li = results.get("list")
    if li:
        print(f"\n{sep}")
        print("  List questions")
        print(sep)
        print(f"  Mean F1    [official] : {li['mean_f1']:.4f}")
        print(f"  Mean precision        : {li['mean_precision']:.4f}")
        print(f"  Mean recall           : {li['mean_recall']:.4f}")
        print(f"  Questions scored      : {li['n_scored']}")

    print(f"\n{'=' * W}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions",   required=True,
                   help="JSON output from run_exact.py")
    p.add_argument("--ground-truth",  required=True,
                   help="Gold JSONL file (e.g. 13B1_golden_documents.jsonl)")
    p.add_argument("--output",        default=None,
                   help="Optional path to save report as JSON")
    p.add_argument("--types",         nargs="+",
                   default=["yesno", "factoid", "list"],
                   choices=["yesno", "factoid", "list"],
                   help="Types to evaluate (default: all three)")
    args = p.parse_args()

    # Load predictions
    with open(args.predictions) as f:
        predictions: dict = json.load(f)

    n_valid = sum(1 for v in predictions.values() if v.get("valid", False))
    print(f"Predictions : {len(predictions)} total, {n_valid} valid JSON")

    # Load ground truth
    loader       = BioASQDataLoader(path=args.ground_truth)
    ground_truth = [q for q in loader if q["id"] in predictions]
    print(f"Ground truth: {len(ground_truth)} matched questions")

    # Score
    run_types = set(args.types)
    results   = {}
    if "yesno"   in run_types:
        results["yesno"]   = score_yesno(predictions, ground_truth)
    if "factoid" in run_types:
        results["factoid"] = score_factoid(predictions, ground_truth)
    if "list"    in run_types:
        results["list"]    = score_list(predictions, ground_truth)

    print_report(results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "predictions_file":  args.predictions,
                "ground_truth_file": args.ground_truth,
                "n_predictions":     len(predictions),
                "n_valid":           n_valid,
                "metrics":           results,
            }, f, indent=2)
        print(f"Report saved to: {out}")


if __name__ == "__main__":
    main()