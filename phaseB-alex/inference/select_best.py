"""
inference/select_best.py

Select the best answer per question based on judge scores.

No LLM calls — pure score-based selection (instant).

Usage:
    python inference/select_best.py \
        --scores  outputs/judged/scores.json \
        --inputs  outputs/exact/model1_factoid.json outputs/exact/model2_factoid.json \
        --output  outputs/selected/best.json \
        --metric  overall
"""

import argparse, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True,
                   help="Judge scores JSON from judge_answers.py")
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Original prediction JSON files (same ones that were judged)")
    p.add_argument("--output", required=True,
                   help="Output prediction JSON with best answer per question")
    p.add_argument("--metric", default="overall",
                   choices=["correctness", "faithfulness", "completeness", "overall"],
                   help="Score dimension to use for selection")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="Skip questions where best score is below this threshold")
    args = p.parse_args()

    # Load judge scores
    with open(args.scores) as f:
        judge_data = json.load(f)
    all_scores = judge_data["scores"]

    # Load prediction files (source_name → predictions)
    sources = {}
    for path in args.inputs:
        name = Path(path).stem
        with open(path) as f:
            sources[name] = json.load(f)

    # Select best per question
    results = {}
    selection_stats = {}

    for qid, qscores in all_scores.items():
        best_source = None
        best_score = -1.0

        for source_name, score_dict in qscores.items():
            s = score_dict.get(args.metric, 0.0)
            if s > best_score:
                best_score = s
                best_source = source_name

        if best_source is None or best_score < args.min_score:
            continue

        if best_source in sources and qid in sources[best_source]:
            results[qid] = dict(sources[best_source][qid])
            results[qid]["_selected_from"] = best_source
            results[qid]["_judge_score"] = best_score
            judge_row = qscores.get(best_source) or {}
            ideal = (judge_row.get("ideal_answer") or "").strip()
            if ideal:
                results[qid]["ideal_answer"] = ideal
            selection_stats[best_source] = selection_stats.get(best_source, 0) + 1

    print(f"Selected best answer for {len(results)} questions (metric: {args.metric})")
    if results:
        print("Selection distribution:")
        for source, count in sorted(selection_stats.items(), key=lambda x: -x[1]):
            print(f"  {source}: {count} questions ({100 * count / len(results):.1f}%)")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
