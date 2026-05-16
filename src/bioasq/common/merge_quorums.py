"""
Merge multiple vx.bioasq.json quorum submissions by randomly selecting
one version per question.

Usage:
    uv run merge_quorums.py <output> <v1.bioasq.json> [<v2.bioasq.json> ...]

Options:
    --seed INT   Random seed for reproducibility (default: random)
"""

import argparse
import glob
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Merge quorum submissions by randomly selecting one version per question."
    )
    parser.add_argument("output", help="Output file path")
    parser.add_argument("inputs", nargs="+", help="Input vx.bioasq.json files (supports globs)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Expand globs
    input_paths = []
    for pattern in args.inputs:
        expanded = glob.glob(pattern)
        if expanded:
            input_paths.extend(expanded)
        else:
            input_paths.append(pattern)

    if len(input_paths) < 2:
        parser.error("At least 2 input files are required.")

    print(f"Loading {len(input_paths)} files:")
    all_questions: dict[str, list] = {}  # id -> list of question dicts from each file

    for path in input_paths:
        print(f"  {path}")
        with open(path) as f:
            data = json.load(f)
        for q in data["questions"]:
            qid = q["id"]
            if qid not in all_questions:
                all_questions[qid] = []
            all_questions[qid].append(q)

    merged = []
    pick_counts: dict[str, int] = {}
    for qid, versions in all_questions.items():
        chosen = random.choice(versions)
        merged.append(chosen)
        # track which file index was chosen for reporting
        idx = versions.index(chosen)
        pick_counts[Path(input_paths[idx]).name] = (
            pick_counts.get(Path(input_paths[idx]).name, 0) + 1
        )

    print("\nPicked per file:")
    for fname, count in sorted(pick_counts.items()):
        print(f"  {fname}: {count}")

    output = {"questions": merged}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(merged)} questions to {out_path}")


if __name__ == "__main__":
    main()
