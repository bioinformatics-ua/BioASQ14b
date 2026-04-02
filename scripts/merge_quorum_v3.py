#!/usr/bin/env python3
"""
Merge split quorum v3 JSONL files into a single v3.final.json submission.

Files are merged in order; later files override earlier ones (by question id),
so resume files take precedence over the base v3.jsonl.
"""

import argparse
import json
from pathlib import Path

DEFAULT_DIR = Path(__file__).parent.parent / "data/batch02/generation/quorum"
DEFAULT_FILES = [
    "v3.jsonl",
    "v3_resume_70_75.jsonl",
    "v3_resume_76_80.jsonl",
]
DEFAULT_OUTPUT = "v3.final.json"


def load_jsonl(path: Path) -> list[dict]:
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def main():
    parser = argparse.ArgumentParser(
        description="Merge quorum v3 JSONL files into a final JSON submission."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help="Directory containing the JSONL files (default: %(default)s)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="JSONL files to merge, in order (later files override earlier ones by id). Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output JSON file (default: <dir>/{DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    output = args.output or (args.dir / DEFAULT_OUTPUT)

    # Merge: later files override earlier ones
    merged: dict[str, dict] = {}
    for filename in args.files:
        path = args.dir / filename
        if not path.exists():
            print(f"WARNING: {path} does not exist, skipping.")
            continue
        questions = load_jsonl(path)
        print(f"Loaded {len(questions)} questions from {filename}")
        for q in questions:
            merged[q["id"]] = q

    questions = list(merged.values())
    print(f"Total unique questions: {len(questions)}")

    result = {"questions": questions}
    with open(output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Written to {output}")


if __name__ == "__main__":
    main()
