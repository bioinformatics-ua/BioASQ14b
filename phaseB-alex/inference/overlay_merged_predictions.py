#!/usr/bin/env python3
"""
Overlay partial merge/exact outputs onto an existing merged predictions file.

Use when you already have a full `submission_*_topk*.json` (id -> prediction dict)
for yesno/factoid/list and only want to add or replace **summary** (or any subset)
without rerunning the whole pipeline.

Example:
  # 1. Generate + judge + merge summary only (separate small run):
  uv run python inference/run_exact.py ... --types summary ...
  uv run python inference/judge_answers.py --types summary ... --inputs ...summary files...
  uv run python inference/merge_answers.py --types summary ... -o outputs/merged/summary_only.json

  # 2. Patch the previous full merge:
  uv run python inference/overlay_merged_predictions.py \\
    --base   phaseB-testset1/outputs/merged/submission_1_topk1.json \\
    --overlay phaseB-testset1/outputs/merged/summary_only_topk1.json \\
    --output  phaseB-testset1/outputs/merged/submission_1_topk1_with_summary.json

For each question id present in --overlay, the prediction dict **replaces** the
entry in --base (other ids are unchanged).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="Existing merged JSON (all types)")
    p.add_argument(
        "--overlay",
        required=True,
        help="Predictions to merge in (e.g. summary-only merge output)",
    )
    p.add_argument("--output", required=True, help="Write combined JSON here")
    args = p.parse_args()

    base_path = Path(args.base)
    over_path = Path(args.overlay)
    out_path = Path(args.output)

    with open(base_path) as f:
        base = json.load(f)
    with open(over_path) as f:
        overlay = json.load(f)

    if not isinstance(base, dict) or not isinstance(overlay, dict):
        raise SystemExit("Both --base and --overlay must be JSON objects: {qid: prediction}")

    merged = dict(base)
    for qid, pred in overlay.items():
        if not isinstance(pred, dict):
            raise SystemExit(f"Overlay value for {qid!r} must be an object, got {type(pred)}")
        merged[qid] = pred

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")

    n_new = len(overlay)
    n_base = len(base)
    print(
        f"Wrote {len(merged)} questions to {out_path} "
        f"(base had {n_base}, replaced/added {n_new} from overlay)"
    )


if __name__ == "__main__":
    main()
