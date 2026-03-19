#!/usr/bin/env bash
# Run pseudo_relevance_feedback.py over all batch01 reranker runs.
# Derives the model checkpoint path from each run filename automatically.
# Run from: phaseA-reranker/
set -euo pipefail

RUNS_DIR="refactored-trainer/batch01/runs"
OUTPUTS_DIR="refactored-trainer/outputs"

ranx_runs=()
models=()

for run_file in "$RUNS_DIR"/*.json; do
    fname="$(basename "$run_file" .json)"

    # Split "run_name-checkpoint-N" → run_name and checkpoint-N
    ckpt_name="$(echo "$fname" | grep -oE 'checkpoint-[0-9]+$')"
    run_name="${fname%-${ckpt_name}}"

    # Find the checkpoint dir anywhere under outputs/
    ckpt_path="$(find "$OUTPUTS_DIR" -type d -name "$ckpt_name" | grep "/${run_name}/" | head -1)"

    if [[ -z "$ckpt_path" ]]; then
        echo "WARNING: checkpoint not found for '$fname' — skipping" >&2
        continue
    fi

    ranx_runs+=("$run_file")
    models+=("$ckpt_path")
done

if [[ ${#ranx_runs[@]} -eq 0 ]]; then
    echo "No runs found in $RUNS_DIR" >&2
    exit 1
fi

RANX_RUNS="$(IFS=,; echo "${ranx_runs[*]}")"
MODELS="$(IFS=,; echo "${models[*]}")"

echo "Found ${#ranx_runs[@]} runs"

uv run pseudo_relevance_feedback.py refactored-trainer/batch01/b01.json \
    --ranx-runs "$RANX_RUNS" \
    --models    "$MODELS" \
    --baseline  ../data/baselines/pubmed_baseline_2026.jsonl \
    --lookup    ../data/lookup.json \
    --output    refactored-trainer/batch01/dprf
