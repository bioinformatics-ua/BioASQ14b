#!/bin/bash
# run_synthesis.sh
#
# Example grid-search synthesis run.
#
# Workflow:
#   1. Run run.py multiple times with different models/prompts → run files
#   2. Run this script → synthesize.py grid-searches over synthesis prompt IDs
#   3. Each output feeds into evaluate.py to find the best synthesis config

# Ideal answers — collects all draft ideal answers per question, feeds them to an LLM with a synthesis prompt, which writes one better combined answer. Grid-searches over prompt IDs in one model load.
# Exact answers — merged in code, no LLM: yesno uses majority vote, factoid uses frequency ranking (top-5), list keeps only entities appearing in ≥ half the runs. Tiebreaks fall back to --best-run.

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-logging: launch detached background process, return terminal immediately
if [ -z "$LOGGING" ]; then
    mkdir -p "${REPO_DIR}/logs"
    RUN_NUM=1
    while [ -f "${REPO_DIR}/logs/synthesis_run${RUN_NUM}.out" ]; do
        RUN_NUM=$((RUN_NUM + 1))
    done
    LOG="${REPO_DIR}/logs/synthesis_run${RUN_NUM}.out"
    echo "Logging to: $LOG"
    LOGGING=1 nohup bash "$0" "$@" > "$LOG" 2>&1 &
    echo "Started in background (PID $!). Follow with: tail -f $LOG"
    exit
fi

DATA_PATH="${REPO_DIR}/../data/training14b/training14b.json"

# ----------------------------------------
# Input runs to synthesize from
# (output files from run.py)
# ----------------------------------------
RUN_DIR="${REPO_DIR}/dev/outputs"

RUNS=(
    "${RUN_DIR}/google-gemini-2-5-flash_prompt_1_test100.json"
    "${RUN_DIR}/google-gemini-2-0-flash-001_prompt_1_test100.json"
    "${RUN_DIR}/stepfun-step-3-5-flash:free_prompt_1_test100.json"
)

# ----------------------------------------
# Synthesis config
# ----------------------------------------
BACKEND="openrouter"
MODEL="google/gemini-2.5-flash"

PROMPT_IDS="1,2,3,4"   # grid search over all synthesis prompts

OUT_ID="exp2"
OUTPUT_DIR="${REPO_DIR}/dev/outputs/synthesis"

MAX_TOKENS=500
TEMPERATURE=0.0

# ----------------------------------------
# Load env (API keys)
# ----------------------------------------
if [ -f "${REPO_DIR}/.env" ]; then
    export $(cat "${REPO_DIR}/.env" | xargs)
fi

# ----------------------------------------
# Run synthesis
# ----------------------------------------
echo "Synthesizing from ${#RUNS[@]} run files..."
echo "Grid searching over prompt IDs: ${PROMPT_IDS}"
echo ""

uv run python "${REPO_DIR}/synthesis/synthesize.py" \
    "${RUNS[@]}" \
    --data-path   "$DATA_PATH" \
    --output-dir  "$OUTPUT_DIR" \
    --out-id      "$OUT_ID" \
    --backend     "$BACKEND" \
    --model       "$MODEL" \
    --prompt-ids  "$PROMPT_IDS" \
    --max-tokens  "$MAX_TOKENS" \
    --temperature "$TEMPERATURE"

# ----------------------------------------
# Evaluate all synthesis outputs
# ----------------------------------------
echo ""
echo "Evaluating synthesis outputs..."

RESULTS_DIR="${REPO_DIR}/dev/results/synthesis"
mkdir -p "$RESULTS_DIR"

for OUT_FILE in "${OUTPUT_DIR}/${OUT_ID}_"*.json; do
    BASENAME=$(basename "$OUT_FILE" .json)
    echo "  Evaluating: ${BASENAME}"
    uv run python "${REPO_DIR}/evaluation/evaluate.py" \
        --predictions  "$OUT_FILE" \
        --ground-truth "$DATA_PATH" \
        --output       "${RESULTS_DIR}/${BASENAME}.json"
done

echo ""
echo "Done. Results in: ${RESULTS_DIR}"
