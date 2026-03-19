#!/bin/bash

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-logging: launch detached background process, return terminal immediately
if [ -z "$LOGGING" ]; then
    mkdir -p "${REPO_DIR}/logs"
    RUN_NUM=1
    while [ -f "${REPO_DIR}/logs/run${RUN_NUM}.out" ]; do
        RUN_NUM=$((RUN_NUM + 1))
    done
    LOG="${REPO_DIR}/logs/run${RUN_NUM}.out"
    echo "Logging to: $LOG"
    LOGGING=1 nohup bash "$0" "$@" > "$LOG" 2>&1 &
    echo "Started in background (PID $!). Follow with: tail -f $LOG"
    exit
fi

# ----------------------------------------
# CONFIGURABLE VARIABLES
# ----------------------------------------

INFERENCE_MODE="openrouter"          # local / openrouter
PROMPT_ID=7
DATASET="dev"                   # dev  /  batch01  /  batch02  /  batch03  /  batch04

    # --- Local Models ---
    # google/medgemma-27b-text-it
    # google/gemma-3-27b-it
    # meta-llama/Llama-3.1-8B-Instruct
    # luisasousa/qwen35-pubmedqa

    # --- Cloud Models (OpenRouter) ---
    # google/gemini-2.5-flash
    # google/gemini-2.0-flash-001
    # anthropic/claude-sonnet-4-6
    # anthropic/claude-opus-4-6
    # anthropic/claude-haiku-4-5
    # openai/gpt-4.1
    # openai/gpt-4.1-mini
    # deepseek/deepseek-v3-2
    # deepseek/deepseek-chat-v3-0324
    # qwen/qwen3-235b-a22b-2507
    # qwen/qwen3-32b
    # meta-llama/llama-3.3-70b-instruct
    # mistralai/mistral-7b-instruct
    # Qwen/Qwen2.5-0.5B-Instruct
    # meta-llama/Llama-3.2-1B

MODEL="google/gemini-2.0-flash-001"

# ----------------------------------------
# INFERENCE SETTINGS
# ----------------------------------------

NUM_SNIPPETS=10                  # number of snippets per question fed to the model
MAX_TOKENS=1000                 # max tokens to generate per question
TEMPERATURE=0.0                 # 0.0 = greedy / deterministic

LIMIT=100                       # max number of questions (leave empty for full run)
RANDOM_SEED=42                  # fixed seed → same questions every run

# GPU engine settings (only used when INFERENCE_MODE=local)
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=8192

QUESTION_TYPES="yesno factoid list summary"
# "yesno factoid list summary" for all types

# ----------------------------------------
# PATHS
# ----------------------------------------

# Input data — training14b.json for dev, competition JSON for batch runs
if [ "$DATASET" = "dev" ]; then
    INPUT="${REPO_DIR}/../data/training14b/training14b.json"
else
    INPUT="${REPO_DIR}/${DATASET}/input/phase_b_data.json"
fi

# Output file naming: model name with slashes/dots replaced by dashes
# Appends _testN suffix when LIMIT is set so test runs never overwrite full runs
MODEL_NAME=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')
[ -n "$LIMIT" ] && SUFFIX="_test${LIMIT}" || SUFFIX=""
OUTPUT_FILE="${MODEL_NAME}_prompt_${PROMPT_ID}${SUFFIX}.json"

OUTPUT_DIR="${REPO_DIR}/${DATASET}/outputs"
RESULTS_DIR="${REPO_DIR}/${DATASET}/results"
SUBMISSION_DIR="${REPO_DIR}/${DATASET}/submission"

PREDICTIONS="${OUTPUT_DIR}/${OUTPUT_FILE}"
EVAL_REPORT="${RESULTS_DIR}/${OUTPUT_FILE}"
SUBMISSION="${SUBMISSION_DIR}/${OUTPUT_FILE}"

mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR" "$SUBMISSION_DIR" "${REPO_DIR}/logs"

# ----------------------------------------
# LOAD ENV
# ----------------------------------------

source .venv/bin/activate

if [ -f "${REPO_DIR}/.env" ]; then
    export $(cat "${REPO_DIR}/.env" | xargs)
fi

# ----------------------------------------
# STEP 1 — Inference
# ----------------------------------------

echo "========================================"
echo "Run: ${DATASET} | ${LIMIT} questions (seed=${RANDOM_SEED})"
echo "  Backend  : ${INFERENCE_MODE}"
echo "  Model    : ${MODEL}"
echo "  Prompt   : ${PROMPT_ID}"
echo "  Output   : ${PREDICTIONS}"
echo "========================================"

uv run python "${REPO_DIR}/inference/run.py" \
    --input                  "$INPUT" \
    --output                 "$PREDICTIONS" \
    --backend                "$INFERENCE_MODE" \
    --model                  "$MODEL" \
    --prompt-id              "$PROMPT_ID" \
    --num-snippets           "$NUM_SNIPPETS" \
    --max-tokens             "$MAX_TOKENS" \
    --temperature            "$TEMPERATURE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size   "$TENSOR_PARALLEL_SIZE" \
    --max-model-len          "$MAX_MODEL_LEN" \
    --limit                  "$LIMIT" \
    --random-seed            "$RANDOM_SEED" \
    --question-types         $QUESTION_TYPES

echo "[1/2] Inference done."

# ----------------------------------------
# STEP 2a — Evaluate (dev mode only)
# ----------------------------------------

if [ "$DATASET" = "dev" ]; then
    echo "[2/2] Evaluating predictions..."
    uv run python "${REPO_DIR}/evaluation/evaluate.py" \
        --predictions  "$PREDICTIONS" \
        --ground-truth "$INPUT" \
        --output       "$EVAL_REPORT"
    echo "[2/2] Evaluation done. Report: ${EVAL_REPORT}"
fi

# ----------------------------------------
# STEP 2b — Format conversion (batch mode only)
# ----------------------------------------

if [ "$DATASET" != "dev" ]; then
    RAW_BATCH="${REPO_DIR}/${DATASET}/input/batch.json"
    echo "[2/2] Converting to BioASQ submission format..."
    uv run python "${REPO_DIR}/../phaseB/bioasq_format_converter.py" \
        "$RAW_BATCH" \
        "$PREDICTIONS" \
        "$SUBMISSION" \
        "$PREDICTIONS"
    echo "[2/2] Conversion done. Submission: ${SUBMISSION}"
fi

echo "========================================"
echo "Done."
