#!/bin/bash
# Quick test run — 100 random questions from training14b.json
# Use this to iterate on prompts and settings before a full run.
#
# Change INFERENCE_MODE and MODEL as needed.
# The random seed is fixed so you always test on the same 100 questions.

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-logging: launch detached background process, return terminal immediately
if [ -z "$LOGGING" ]; then
    mkdir -p "${REPO_DIR}/logs"
    RUN_NUM=1
    while [ -f "${REPO_DIR}/logs/test_run${RUN_NUM}.out" ]; do
        RUN_NUM=$((RUN_NUM + 1))
    done
    LOG="${REPO_DIR}/logs/test_run${RUN_NUM}.out"
    echo "Logging to: $LOG"
    LOGGING=1 nohup bash "$0" "$@" > "$LOG" 2>&1 &
    echo "Started in background (PID $!). Follow with: tail -f $LOG"
    exit
fi

# ----------------------------------------
# CONFIGURABLE
# ----------------------------------------

INFERENCE_MODE="local"          # local / openrouter
PROMPT_ID=1

    # --- Local Models ---
    # Qwen/Qwen2.5-0.5B-Instruct
    # meta-llama/Llama-3.2-1B
    # google/medgemma-27b-text-it
    # google/medgemma-4b-it
    # google/medgemma-1.5-4b-it


    # --- Cloud Models (OpenRouter) ---
    # google/gemini-2.5-flash
    # anthropic/claude-sonnet-4-6
    # openrouter/hunter-alpha
    # z-ai/glm-5-turbo

    # nvidia/nemotron-3-super-120b-a12b:free
    # minimax/minimax-m2.5:free
    # stepfun/step-3.5-flash:free
    # arcee-ai/trinity-large-preview:free
    # nvidia/nemotron-3-nano-30b-a3b:free


MODEL="google/medgemma-27b-text-it"

# ----------------------------------------
# TEST SETTINGS (fixed — do not change between runs for comparability)
# ----------------------------------------

LIMIT=100
RANDOM_SEED=42                  # fixed seed → same 100 questions every run

NUM_SNIPPETS=5
MAX_TOKENS=1000
TEMPERATURE=0.0

# GPU settings (local only)
# CUDA_VISIBLE_DEVICES: unset = use all GPUs; override at runtime, e.g.: CUDA_VISIBLE_DEVICES=0,1 ./test.sh
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=8192

QUESTION_TYPES="yesno factoid list summary" # yesno / factoid / list / summary
#  factoid list summary


# ----------------------------------------
# PATHS
# ----------------------------------------

INPUT="${REPO_DIR}/../data/training14b/training14b.json"

MODEL_NAME=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')
OUTPUT_FILE="${MODEL_NAME}_prompt_${PROMPT_ID}_test${LIMIT}.json"

OUTPUT_DIR="${REPO_DIR}/dev/outputs"
RESULTS_DIR="${REPO_DIR}/dev/results"

PREDICTIONS="${OUTPUT_DIR}/${OUTPUT_FILE}"
EVAL_REPORT="${RESULTS_DIR}/${OUTPUT_FILE}"

mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR" "${REPO_DIR}/logs"

# ----------------------------------------
# LOAD ENV (for OpenRouter API key)
# ----------------------------------------

if [ -f "${REPO_DIR}/.env" ]; then
    export $(cat "${REPO_DIR}/.env" | xargs)
fi


# ----------------------------------------
# INFERENCE
# ----------------------------------------

echo "========================================"
echo "Test run: ${LIMIT} random questions (seed=${RANDOM_SEED})"
echo "  Backend : ${INFERENCE_MODE}"
echo "  Model   : ${MODEL}"
echo "  Prompt  : ${PROMPT_ID}"
echo "========================================"

source .venv/bin/activate

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

echo "Inference done. Evaluating..."

# ----------------------------------------
# EVALUATE
# ----------------------------------------

uv run python "${REPO_DIR}/evaluation/evaluate.py" \
    --predictions  "$PREDICTIONS" \
    --ground-truth "$INPUT" \
    --output       "$EVAL_REPORT"

echo "========================================"
echo "Done. Report: ${EVAL_REPORT}"
