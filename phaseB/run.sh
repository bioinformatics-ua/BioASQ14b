#!/bin/bash

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-logging: launch as detached background process, return terminal immediately
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

    # --- Cloud Models (OpenRouter) ---
    # google/gemini-2.5-flash
    # google/gemini-2.0-flash-001
    # anthropic/claude-sonnet-4-6
    # anthropic/claude-opus-4-6
    # openai/gpt-4.1
    # qwen/qwen3-235b-a22b-2507
    # qwen/qwen3-32b

    # --- Local Models (vLLM) ---
    # google/medgemma-27b-text-it
    # google/gemma-3-27b-it
    # meta-llama/Llama-3.1-8B-Instruct

MODEL="google/gemini-2.0-flash-001"

BACKEND="openrouter"           # local / openrouter

PROMPT_IDS="2"                 # comma-separated, e.g. "2,6,7"
NUM_SUPPORT="5"                # comma-separated abstract counts, e.g. "3,5,10"
INPUT_TYPE="abstracts"         # abstracts / snippets

MAX_TOKENS=500
TEMPERATURE=0.0

# GPU settings (only used when BACKEND=local)
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=8192

# ----------------------------------------
# PATHS
# ----------------------------------------

DATASET="dev"                  # dev / batch01

if [ "$DATASET" = "dev" ]; then
    INPUT="/home/ucloud/BioASQ13B/data/val_data/13B1_golden_documents.jsonl"
elif [ "$DATASET" = "batch01" ]; then
    INPUT="/home/ucloud/BioASQ13B/phaseA-reranker/runs_bioasq_format_hydrated/submission1-0.64_hydrated.jsonl"
fi

OUTPUT_DIR="${REPO_DIR}/${DATASET}/outputs"

mkdir -p "$OUTPUT_DIR" "${REPO_DIR}/logs"

# ----------------------------------------
# LOAD ENV
# ----------------------------------------

source "${REPO_DIR}/.venv/bin/activate"

if [ -f "${REPO_DIR}/.env" ]; then
    export $(cat "${REPO_DIR}/.env" | xargs)
fi

# ----------------------------------------
# RUN INFERENCE
# ----------------------------------------

echo "========================================"
echo "  Backend  : ${BACKEND}"
echo "  Model    : ${MODEL}"
echo "  Prompts  : ${PROMPT_IDS}"
echo "  Abstracts: ${NUM_SUPPORT}"
echo "  Input    : ${INPUT}"
echo "  Output   : ${OUTPUT_DIR}"
echo "========================================"

uv run python "${REPO_DIR}/inference/run.py" \
    --data-path   "$INPUT" \
    --output-dir  "$OUTPUT_DIR" \
    --backend     "$BACKEND" \
    --model       "$MODEL" \
    --prompt-ids  "$PROMPT_IDS" \
    --num-support "$NUM_SUPPORT" \
    --input-type  "$INPUT_TYPE" \
    --max-tokens  "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size   "$TENSOR_PARALLEL_SIZE" \
    --max-model-len          "$MAX_MODEL_LEN"

echo "========================================"
echo "Done. Output files in: ${OUTPUT_DIR}"
echo "========================================"
