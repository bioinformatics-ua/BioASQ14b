#!/bin/bash
# run_reconcile_exact.sh
#
# LLM-based reconciliation for factoid and list exact answers.
#
# Unlike ensembling (majority vote / RRF), reconciliation makes a new LLM call
# per question that sees ALL candidate answers and synthesises the best one.
#
# Factoid: fixes partial/rephrased answers — e.g. "FOLFOXIRI Plus Bevacizumab"
#          vs "mFOLFOXIRI and Bevacizumab" → reconciler picks the complete form.
#
# List:    takes the union of all model candidates, then verifies which entities
#          actually answer the question — combines recall + precision automatically.
#
# Usage:
#   bash run_reconcile_exact.sh

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$LOGGING" ]; then
    mkdir -p "${REPO_DIR}/logs"
    RUN_NUM=1
    while [ -f "${REPO_DIR}/logs/run_reconcile${RUN_NUM}.out" ]; do
        RUN_NUM=$((RUN_NUM + 1))
    done
    LOG="${REPO_DIR}/logs/run_reconcile${RUN_NUM}.out"
    echo "Logging to: $LOG"
    LOGGING=1 nohup bash "$0" "$@" > "$LOG" 2>&1 &
    echo "Started in background (PID $!). Follow with: tail -f $LOG"
    exit
fi

# ============================================================
# WHAT TO RECONCILE
# ============================================================

QTYPE="factoid"          # factoid / list

DATASET="batch01-phaseB"            # dev / batch01 / batch01-phaseB

# Input files — filenames only (no path, no extension needed unless .json provided)
# Must exist in dev/outputs/exact/ (or batch01/outputs/exact/)
# Use your best performing outputs — more diverse models = better reconciliation
INPUT_FILES=(
    # "google-gemini-2-0-flash-001_p3_abstracts_factoid.json"      # MRR 0.5577
    # "google-gemini-2-0-flash-001_p5_abstracts_factoid.json"
    # "google-gemini-2-0-flash-001_p5_snippets_factoid.json"
    # "anthropic-claude-sonnet-4-6_p5_abstracts_factoid.json"      # MRR 0.5385
    # "anthropic-claude-sonnet-4-6_p5_snippets_factoid.json"
    # "openai-gpt-oss-20b_p2_snippets_factoid.json"                # MRR 0.5321
    # "x-ai-grok-4-20-beta_p5_abstracts_factoid.json"              # MRR 0.5128

    # "qwen-qwen3-max-thinking_p6_abstracts_list.json"
    # "openrouter-hunter-alpha_p6_abstracts_list.json"
    # "google-gemini-2-0-flash-001_p6_abstracts_list.json"
    # "openai-gpt-oss-20b_p6_snippets_list.json"
    # "google-gemini-2-0-flash-001_p5_abstracts_list.json"
    # "google-gemini-2-5-flash_p6_abstracts_list.json"

    "google-gemini-2-0-flash-001_p3_snippets_factoid.json"
    "anthropic-claude-sonnet-4-6_p5_snippets_factoid.json"
    "google-gemini-2-0-flash-001_p3_abstracts_factoid.json"
    "qwen-qwen3-max-thinking_p6_abstracts_factoid.json"
)

# Output filename — saved to dev/outputs/reconciled/
OUTPUT_NAME="reconciled_${QTYPE}"

# ============================================================
# RECONCILER MODEL
# The LLM that reads all candidates and synthesises the best answer.
# A strong, instruction-following model works best here.
# ============================================================

MODEL="qwen/qwen3-max-thinking"
BACKEND="openrouter"          # openrouter / local

CONTEXT_SOURCE="snippets"    # abstracts (Phase A+) / snippets (Phase B)
NUM_CONTEXT=10                # number of abstracts/snippets to include as context
MAX_TOKENS=4096               # reconciled answers can be slightly longer
TEMPERATURE=0.7
REQUEST_DELAY=0.0             # set to 4.0 for free OpenRouter models

# GPU settings (only used when BACKEND=local)
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=8192

# ============================================================
# PATHS
# ============================================================

if [ "$DATASET" = "dev" ]; then
    GROUND_TRUTH="${REPO_DIR}/../data/val_data/13B1_golden_documents.jsonl"
elif [ "$DATASET" = "batch01" ]; then
    GROUND_TRUTH="/home/ucloud/BioASQ13B/phaseA-reranker/runs_bioasq_format_hydrated/submission1-0.64_hydrated.jsonl"
elif [ "$DATASET" = "batch01-phaseB" ]; then
    GROUND_TRUTH="/home/ucloud/BioASQ13B/data/BioASQ-task14bPhaseB-testset1_hydrated.jsonl"
fi

INFERENCE_DIR="${REPO_DIR}/${DATASET}/outputs/exact"
OUTPUT_DIR="${REPO_DIR}/${DATASET}/outputs/reconciled"
RESULTS_DIR="${REPO_DIR}/${DATASET}/results/reconciled"
OUTPUT="${OUTPUT_DIR}/${OUTPUT_NAME}.json"
REPORT="${RESULTS_DIR}/${OUTPUT_NAME}.json"

mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR"

# Build full input paths
RESOLVED_INPUTS=()
for name in "${INPUT_FILES[@]}"; do
    RESOLVED_INPUTS+=("${INFERENCE_DIR}/${name}")
done

# ============================================================
# LOAD ENV
# ============================================================

source "${REPO_DIR}/.venv/bin/activate"

if [ -f "${REPO_DIR}/.env" ]; then
    export $(cat "${REPO_DIR}/.env" | xargs)
fi

echo "========================================"
echo "  Type      : $QTYPE"
echo "  Dataset   : $DATASET"
echo "  Model     : $MODEL"
echo "  Inputs    : ${#RESOLVED_INPUTS[@]} files"
for f in "${RESOLVED_INPUTS[@]}"; do echo "    $f"; done
echo "  Output    : $OUTPUT"
echo "========================================"

# ============================================================
# 1 — RECONCILE
# ============================================================

uv run python "${REPO_DIR}/inference/reconcile_exact.py" \
    --inputs          "${RESOLVED_INPUTS[@]}" \
    --ground-truth    "$GROUND_TRUTH" \
    --output          "$OUTPUT" \
    --type            "$QTYPE" \
    --model           "$MODEL" \
    --backend         "$BACKEND" \
    --context-source  "$CONTEXT_SOURCE" \
    --num-context     "$NUM_CONTEXT" \
    --max-tokens      "$MAX_TOKENS" \
    --temperature     "$TEMPERATURE" \
    --request-delay   "$REQUEST_DELAY" \
    --tensor-parallel-size    "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization  "$GPU_MEMORY_UTILIZATION" \
    --max-model-len           "$MAX_MODEL_LEN"

echo "[1/2] Reconciliation done."

# ============================================================
# 2 — EVALUATE (dev only)
# ============================================================

if [ "$DATASET" = "dev" ]; then
    uv run python "${REPO_DIR}/evaluation/evaluation_exact.py" \
        --predictions  "$OUTPUT" \
        --ground-truth "$GROUND_TRUTH" \
        --output       "$REPORT" \
        --types        "$QTYPE"
    echo "[2/2] Evaluation done. Report: $REPORT"
else
    echo "[2/2] Skipping evaluation (no gold answers on batch01)."
fi

echo "========================================"
