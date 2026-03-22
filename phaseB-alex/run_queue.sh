#!/bin/bash
# ============================================================
# run_queue.sh — overnight queue for exact answer inference
#
# Usage: bash run_queue.sh
#
# Edit the JOBS list below. Each job is one line:
#   "MODEL | BACKEND | TYPES | PROMPT_ID | CONTEXT_SOURCE | DATASET"
#
# Jobs run one at a time — GPU is fully free between runs.
# ============================================================

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-detach under nohup
if [ -z "$LOGGING" ]; then
    mkdir -p "${REPO_DIR}/logs"
    LOG="${REPO_DIR}/logs/queue_$(date +%Y%m%d_%H%M%S).out"
    echo "Queue logging to: $LOG"
    LOGGING=1 nohup bash "$0" > "$LOG" 2>&1 &
    echo "Started (PID $!). Follow with: tail -f $LOG"
    exit
fi

# ============================================================
# DEFINE YOUR JOBS HERE
# ============================================================

# Model | Backend (local / openrouter) | Types (yesno / factoid / list) | Prompt ID | Context Source (snippets / abstracts) | Dataset (dev / batch01)

JOBS=(
    "Qwen/Qwen3.5-27B        | local | yesno   | 1 | snippets | dev"
    "Qwen/Qwen3.5-27B        | local | yesno   | 2 | snippets | dev"
    "Qwen/Qwen3.5-27B        | local | yesno   | 3 | snippets | dev"
    "Qwen/Qwen3.5-27B        | local | yesno   | 4 | snippets | dev"
    "Qwen/Qwen3.5-27B        | local | factoid   | 1 | snippets | dev"
    "Qwen/Qwen3.5-27B        | local | factoid   | 2 | snippets | dev"
    "Qwen/Qwen3.5-27B        | local | factoid   | 3 | snippets | dev"
    "Qwen/Qwen3.5-27B        | local | factoid   | 4 | snippets | dev"
    "Qwen/Qwen3.5-27B        | local | factoid   | 5 | snippets | dev"
    "Qwen/Qwen3.5-27B        | local | factoid   | 6 | snippets | dev"
    "Qwen/Qwen3.5-27B        | local | list   | 1 | snippets | dev"
    "Qwen/Qwen3.5-27B        | local | list   | 2 | snippets | dev"
    "Qwen/Qwen3.5-27B        | local | list   | 3 | snippets | dev"
    "Qwen/Qwen3.5-27B        | local | list   | 4 | snippets | dev"
    "Qwen/Qwen3.5-27B        | local | list   | 5 | snippets | dev"
    "Qwen/Qwen3.5-27B        | local | list   | 6 | snippets | dev"
    "google/medgemma-27b-text-it        | local | yesno   | 1 | snippets | dev"
    "google/medgemma-27b-text-it        | local | yesno   | 2 | snippets | dev"
    "google/medgemma-27b-text-it        | local | yesno   | 3 | snippets | dev"
    "google/medgemma-27b-text-it        | local | yesno   | 4 | snippets | dev"
    "google/medgemma-27b-text-it        | local | factoid   | 1 | snippets | dev"
    "google/medgemma-27b-text-it        | local | factoid   | 2 | snippets | dev"
    "google/medgemma-27b-text-it        | local | factoid   | 3 | snippets | dev"
    "google/medgemma-27b-text-it        | local | factoid   | 4 | snippets | dev"
    "google/medgemma-27b-text-it        | local | factoid   | 5 | snippets | dev"
    "google/medgemma-27b-text-it        | local | factoid   | 6 | snippets | dev"
    "google/medgemma-27b-text-it        | local | list   | 1 | snippets | dev"
    "google/medgemma-27b-text-it        | local | list   | 2 | snippets | dev"
    "google/medgemma-27b-text-it        | local | list   | 3 | snippets | dev"
    "google/medgemma-27b-text-it        | local | list   | 4 | snippets | dev"
    "google/medgemma-27b-text-it        | local | list   | 5 | snippets | dev"
    "google/medgemma-27b-text-it        | local | list   | 6 | snippets | dev"
    "openai/gpt-oss-20b        | local | yesno   | 1 | snippets | dev"
    "openai/gpt-oss-20b        | local | yesno   | 2 | snippets | dev"
    "openai/gpt-oss-20b        | local | yesno   | 3 | snippets | dev"
    "openai/gpt-oss-20b        | local | yesno   | 4 | snippets | dev"
    "openai/gpt-oss-20b        | local | factoid   | 1 | snippets | dev"
    "openai/gpt-oss-20b        | local | factoid   | 2 | snippets | dev"
    "openai/gpt-oss-20b        | local | factoid   | 3 | snippets | dev"
    "openai/gpt-oss-20b        | local | factoid   | 4 | snippets | dev"
    "openai/gpt-oss-20b        | local | factoid   | 5 | snippets | dev"
    "openai/gpt-oss-20b        | local | factoid   | 6 | snippets | dev"
    "openai/gpt-oss-20b        | local | list   | 1 | snippets | dev"
    "openai/gpt-oss-20b        | local | list   | 2 | snippets | dev"
    "openai/gpt-oss-20b        | local | list   | 3 | snippets | dev"
    "openai/gpt-oss-20b        | local | list   | 4 | snippets | dev"
    "openai/gpt-oss-20b        | local | list   | 5 | snippets | dev"
    "openai/gpt-oss-20b        | local | list   | 6 | snippets | dev"
    "meta-llama/Llama-3.1-8B-Instruct        | local | yesno   | 1 | snippets | dev"
    "meta-llama/Llama-3.1-8B-Instruct        | local | yesno   | 2 | snippets | dev"
    "meta-llama/Llama-3.1-8B-Instruct        | local | yesno   | 3 | snippets | dev"
    "meta-llama/Llama-3.1-8B-Instruct        | local | yesno   | 4 | snippets | dev"
    "meta-llama/Llama-3.1-8B-Instruct        | local | factoid   | 1 | snippets | dev"
    "meta-llama/Llama-3.1-8B-Instruct        | local | factoid   | 2 | snippets | dev"
    "meta-llama/Llama-3.1-8B-Instruct        | local | factoid   | 3 | snippets | dev"
    "meta-llama/Llama-3.1-8B-Instruct        | local | factoid   | 4 | snippets | dev"
    "meta-llama/Llama-3.1-8B-Instruct        | local | factoid   | 5 | snippets | dev"
    "meta-llama/Llama-3.1-8B-Instruct        | local | factoid   | 6 | snippets | dev"
    "meta-llama/Llama-3.1-8B-Instruct        | local | list   | 1 | snippets | dev"
    "meta-llama/Llama-3.1-8B-Instruct        | local | list   | 2 | snippets | dev"
    "meta-llama/Llama-3.1-8B-Instruct        | local | list   | 3 | snippets | dev"
    "meta-llama/Llama-3.1-8B-Instruct        | local | list   | 4 | snippets | dev"
    "meta-llama/Llama-3.1-8B-Instruct        | local | list   | 5 | snippets | dev"
    "meta-llama/Llama-3.1-8B-Instruct        | local | list   | 6 | snippets | dev"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | local | yesno   | 1 | snippets | dev"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | local | yesno   | 2 | snippets | dev"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | local | yesno   | 3 | snippets | dev"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | local | yesno   | 4 | snippets | dev"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | local | factoid   | 1 | snippets | dev"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | local | factoid   | 2 | snippets | dev"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | local | factoid   | 3 | snippets | dev"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | local | factoid   | 4 | snippets | dev"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | local | factoid   | 5 | snippets | dev"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | local | factoid   | 6 | snippets | dev"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | local | list   | 1 | snippets | dev"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | local | list   | 2 | snippets | dev"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | local | list   | 3 | snippets | dev"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | local | list   | 4 | snippets | dev"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | local | list   | 5 | snippets | dev"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B        | local | list   | 6 | snippets | dev"
    "mistralai/Mistral-7B-Instruct-v0.3        | local | yesno   | 1 | snippets | dev"
    "mistralai/Mistral-7B-Instruct-v0.3        | local | yesno   | 2 | snippets | dev"
    "mistralai/Mistral-7B-Instruct-v0.3        | local | yesno   | 3 | snippets | dev"
    "mistralai/Mistral-7B-Instruct-v0.3        | local | yesno   | 4 | snippets | dev"
    "mistralai/Mistral-7B-Instruct-v0.3        | local | factoid   | 1 | snippets | dev"
    "mistralai/Mistral-7B-Instruct-v0.3        | local | factoid   | 2 | snippets | dev"
    "mistralai/Mistral-7B-Instruct-v0.3        | local | factoid   | 3 | snippets | dev"
    "mistralai/Mistral-7B-Instruct-v0.3        | local | factoid   | 4 | snippets | dev"
    "mistralai/Mistral-7B-Instruct-v0.3        | local | factoid   | 5 | snippets | dev"
    "mistralai/Mistral-7B-Instruct-v0.3        | local | factoid   | 6 | snippets | dev"
    "mistralai/Mistral-7B-Instruct-v0.3        | local | list   | 1 | snippets | dev"
    "mistralai/Mistral-7B-Instruct-v0.3        | local | list   | 2 | snippets | dev"
    "mistralai/Mistral-7B-Instruct-v0.3        | local | list   | 3 | snippets | dev"
    "mistralai/Mistral-7B-Instruct-v0.3        | local | list   | 4 | snippets | dev"
    "mistralai/Mistral-7B-Instruct-v0.3        | local | list   | 5 | snippets | dev"
    "mistralai/Mistral-7B-Instruct-v0.3        | local | list   | 6 | snippets | dev"
)

# ============================================================
# SHARED INFERENCE SETTINGS
# ============================================================

NUM_CONTEXT=10
MAX_TOKENS=4098
TEMPERATURE=0.7
REQUEST_DELAY=0.0

# GPU (local only)
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=16384
ENFORCE_EAGER=false   # set to true to force eager execution (no async batching) — useful for debugging

# ============================================================
# RUNNER — do not edit below
# ============================================================

source "${REPO_DIR}/.venv/bin/activate"
if [ -f "${REPO_DIR}/.env" ]; then
    export $(cat "${REPO_DIR}/.env" | xargs)
fi

TOTAL=${#JOBS[@]}
echo "====================================================="
echo "  Queue started: $(date)   Total jobs: $TOTAL"
echo "====================================================="

for i in "${!JOBS[@]}"; do
    IFS='|' read -r MODEL BACKEND TYPES PROMPT_ID CONTEXT_SOURCE DATASET <<< "${JOBS[$i]}"
    MODEL=$(echo "$MODEL" | xargs)
    BACKEND=$(echo "$BACKEND" | xargs)
    TYPES=$(echo "$TYPES" | xargs)
    PROMPT_ID=$(echo "$PROMPT_ID" | xargs)
    CONTEXT_SOURCE=$(echo "$CONTEXT_SOURCE" | xargs)
    DATASET=$(echo "$DATASET" | xargs)

    if [ "$DATASET" = "dev" ]; then
        INPUT="${REPO_DIR}/../data/val_data/13B1_golden_documents.jsonl"
    elif [ "$DATASET" = "batch01" ]; then
        INPUT="/home/ucloud/BioASQ13B/phaseA-reranker/runs_bioasq_format_hydrated/submission1-0.64_hydrated.jsonl"
    elif [ "$DATASET" = "batch01-phaseB" ]; then
        INPUT="/home/ucloud/BioASQ13B/data/BioASQ-task14bPhaseB-testset1_hydrated.jsonl"
    fi

    MODEL_SLUG=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')
    OUTPUT_DIR="${REPO_DIR}/${DATASET}/outputs/exact"
    RESULTS_DIR="${REPO_DIR}/${DATASET}/results/exact"
    OUTPUT="${OUTPUT_DIR}/${MODEL_SLUG}_p${PROMPT_ID}_${CONTEXT_SOURCE}_${TYPES}.json"
    REPORT="${RESULTS_DIR}/${MODEL_SLUG}_p${PROMPT_ID}_${CONTEXT_SOURCE}_${TYPES}.json"
    mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR"

    echo ""
    echo "-----------------------------------------------------"
    echo "  Job $((i+1))/$TOTAL — $(date)"
    echo "  $MODEL | $BACKEND | $TYPES | prompt $PROMPT_ID | $CONTEXT_SOURCE | $DATASET"
    echo "-----------------------------------------------------"

    if [ -f "$OUTPUT" ]; then
        echo "  SKIPPED — output already exists"
        continue
    fi

    EAGER_FLAG=""
    [ "$ENFORCE_EAGER" = "true" ] && EAGER_FLAG="--enforce-eager"

    uv run python "${REPO_DIR}/inference/run_exact.py" \
        --input                  "$INPUT" \
        --output                 "$OUTPUT" \
        --backend                "$BACKEND" \
        --model                  "$MODEL" \
        --prompt-id              "$PROMPT_ID" \
        --num-snippets           "$NUM_CONTEXT" \
        --max-tokens             "$MAX_TOKENS" \
        --temperature            "$TEMPERATURE" \
        --context-source         "$CONTEXT_SOURCE" \
        --tensor-parallel-size   "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --max-model-len          "$MAX_MODEL_LEN" \
        --request-delay          "$REQUEST_DELAY" \
        $EAGER_FLAG \
        --types                  $TYPES

    if [ $? -ne 0 ]; then
        echo "  FAILED — skipping to next job"
        continue
    fi

    if [ "$DATASET" = "dev" ]; then
        uv run python "${REPO_DIR}/evaluation/evaluation_exact.py" \
            --predictions  "$OUTPUT" \
            --ground-truth "$INPUT" \
            --output       "$REPORT" \
            --types        $TYPES
        echo "  Evaluation done → $REPORT"
    fi
done

echo ""
echo "====================================================="
echo "  Queue finished: $(date)"
echo "====================================================="