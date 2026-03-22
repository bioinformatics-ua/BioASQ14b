
# Batch1 scores Phase A+ last year
# Type	    Official metric	    Best score	    Best system(s)
# yes/no	macro F1	        1.0000	        Multiple (UR-IW-1/2/4, bious2-5, deepseek-r1:32b, etc.)
# factoid	MRR	                0.4615	        Using KG for list q / Main pipeline / Fleming-1/2/3
# list	    mean F1	            0.3223	        UR-IW-5

#!/bin/bash

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-detach: re-launch under nohup if not already logging
if [ -z "$LOGGING" ]; then
    mkdir -p "${REPO_DIR}/logs"
    RUN_NUM=1
    while [ -f "${REPO_DIR}/logs/run_exact${RUN_NUM}.out" ]; do
        RUN_NUM=$((RUN_NUM + 1))
    done
    LOG="${REPO_DIR}/logs/run_exact${RUN_NUM}.out"
    echo "Logging to: $LOG"
    LOGGING=1 nohup bash "$0" "$@" > "$LOG" 2>&1 &
    echo "Started in background (PID $!). Follow with: tail -f $LOG"
    exit
fi

# ----------------------------------------
# CONFIGURABLE VARIABLES
# ----------------------------------------

    # --- Local Models ---
    # qwen/qwen3.5-35b-a3b
    # qwen/qwen3.5-flash-02-23
    # qwen/qwen3.5-397b-a17b
    # qwen/qwen3.5-plus-02-15
    # qwen/qwen3-30b-a3b-instruct-2507
    # qwen/qwen3-235b-a22b-2507
    # deepseek/deepseek-v3.2
    # openai/gpt-oss-120b
    # openai/gpt-oss-20b

    # nvidia/nemotron-3-super-120b-a12b:free
    # nvidia/nemotron-3-nano-30b-a3b:free
    # google/gemma-3-27b-it:free
    # minimax/minimax-m2.5:free
    # mistralai/mistral-small-3.1-24b-instruct:free
    # meta-llama/llama-3.3-70b-instruct:free
    # openai/gpt-oss-120b:free
    # qwen/qwen3-next-80b-a3b-instruct:free
    # arcee-ai/trinity-large-preview:free
    # stepfun/step-3.5-flash:free

    # google/medgemma-27b-text-it
    # qwen/qwen3-32b
    # khazarai/Bio-8B-it
    # AdaptLLM/biomed-Qwen2.5-VL-3B-Instruct
    # xw1234gan/Merging_Qwen2.5-1.5B-Instruct_MedQA_lr1e-05_mb2_ga128_n2048_seed42
    # MBZUAI/MedMO-4B-Next
    # ZJU-AI4H/Hulu-Med-30A3

    # --- Cloud Models (OpenRouter) ---
    # google/gemini-2.5-flash
    # google/gemini-2.5-flash-lite
    # google/gemini-2.5-flash-lite-preview-09-2025
    # google/gemini-2.5-pro
    # google/gemini-2.0-flash-001
    # google/gemini-3.1-flash-lite-preview
    # google/gemini-3.1-pro-preview
    # google/gemini-3-flash-preview
    # google/gemini-3-pro-preview
    # anthropic/claude-sonnet-4-6
    # anthropic/claude-sonnet-4-5
    # anthropic/claude-sonnet-4
    # anthropic/claude-haiku-4-5
    # anthropic/claude-opus-4.6
    # anthropic/claude-opus-4.5
    # deepseek/deepseek-chat-v3-0324    
    # openrouter/hunter-alpha
    # openrouter/healer-alpha
    # x-ai/grok-4.20-beta
    # x-ai/grok-4.1-fast
    # x-ai/grok-4-fast
    # openai/gpt-5.4-pro
    # openai/gpt-5.4
    # openai/gpt-5.4-nano
    # openai/gpt-5.4-mini
    # openai/gpt-5.3-chat
    # openai/gpt-5.2
    # openai/gpt-5.1
    # openai/gpt-5
    # openai/gpt-5-mini
    # openai/gpt-5-nano
    # openai/gpt-4.1-mini
    # z-ai/glm-5-turbo
    # z-ai/glm-5
    # qwen/qwen3-max-thinking
    # moonshotai/kimi-k2.5
    # xiaomi/mimo-v2-flash

    # --- Best Models ---
    # google/gemini-2.0-flash-001
    # qwen/qwen3-max-thinking
    # openai/gpt-oss-120b
    # anthropic/claude-sonnet-4-6

    # nvidia/nemotron-3-super-120b-a12b:free
    # nvidia/nemotron-3-nano-30b-a3b:free
    # google/gemma-3-27b-it:free
    # minimax/minimax-m2.5:free
    # mistralai/mistral-small-3.1-24b-instruct:free
    # meta-llama/llama-3.3-70b-instruct:free
    # openai/gpt-oss-120b:free
    # qwen/qwen3-next-80b-a3b-instruct:free
    # arcee-ai/trinity-large-preview:free
    # stepfun/step-3.5-flash:free

    # google/medgemma-27b-text-it
    # google/gemma-3-27b-it
    # qwen/qwen3-32b
    # Qwen/Qwen3.5-27B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
    # meta-llama/Llama-3.1-8B-Instruct
    # openai/gpt-oss-20b
    # mistralai/Mistral-7B-Instruct-v0.3

    # minimax/minimax-m2.5
    # deepseek/deepseek-v3.2
    # moonshotai/kimi-k2.5
    # openai/gpt-oss-120b
    # xiaomi/mimo-v2-flash

    # deepseek/deepseek-v3.2
    # qwen/qwen3-235b-a22b-2507

MODEL="qwen/qwen3-235b-a22b-2507"

PROMPT_ID="3"

BACKEND="openrouter"         # local / openrouter
CONTEXT_SOURCE="snippets"   # abstracts (Phase A+) / snippets (Phase B)

# Which types to run — space-separated subset of: yesno factoid list
TYPES="factoid"

DATASET="batch01-phaseB"                  # dev / batch01 / batch01-phaseB

# ----------------------------------------
# INFERENCE SETTINGS
# ----------------------------------------

NUM_CONTEXT=10
MAX_TOKENS=4098
TEMPERATURE=0.7
REQUEST_DELAY=0.0          # seconds between requests — set to 4.0 for free OpenRouter models, 0.0 for paid

# GPU settings (only used when BACKEND=local)
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=8192
ENFORCE_EAGER=true   # true = disable CUDAGraphs/torch.compile (safer, slower); false = faster inference

# ----------------------------------------
# PATHS
# ----------------------------------------

if [ "$DATASET" = "dev" ]; then
    INPUT="${REPO_DIR}/../data/val_data/13B1_golden_documents.jsonl"
elif [ "$DATASET" = "batch01" ]; then
    INPUT="/home/ucloud/BioASQ13B/phaseA-reranker/runs_bioasq_format_hydrated/submission1-0.64_hydrated.jsonl"
elif [ "$DATASET" = "batch01-phaseB" ]; then
    INPUT="/home/ucloud/BioASQ13B/data/BioASQ-task14bPhaseB-testset1_hydrated.jsonl"
fi

MODEL_SLUG=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')
TYPES_SLUG=$(echo "$TYPES" | tr ' ' '_')
OUTPUT_DIR="${REPO_DIR}/${DATASET}/outputs/exact"
RESULTS_DIR="${REPO_DIR}/${DATASET}/results/exact"
OUTPUT="${OUTPUT_DIR}/${MODEL_SLUG}_p${PROMPT_ID}_${CONTEXT_SOURCE}_${TYPES_SLUG}.json"
REPORT="${RESULTS_DIR}/${MODEL_SLUG}_p${PROMPT_ID}_${CONTEXT_SOURCE}_${TYPES_SLUG}.json"

mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR" "${REPO_DIR}/logs"

# ----------------------------------------
# LOAD ENV
# ----------------------------------------

source "${REPO_DIR}/.venv/bin/activate"

if [ -f "${REPO_DIR}/.env" ]; then
    export $(cat "${REPO_DIR}/.env" | xargs)
fi

# ----------------------------------------
# RUN
# ----------------------------------------

echo "========================================"
echo "  Backend : ${BACKEND}"
echo "  Model   : ${MODEL}"
echo "  Types   : ${TYPES}"
echo "  Context : ${CONTEXT_SOURCE}"
echo "  Prompt  : ${PROMPT_ID}"
echo "  Output  : ${OUTPUT}"
echo "  Report  : ${REPORT}"
echo "========================================"

# ----------------------------------------
# STEP 1 — Inference
# ----------------------------------------

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
    $EAGER_FLAG \
    --request-delay          "$REQUEST_DELAY" \
    --types                  $TYPES

echo "[1/2] Inference done."

# ----------------------------------------
# STEP 2 — Evaluation (dev only)
# ----------------------------------------

if [ "$DATASET" = "dev" ]; then
    uv run python "${REPO_DIR}/evaluation/evaluation_exact.py" \
        --predictions  "$OUTPUT" \
        --ground-truth "$INPUT" \
        --output       "$REPORT" \
        --types        $TYPES
    echo "[2/2] Evaluation done. Report: ${REPORT}"
else
    echo "[2/2] Skipping evaluation (no gold answers on test set)."
fi
echo "========================================"
