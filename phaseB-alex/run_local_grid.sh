#!/bin/bash
# run_local_grid.sh
#
# Grid runner for local models: loads each model ONCE and runs all specified
# prompt IDs in a single batched inference call. Much faster than run_queue.sh
# for local models because model loading (~28 min for 27B) happens only once
# per model instead of once per prompt.
#
# Usage:
#   bash run_local_grid.sh
#
# Output: one file per (model × prompt_id × context_source × types) combination,
# saved to dev/outputs/exact/ or batch01/outputs/exact/

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-detach under nohup so you can close the terminal
if [ -z "$LOGGING" ]; then
    mkdir -p "${REPO_DIR}/logs"
    TS=$(date +%Y%m%d_%H%M%S)
    LOG="${REPO_DIR}/logs/grid_${TS}.out"
    echo "Logging to: $LOG"
    LOGGING=1 nohup bash "$0" "$@" > "$LOG" 2>&1 &
    echo "Started in background (PID $!). Follow with: tail -f $LOG"
    exit
fi

# ============================================================
# LOCAL MODELS TO RUN
# Each model loads once and runs all PROMPT_IDS below.
# ============================================================

LOCAL_MODELS=(
    "Qwen/Qwen3.5-27B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    "google/medgemma-27b-text-it"
    "mlx-community/Qwen3.5-122B-A10B-4bit"
    "cyankiwi/MiniMax-M2.5-AWQ-4bit"
    "cyankiwi/MiniMax-M2.5-REAP-139B-A10B-AWQ-4bit"
    "cyankiwi/NVIDIA-Nemotron-3-Super-120B-A12B-AWQ-4bit"
    "mlx-community/Llama-3.3-70B-Instruct-4bit"
    "cyankiwi/GLM-4.7-Flash-AWQ-4bit"
)

# ============================================================
# PROMPT IDS
# "all" → runs every prompt in prompts_exact.json
# Or specify a subset: PROMPT_IDS=("1" "4" "5")
# ============================================================

PROMPT_IDS=("all")   # or e.g. ("1" "2" "3" "4" "5" "6")

# ============================================================
# INFERENCE SETTINGS
# ============================================================

TYPES="yesno factoid list"          # space-separated: yesno factoid list
CONTEXT_SOURCE="snippets"           # abstracts (Phase A+) / snippets (Phase B)
DATASET="dev"                       # dev / batch01 / batch01-phaseB

NUM_CONTEXT=6
MAX_TOKENS=4098                      # exact answers are short — no need for 4000+
TEMPERATURE=0.7                     # must be 0 for deterministic exact answers

# GPU settings
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=8192
ENFORCE_EAGER=true                  # true = required for 27B+ models (no headroom for CUDAGraphs)
                                    # false = faster for smaller models (7B-14B) if they don't OOM

# ============================================================
# PATHS
# ============================================================

if [ "$DATASET" = "dev" ]; then
    INPUT="${REPO_DIR}/../data/val_data/13B1_golden_documents.jsonl"
elif [ "$DATASET" = "batch01" ]; then
    INPUT="/home/ucloud/BioASQ13B/phaseA-reranker/runs_bioasq_format_hydrated/submission1-0.64_hydrated.jsonl"
elif [ "$DATASET" = "batch01-phaseB" ]; then
    INPUT="/home/ucloud/BioASQ13B/data/BioASQ-task14bPhaseB-testset1_hydrated.jsonl"
fi

OUTPUT_DIR="${REPO_DIR}/${DATASET}/outputs/exact"
RESULTS_DIR="${REPO_DIR}/${DATASET}/results/exact"
mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR"

# ============================================================
# GRID LOOP
# ============================================================

TOTAL=${#LOCAL_MODELS[@]}
echo "====================================================="
echo "  Grid started: $(date)"
echo "  Models: $TOTAL"
echo "  Prompt IDs: ${PROMPT_IDS[*]}"
echo "  Types: $TYPES"
echo "  Context: $CONTEXT_SOURCE | Dataset: $DATASET"
echo "====================================================="

for i in "${!LOCAL_MODELS[@]}"; do
    MODEL="${LOCAL_MODELS[$i]}"
    MODEL_NUM=$((i + 1))

    echo ""
    echo "-----------------------------------------------------"
    echo "  Model $MODEL_NUM/$TOTAL — $(date)"
    echo "  $MODEL"
    echo "-----------------------------------------------------"

    EAGER_FLAG=""
    if [ "$ENFORCE_EAGER" = "true" ]; then
        EAGER_FLAG="--enforce-eager"
    fi

    # Run all prompts for this model in one call (model loads once)
    uv run python "${REPO_DIR}/inference/run_exact.py" \
        --input           "$INPUT" \
        --output-dir      "$OUTPUT_DIR" \
        --backend         local \
        --model           "$MODEL" \
        --prompt-ids      "${PROMPT_IDS[@]}" \
        --num-snippets    "$NUM_CONTEXT" \
        --max-tokens      "$MAX_TOKENS" \
        --temperature     "$TEMPERATURE" \
        --context-source  "$CONTEXT_SOURCE" \
        --types           $TYPES \
        --tensor-parallel-size    "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization  "$GPU_MEMORY_UTILIZATION" \
        --max-model-len           "$MAX_MODEL_LEN" \
        $EAGER_FLAG

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "  [FAILED] Model $MODEL exited with code $EXIT_CODE — skipping to next model"
        continue
    fi

    # Evaluate each output file produced for this model
    if [ "$DATASET" = "dev" ]; then
        MODEL_SLUG=$(echo "$MODEL" | tr '/' '-' | tr '.' '-' | tr ':' '-')
        TYPES_SLUG=$(echo "$TYPES" | tr ' ' '_')

        # Determine which prompt IDs were actually run
        if [ "${PROMPT_IDS[*]}" = "all" ]; then
            PIDS_TO_EVAL=("1" "2" "3" "4" "5" "6")
        else
            PIDS_TO_EVAL=("${PROMPT_IDS[@]}")
        fi

        for PID in "${PIDS_TO_EVAL[@]}"; do
            for QTYPE in $TYPES; do
                OUTPUT="${OUTPUT_DIR}/${MODEL_SLUG}_p${PID}_${CONTEXT_SOURCE}_${QTYPE}.json"
                REPORT="${RESULTS_DIR}/${MODEL_SLUG}_p${PID}_${CONTEXT_SOURCE}_${QTYPE}.json"
                if [ -f "$OUTPUT" ]; then
                    uv run python "${REPO_DIR}/evaluation/evaluation_exact.py" \
                        --predictions  "$OUTPUT" \
                        --ground-truth "$INPUT" \
                        --output       "$REPORT"
                fi
            done
        done
    fi

    echo "  [DONE] Model $MODEL_NUM/$TOTAL complete"
done

echo ""
echo "====================================================="
echo "  Grid finished: $(date)"
echo "====================================================="
