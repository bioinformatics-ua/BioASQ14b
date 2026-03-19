#!/bin/bash
# run_sweep.sh — Overnight parameter sweep for Phase B inference
#
# USAGE:
#   bash phaseB/run_sweep.sh              # runs in background, logs to phaseB/logs/sweep{N}.out
#   LOGGING=1 bash phaseB/run_sweep.sh    # run in foreground (no nohup)
#
# SKIP LOGIC: if all output files for a (model, input_file, input_type, prompts_file)
#   combination already exist, the combination is skipped. Safe to re-run after partial failure.
#
# OUTPUT LAYOUT:
#   phaseB/dev/sweep_outputs/{input_basename}/{model_slug}_{input_type}_{num_support}_{pid}.json

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Auto-logging ──────────────────────────────────────────────────────────────
if [ -z "$LOGGING" ]; then
    mkdir -p "${REPO_DIR}/logs"
    N=1; while [ -f "${REPO_DIR}/logs/sweep${N}.out" ]; do N=$((N+1)); done
    LOG="${REPO_DIR}/logs/sweep${N}.out"
    echo "Logging to: $LOG"
    LOGGING=1 nohup bash "$0" "$@" > "$LOG" 2>&1 &
    echo "Started sweep in background (PID $!). Follow with: tail -f $LOG"
    exit
fi

# ── Never exit on error — a failed model run must not kill the sweep ──────────
set +e

# ── Setup ─────────────────────────────────────────────────────────────────────
source "${REPO_DIR}/.venv/bin/activate" || true
if [ -f "${REPO_DIR}/.env" ]; then
    export $(grep -v '^#' "${REPO_DIR}/.env" | xargs)
fi

# ════════════════════════════════════════════════════════════════════════════
# SWEEP CONFIGURATION — edit this block before running
# ════════════════════════════════════════════════════════════════════════════

# ── Input files (the 5 hydrated batches) ─────────────────────────────────────
INPUT_FILES=(
    "/home/ucloud/BioASQ13B/phaseA-reranker/runs_bioasq_format_hydrated/submission1-0.64_hydrated.jsonl"
    # "/home/ucloud/BioASQ13B/phaseA-reranker/runs_bioasq_format_hydrated/submission2-dprf_0.64_hydrated.jsonl"
    # "/home/ucloud/BioASQ13B/phaseA-reranker/runs_bioasq_format_hydrated/submission3-all_trained_dprf_hydrated.jsonl"
    # "/home/ucloud/BioASQ13B/phaseA-reranker/runs_bioasq_format_hydrated/submission4-2025_dprf_hydrated.jsonl"
    # "/home/ucloud/BioASQ13B/phaseA-reranker/runs_bioasq_format_hydrated/submission5-all_of_all_dprf_hydrated.jsonl"

    # "/home/ucloud/BioASQ13B/data/val_data/13B2_golden_documents.jsonl"
    # "/home/ucloud/BioASQ13B/data/val_data/13B3_golden_documents.jsonl"
    # "/home/ucloud/BioASQ13B/data/val_data/13B4_golden_documents.jsonl"
    # "/home/ucloud/BioASQ13B/data/val_data/13B5_golden_documents.jsonl"
)

# ── Cloud models (OpenRouter) ─────────────────────────────────────────────────
# Coordinator picks: Gemini 2.5 Flash, Claude Opus, GPT
CLOUD_MODELS=(
    "anthropic/claude-opus-4.6"
    # "google/gemini-2.5-flash"
    # "x-ai/grok-4.1-fast"
    # "qwen/qwen3-32b" 
)


# ── Local models (vLLM) ───────────────────────────────────────────────────────
# Format: "hf_model_id|tensor_parallel_size|gpu_memory_utilization|max_model_len"
# 2x A40 (48 GB each, PCIe): tp=1 → single GPU (~43 GB), tp=2 → both (~86 GB)
LOCAL_MODELS=(
    # "Qwen/Qwen3.5-35B-A3B|2|0.90|8192"
    # "google/gemma-3-27b-it|2|0.90|8192"
    # "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16|2|0.90|8192"              # medical-specialized, ~54 GB bf16
    # "google/gemma-3-27b-it|2|0.90|8192"                  # general gemma, same size
)

# ── Prompts ───────────────────────────────────────────────────────────────────
# Prompt 5 = best last year (few-shot CoT, plain text answer)
# Prompt 6 = new this year  (XML structure, reasoning + answer in JSON)
PROMPT_CONFIGS=(
    "${REPO_DIR}/inference/prompts_generic.json|5,6"
)

# ── Context size ──────────────────────────────────────────────────────────────
NUM_SUPPORT_VALUES="5,10"

# ── Input types ───────────────────────────────────────────────────────────────
INPUT_TYPES=(
    "abstracts"
    # "snippets"
)

# ── Inference parameters ──────────────────────────────────────────────────────
MAX_TOKENS=4000
TEMPERATURE=0.0

# GPU defaults (used only if a local model entry omits its own values)
DEFAULT_TP=1
DEFAULT_GPU_MEM=0.90
DEFAULT_MAX_MODEL_LEN=8192

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_BASE="${REPO_DIR}/dev/sweep_outputs"
mkdir -p "$OUTPUT_BASE"

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

# Returns the model slug used in output filenames (mirrors run.py logic)
model_slug() {
    # run.py uses the last component of the model path, replacing / with -
    echo "$1" | awk -F'/' '{print $NF}'
}

# Check if all expected output files exist (skip if yes)
all_outputs_exist() {
    local OUTPUT_DIR="$1"
    local MODEL_SLUG="$2"
    local INPUT_TYPE="$3"
    local PROMPT_IDS="$4"   # comma-separated
    local NUM_SUPPORT="$5"  # comma-separated

    IFS=',' read -ra PIDS  <<< "$PROMPT_IDS"
    IFS=',' read -ra NSUPS <<< "$NUM_SUPPORT"
    for pid in "${PIDS[@]}"; do
        pid=$(echo "$pid" | tr -d ' ')
        for ns in "${NSUPS[@]}"; do
            ns=$(echo "$ns" | tr -d ' ')
            local F="${OUTPUT_DIR}/${MODEL_SLUG}_${INPUT_TYPE}_${ns}_${pid}.json"
            [ ! -f "$F" ] && return 1
        done
    done
    return 0
}

# Run one combination
# Args: MODEL BACKEND INPUT PROMPTS_FILE PROMPT_IDS INPUT_TYPE [TP] [GPU_MEM] [MAX_LEN]
run_one() {
    local MODEL="$1"
    local BACKEND="$2"
    local INPUT="$3"
    local PROMPTS_FILE="$4"
    local PROMPT_IDS="$5"
    local INPUT_TYPE="$6"
    local TP="${7:-$DEFAULT_TP}"
    local GPU_MEM="${8:-$DEFAULT_GPU_MEM}"
    local MAX_LEN="${9:-$DEFAULT_MAX_MODEL_LEN}"

    local INPUT_BASENAME
    INPUT_BASENAME=$(basename "$INPUT" .jsonl)
    local OUTPUT_DIR="${OUTPUT_BASE}/${INPUT_BASENAME}"
    mkdir -p "$OUTPUT_DIR"

    local SLUG
    SLUG=$(model_slug "$MODEL")
    local PFILE_NAME
    PFILE_NAME=$(basename "$PROMPTS_FILE" .json)

    if all_outputs_exist "$OUTPUT_DIR" "$SLUG" "$INPUT_TYPE" "$PROMPT_IDS" "$NUM_SUPPORT_VALUES"; then
        echo "[SKIP] ${MODEL} | ${INPUT_TYPE} | ${PFILE_NAME} | prompts=${PROMPT_IDS} | support=${NUM_SUPPORT_VALUES}"
        return 0
    fi

    echo "────────────────────────────────────────"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Backend   : ${BACKEND}"
    echo "  Model     : ${MODEL}"
    echo "  File      : ${INPUT_BASENAME}"
    echo "  Type      : ${INPUT_TYPE}"
    echo "  Prompts   : ${PFILE_NAME} [${PROMPT_IDS}]"
    echo "  Support   : ${NUM_SUPPORT_VALUES}"
    echo "  GPU       : tp=${TP}, mem=${GPU_MEM}, maxlen=${MAX_LEN}"
    echo "  Output    : ${OUTPUT_DIR}"
    echo "────────────────────────────────────────"

    uv run python "${REPO_DIR}/inference/run.py" \
        --data-path    "$INPUT" \
        --output-dir   "$OUTPUT_DIR" \
        --backend      "$BACKEND" \
        --model        "$MODEL" \
        --prompts-file "$PROMPTS_FILE" \
        --prompt-ids   "$PROMPT_IDS" \
        --num-support  "$NUM_SUPPORT_VALUES" \
        --input-type   "$INPUT_TYPE" \
        --max-tokens   "$MAX_TOKENS" \
        --temperature  "$TEMPERATURE" \
        --gpu-memory-utilization "$GPU_MEM" \
        --tensor-parallel-size   "$TP" \
        --max-model-len          "$MAX_LEN"

    local RC=$?
    if [ $RC -ne 0 ]; then
        echo "[ERROR] Exit code $RC — ${MODEL} | ${INPUT_TYPE} | ${PFILE_NAME}"
    else
        echo "[OK] Done — ${MODEL} | ${INPUT_TYPE} | ${PFILE_NAME}"
    fi
    return $RC
}

# ════════════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ════════════════════════════════════════════════════════════════════════════

SWEEP_START=$(date)
TOTAL=0
FAILED=0

echo "════════════════════════════════════════"
echo "  SWEEP STARTED: $SWEEP_START"
echo "  Input files  : ${#INPUT_FILES[@]}"
echo "  Cloud models : ${#CLOUD_MODELS[@]}"
echo "  Local models : ${#LOCAL_MODELS[@]}"
echo "  Input types  : ${INPUT_TYPES[*]}"
echo "  Prompt cfgs  : ${#PROMPT_CONFIGS[@]}"
echo "  Num support  : ${NUM_SUPPORT_VALUES}"
echo "════════════════════════════════════════"

for INPUT in "${INPUT_FILES[@]}"; do
    if [ ! -f "$INPUT" ]; then
        echo "[WARN] File not found, skipping: $INPUT"
        continue
    fi

    # ── Cloud models ──────────────────────────────────────────────────────────
    for MODEL in "${CLOUD_MODELS[@]}"; do
        for PCFG in "${PROMPT_CONFIGS[@]}"; do
            PROMPTS_FILE="${PCFG%%|*}"
            PROMPT_IDS="${PCFG##*|}"
            for INPUT_TYPE in "${INPUT_TYPES[@]}"; do
                run_one "$MODEL" "openrouter" "$INPUT" \
                    "$PROMPTS_FILE" "$PROMPT_IDS" "$INPUT_TYPE" \
                    || FAILED=$((FAILED+1))
                TOTAL=$((TOTAL+1))
            done
        done
    done

    # ── Local models (model loaded once per call — all prompts/support batched) ─
    # Each entry: "model_id|tp|gpu_mem|max_model_len"
    # || FAILED=... guarantees the loop always continues, even on crash/OOM/SIGKILL
    for MODEL_CFG in "${LOCAL_MODELS[@]}"; do
        IFS='|' read -r MODEL TP GPU_MEM MAX_LEN <<< "$MODEL_CFG"
        TP="${TP:-$DEFAULT_TP}"
        GPU_MEM="${GPU_MEM:-$DEFAULT_GPU_MEM}"
        MAX_LEN="${MAX_LEN:-$DEFAULT_MAX_MODEL_LEN}"
        for PCFG in "${PROMPT_CONFIGS[@]}"; do
            PROMPTS_FILE="${PCFG%%|*}"
            PROMPT_IDS="${PCFG##*|}"
            for INPUT_TYPE in "${INPUT_TYPES[@]}"; do
                run_one "$MODEL" "local" "$INPUT" \
                    "$PROMPTS_FILE" "$PROMPT_IDS" "$INPUT_TYPE" \
                    "$TP" "$GPU_MEM" "$MAX_LEN" \
                    || FAILED=$((FAILED+1))
                TOTAL=$((TOTAL+1))
            done
        done
    done

done

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════"
echo "  SWEEP COMPLETE"
echo "  Start  : $SWEEP_START"
echo "  End    : $(date)"
echo "  Total  : $TOTAL runs"
echo "  Failed : $FAILED runs"
echo "  Outputs: $OUTPUT_BASE"
echo "════════════════════════════════════════"

# ════════════════════════════════════════════════════════════════════════════
# OPTIONAL: POST-SWEEP SYNTHESIS
# Uncomment and configure to automatically synthesize after all inference runs.
# synthesis/synthesize.py takes multiple run output files and combines them.
# ════════════════════════════════════════════════════════════════════════════
#
# SYNTH_MODEL="google/gemini-2.0-flash-001"
# SYNTH_BACKEND="openrouter"
# SYNTH_PROMPT_IDS="1,2,3,4,5,6"
# SYNTH_PROMPTS_FILE="${REPO_DIR}/synthesis/prompts.json"
# SYNTH_OUT_DIR="${REPO_DIR}/dev/sweep_synthesis"
# mkdir -p "$SYNTH_OUT_DIR"
#
# for INPUT in "${INPUT_FILES[@]}"; do
#     INPUT_BASENAME=$(basename "$INPUT" .jsonl)
#     RUN_DIR="${OUTPUT_BASE}/${INPUT_BASENAME}"
#
#     # Collect all output files for this input
#     mapfile -t RUN_FILES < <(find "$RUN_DIR" -name "*.json" | sort)
#
#     if [ ${#RUN_FILES[@]} -eq 0 ]; then
#         echo "[SYNTH SKIP] No outputs found for $INPUT_BASENAME"
#         continue
#     fi
#
#     echo "[SYNTH] Synthesizing ${#RUN_FILES[@]} runs for $INPUT_BASENAME"
#     uv run python "${REPO_DIR}/synthesis/synthesize.py" \
#         "${RUN_FILES[@]}" \
#         --data-path    "$INPUT" \
#         --output-dir   "$SYNTH_OUT_DIR" \
#         --out-id       "$INPUT_BASENAME" \
#         --backend      "$SYNTH_BACKEND" \
#         --model        "$SYNTH_MODEL" \
#         --prompts-file "$SYNTH_PROMPTS_FILE" \
#         --prompt-ids   "$SYNTH_PROMPT_IDS" \
#         --max-tokens   "$MAX_TOKENS" \
#         --temperature  "$TEMPERATURE"
# done
