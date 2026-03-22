#!/bin/bash
# =============================================================================
#  BioASQ Phase B — Full Automated Pipeline
#
#  Generate (4 models × 3 prompts × 2 contexts) → Judge → Select → Merge
#
#  Auto-detaches under nohup with logging. Re-runnable: generation skips
#  existing files automatically.
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# ── Auto-detach under nohup ─────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

if [ -z "${RUNNING_UNDER_NOHUP:-}" ]; then
    echo "Detaching under nohup. Log: ${LOGFILE}"
    RUNNING_UNDER_NOHUP=1 nohup bash "$0" "$@" > "$LOGFILE" 2>&1 &
    PID=$!
    echo "PID: ${PID}"
    echo "Follow: tail -f ${LOGFILE}"
    exit 0
fi

# ── Configuration ────────────────────────────────────────────────────────────

INPUT_DATA="../data/BioASQ-task14bPhaseB-testset1_hydrated.jsonl"
DATASET="phaseB-testset1"

# Models to generate with
MODELS=(
    "google/gemini-2.5-flash"
    "qwen/qwen3.5-122b-a10b"
    "google/gemini-2.0-flash-001"
)

# Prompt / context grid
PROMPT_IDS="5 6"
CONTEXT_SOURCES=("abstracts" "snippets")
TYPES="summary yesno factoid list"

# Generation settings
GEN_MAX_TOKENS=4096
GEN_TEMPERATURE=0.7
GEN_REQUEST_DELAY=0.0
NUM_CONTEXT=10

# Judge settings (ideal_answer + reasoning + scores — needs headroom vs. score-only)
JUDGE_MODEL="google/gemini-2.5-flash"
JUDGE_MAX_TOKENS=2048
JUDGE_TEMPERATURE=0.7
JUDGE_REQUEST_DELAY=0.0

# Merge settings — 5 submissions with different top-K diversity
MERGE_MODEL="anthropic/claude-sonnet-4-6"
MERGE_MAX_TOKENS=1512
MERGE_TEMPERATURE=0.7
MERGE_REQUEST_DELAY=0.0
MERGE_TOP_KS=(1 2 3 5 50)   # 5 submissions: top-1 .. top-all

# Output paths
EXACT_DIR="${REPO_DIR}/${DATASET}/outputs/exact"
JUDGE_DIR="${REPO_DIR}/${DATASET}/outputs/judged"
SELECT_DIR="${REPO_DIR}/${DATASET}/outputs/selected"
MERGE_DIR="${REPO_DIR}/${DATASET}/outputs/merged"

mkdir -p "$EXACT_DIR" "$JUDGE_DIR" "$SELECT_DIR" "$MERGE_DIR"

# Load .env
if [ -f ../.env ]; then
    set -a; source ../.env; set +a
fi

# ── Helpers ──────────────────────────────────────────────────────────────────

header() {
    echo ""
    echo "======================================================"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================"
    echo ""
}

elapsed() {
    local secs=$1
    printf '%02dh %02dm %02ds' $((secs/3600)) $((secs%3600/60)) $((secs%60))
}

PIPELINE_START=$(date +%s)

# ── Print config ─────────────────────────────────────────────────────────────

header "BioASQ Phase B — Full Pipeline"

echo "Input data:       ${INPUT_DATA}"
echo "Dataset:          ${DATASET}"
echo "Models:           ${MODELS[*]}"
echo "Prompts:          ${PROMPT_IDS}"
echo "Context sources:  ${CONTEXT_SOURCES[*]}"
echo "Types:            ${TYPES}"
echo "Judge model:      ${JUDGE_MODEL}"
echo "Merge model:      ${MERGE_MODEL}"
echo "Merge top-Ks:     ${MERGE_TOP_KS[*]}  (5 submissions)"
echo ""

# =============================================================================
#  STEP 1: GENERATE — models × context sources × PROMPT_IDS ($PROMPT_IDS)
#  (yesno/factoid/list → exact_answer JSON; summary → ideal_answer paragraph JSON, exact empty)
# =============================================================================

header "STEP 1/4 — GENERATE"

STEP_START=$(date +%s)
RUN_IDX=0
TOTAL_RUNS=$(( ${#MODELS[@]} * ${#CONTEXT_SOURCES[@]} ))

for MODEL in "${MODELS[@]}"; do
    for CTX in "${CONTEXT_SOURCES[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))
        echo ""
        echo "--- Run ${RUN_IDX}/${TOTAL_RUNS}: ${MODEL} / ${CTX} ---"
        echo ""

        # uv run python inference/run_exact.py \
        #     --input "$INPUT_DATA" \
        #     --output-dir "$EXACT_DIR" \
        #     --model "$MODEL" \
        #     --backend openrouter \
        #     --prompt-ids ${PROMPT_IDS} \
        #     --context-source "$CTX" \
        #     --types ${TYPES} \
        #     --num-snippets ${NUM_CONTEXT} \
        #     --max-tokens ${GEN_MAX_TOKENS} \
        #     --temperature ${GEN_TEMPERATURE} \
        #     --request-delay ${GEN_REQUEST_DELAY}

        echo "--- Done: ${MODEL} / ${CTX} ---"
    done
done

STEP_END=$(date +%s)
echo ""
echo "Generation complete. $(elapsed $((STEP_END - STEP_START)))"
echo "Files in ${EXACT_DIR}/:"
ls -1 "$EXACT_DIR"/*.json 2>/dev/null | wc -l
echo ""

# =============================================================================
#  STEP 2: JUDGE — LLM drafts ideal_answer (prompts_generic #7) + self-scores each pair
# =============================================================================

header "STEP 2/4 — JUDGE (ideal answers)"

STEP_START=$(date +%s)

PRED_FILES=(${EXACT_DIR}/*.json)
echo "Judging ${#PRED_FILES[@]} prediction file(s)..."

# uv run python inference/judge_answers.py \
#     --inputs "${PRED_FILES[@]}" \
#     --input-data "$INPUT_DATA" \
#     --output "${JUDGE_DIR}/scores.json" \
#     --types ${TYPES} \
#     --model "$JUDGE_MODEL" \
#     --backend openrouter \
#     --context-source abstracts \
#     --num-context ${NUM_CONTEXT} \
#     --max-tokens ${JUDGE_MAX_TOKENS} \
#     --temperature ${JUDGE_TEMPERATURE} \
#     --request-delay ${JUDGE_REQUEST_DELAY} \
#     --tensor-parallel-size 2

STEP_END=$(date +%s)
echo ""
echo "Judging complete. $(elapsed $((STEP_END - STEP_START)))"

# =============================================================================
#  STEP 3: SELECT — pick best answer per question (no LLM, instant)
# =============================================================================

header "STEP 3/4 — SELECT BEST"

uv run python inference/select_best.py \
    --scores "${JUDGE_DIR}/scores.json" \
    --inputs "${PRED_FILES[@]}" \
    --output "${SELECT_DIR}/best.json" \
    --metric overall

# =============================================================================
#  STEP 4: MERGE — 5 submissions with different top-K diversity
# =============================================================================

header "STEP 4/4 — MERGE (5 submissions)"

STEP_START=$(date +%s)

for i in "${!MERGE_TOP_KS[@]}"; do
    K=${MERGE_TOP_KS[$i]}
    SUB_NUM=$((i + 1))
    OUT_FILE="${MERGE_DIR}/submission_${SUB_NUM}_topk${K}.json"

    echo ""
    echo "--- Submission ${SUB_NUM}/5: top-K=${K} ---"
    echo ""

    uv run python inference/merge_answers.py \
        --scores "${JUDGE_DIR}/scores.json" \
        --inputs "${PRED_FILES[@]}" \
        --input-data "$INPUT_DATA" \
        --output "$OUT_FILE" \
        --top-k ${K} \
        --types ${TYPES} \
        --model "$MERGE_MODEL" \
        --backend openrouter \
        --context-source abstracts \
        --num-context ${NUM_CONTEXT} \
        --max-tokens ${MERGE_MAX_TOKENS} \
        --temperature ${MERGE_TEMPERATURE} \
        --request-delay ${MERGE_REQUEST_DELAY}
done

STEP_END=$(date +%s)
echo ""
echo "Merging complete. $(elapsed $((STEP_END - STEP_START)))"

# =============================================================================
#  DONE
# =============================================================================

PIPELINE_END=$(date +%s)

header "PIPELINE COMPLETE"

echo "Total time: $(elapsed $((PIPELINE_END - PIPELINE_START)))"
echo ""
echo "5 Submissions:"
for i in "${!MERGE_TOP_KS[@]}"; do
    K=${MERGE_TOP_KS[$i]}
    SUB_NUM=$((i + 1))
    echo "  ${SUB_NUM}. ${MERGE_DIR}/submission_${SUB_NUM}_topk${K}.json  (top-${K} merge)"
done
echo ""
echo "Other outputs:"
echo "  Predictions:  ${EXACT_DIR}/ ($(ls -1 ${EXACT_DIR}/*.json 2>/dev/null | wc -l) files)"
echo "  Judge scores: ${JUDGE_DIR}/scores.json"
echo "  Selected:     ${SELECT_DIR}/best.json"
echo ""
echo "To add summary later without regenerating yesno/factoid/list:"
echo "  1) Run run_exact + judge + merge with --types summary only (reuse same EXACT_DIR layout;"
echo "     judge/merge still need --inputs to include all *.json that hold scores for summary)."
echo "  2) uv run python inference/overlay_merged_predictions.py \\"
echo "       --base   <existing submission_1_topk1.json> \\"
echo "       --overlay <summary-only merge out.json> \\"
echo "       --output  <combined.json>"
echo "  3) Run to_bioasq_submission.py on the combined file."
