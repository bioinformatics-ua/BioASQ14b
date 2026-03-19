#!/bin/bash
# run_synthesize.sh — Synthesis sweep for Phase B
#
# USAGE:
#   bash phaseB/run_synthesize.sh              # runs in background, logs to phaseB/logs/synth{N}.out
#   LOGGING=1 bash phaseB/run_synthesize.sh    # run in foreground (no nohup)
#
# THREE CONFIGS:
#   1. Open-source run files  → open-source local model  (vLLM)
#   2. Open-source run files  → proprietary cloud model  (OpenRouter)
#   3. Proprietary run files  → proprietary cloud model  (OpenRouter)
#
# SKIP LOGIC: existing output files are skipped. Safe to re-run after partial failure.

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Auto-logging ──────────────────────────────────────────────────────────────
if [ -z "$LOGGING" ]; then
    mkdir -p "${REPO_DIR}/logs"
    N=1; while [ -f "${REPO_DIR}/logs/synth${N}.out" ]; do N=$((N+1)); done
    LOG="${REPO_DIR}/logs/synth${N}.out"
    echo "Logging to: $LOG"
    LOGGING=1 nohup bash "$0" "$@" > "$LOG" 2>&1 &
    echo "Started synthesis in background (PID $!). Follow with: tail -f $LOG"
    exit
fi

set +e

# ── Setup ─────────────────────────────────────────────────────────────────────
source "${REPO_DIR}/.venv/bin/activate" || true
if [ -f "${REPO_DIR}/.env" ]; then
    export $(grep -v '^#' "${REPO_DIR}/.env" | xargs)
fi

# ════════════════════════════════════════════════════════════════════════════
# SYNTHESIS CONFIGURATION — edit this block before running
# ════════════════════════════════════════════════════════════════════════════

SWEEP_DIR="${REPO_DIR}/dev/sweep_outputs/submission1-0.64_hydrated"

# ── Config 1: open-source runs → open-source local model ─────────────────────
OPENSOURCE_RUNS=(
    # "${SWEEP_DIR}/gemma-3-27b-it_abstracts_5_6.json"
    # "${SWEEP_DIR}/medgemma-27b-text-it_abstracts_5_6.json"
    # "${SWEEP_DIR}/medgemma-27b-text-it_abstracts_10_6.json"
    # "${SWEEP_DIR}/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_abstracts_10_6.json"
)
OPENSOURCE_MODEL="google/medgemma-27b-text-it"          # e.g. "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
OPENSOURCE_TP=2
OPENSOURCE_GPU_MEM=0.90
OPENSOURCE_MAX_MODEL_LEN=8192

# ── Config 2 & 3: proprietary cloud model ────────────────────────────────────
PROPRIETARY_MODEL="anthropic/claude-sonnet-4-6"   # OpenRouter model ID

# ── Config 2: open-source runs → proprietary model ───────────────────────────
# (reuses OPENSOURCE_RUNS above)

# ── Config 3: proprietary runs → proprietary model ───────────────────────────
PROPRIETARY_RUNS=(
    "${SWEEP_DIR}/claude-opus-4.6_abstracts_10_6.json"
    "${SWEEP_DIR}/gemini-2.0-flash-001_abstracts_10_6.json"
    "${SWEEP_DIR}/grok-4.1-fast_abstracts_10_6.json"
    "${SWEEP_DIR}/gemini-2.5-flash_abstracts_10_6.json"
	

)

# ── Shared synthesis settings ─────────────────────────────────────────────────
DATA_PATH="/home/ucloud/BioASQ13B/phaseA-reranker/runs_bioasq_format_hydrated/submission1-0.64_hydrated.jsonl"
PROMPTS_FILE="${REPO_DIR}/synthesis/prompts.json"
PROMPT_IDS="2"               # comma-separated, e.g. "1,2,3"
MAX_TOKENS=1000
TEMPERATURE=0.5

OUTPUT_DIR="${REPO_DIR}/dev/sweep_synthesis"
mkdir -p "$OUTPUT_DIR"

# ════════════════════════════════════════════════════════════════════════════
# HELPER
# ════════════════════════════════════════════════════════════════════════════

run_synthesis() {
    local OUT_ID="$1"
    local BACKEND="$2"
    local MODEL="$3"
    local TP="${4:-1}"
    local GPU_MEM="${5:-0.90}"
    local MAX_LEN="${6:-8192}"
    shift 6
    local RUNS=("$@")

    if [ ${#RUNS[@]} -eq 0 ]; then
        echo "[SKIP] ${OUT_ID}: no run files configured"
        return 0
    fi

    # Skip check: look for any existing output file with this out-id
    local MODEL_NAME
    MODEL_NAME=$(basename "$MODEL")
    local N_RUNS=${#RUNS[@]}
    local EXISTING
    EXISTING=$(ls "${OUTPUT_DIR}/${OUT_ID}_${MODEL_NAME}_${N_RUNS}_"*.json 2>/dev/null | head -1)
    if [ -n "$EXISTING" ] && [ -n "$(echo "$PROMPT_IDS" | tr ',' '\n' | while read pid; do
        [ -f "${OUTPUT_DIR}/${OUT_ID}_${MODEL_NAME}_${N_RUNS}_${pid}.json" ] && echo "ok"
    done)" ]; then
        echo "[SKIP] ${OUT_ID} — outputs already exist"
        return 0
    fi

    echo "════════════════════════════════════════"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Config    : ${OUT_ID}"
    echo "  Backend   : ${BACKEND}"
    echo "  Model     : ${MODEL}"
    echo "  Runs      : ${N_RUNS} file(s)"
    echo "  Prompt IDs: ${PROMPT_IDS}"
    for r in "${RUNS[@]}"; do echo "    - $(basename "$r")"; done
    echo "════════════════════════════════════════"

    uv run python "${REPO_DIR}/synthesis/synthesize.py" \
        "${RUNS[@]}" \
        --data-path    "$DATA_PATH" \
        --output-dir   "$OUTPUT_DIR" \
        --out-id       "$OUT_ID" \
        --backend      "$BACKEND" \
        --model        "$MODEL" \
        --prompts-file "$PROMPTS_FILE" \
        --prompt-ids   "$PROMPT_IDS" \
        --max-tokens   "$MAX_TOKENS" \
        --temperature  "$TEMPERATURE" \
        --tensor-parallel-size        "$TP" \
        --gpu-memory-utilization      "$GPU_MEM" \
        --max-model-len               "$MAX_LEN"

    local RC=$?
    [ $RC -ne 0 ] && echo "[ERROR] Exit code $RC — ${OUT_ID}" || echo "[OK] Done — ${OUT_ID}"
    return $RC
}

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

TOTAL=0
FAILED=0

echo "════════════════════════════════════════"
echo "  SYNTHESIS STARTED: $(date)"
echo "  Output dir: $OUTPUT_DIR"
echo "════════════════════════════════════════"

# ── Config 1: open-source runs → open-source local model ─────────────────────
run_synthesis \
    "opensource_to_opensource" \
    "local" \
    "$OPENSOURCE_MODEL" \
    "$OPENSOURCE_TP" "$OPENSOURCE_GPU_MEM" "$OPENSOURCE_MAX_MODEL_LEN" \
    "${OPENSOURCE_RUNS[@]}" \
    || FAILED=$((FAILED+1))
TOTAL=$((TOTAL+1))

# ── Config 2: open-source runs → proprietary model ───────────────────────────
run_synthesis \
    "opensource_to_proprietary" \
    "openrouter" \
    "$PROPRIETARY_MODEL" \
    1 0.90 8192 \
    "${OPENSOURCE_RUNS[@]}" \
    || FAILED=$((FAILED+1))
TOTAL=$((TOTAL+1))

# ── Config 3: proprietary runs → proprietary model ───────────────────────────
run_synthesis \
    "proprietary_to_proprietary" \
    "openrouter" \
    "$PROPRIETARY_MODEL" \
    1 0.90 8192 \
    "${PROPRIETARY_RUNS[@]}" \
    || FAILED=$((FAILED+1))
TOTAL=$((TOTAL+1))

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════"
echo "  SYNTHESIS COMPLETE"
echo "  End    : $(date)"
echo "  Total  : $TOTAL configs"
echo "  Failed : $FAILED configs"
echo "  Outputs: $OUTPUT_DIR"
echo "════════════════════════════════════════"
