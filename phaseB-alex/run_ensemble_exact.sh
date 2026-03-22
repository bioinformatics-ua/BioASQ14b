#!/bin/bash

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-detach under nohup if not already logging
if [ -z "$LOGGING" ]; then
    mkdir -p "${REPO_DIR}/logs"
    RUN_NUM=1
    while [ -f "${REPO_DIR}/logs/run_ensemble${RUN_NUM}.out" ]; do
        RUN_NUM=$((RUN_NUM + 1))
    done
    LOG="${REPO_DIR}/logs/run_ensemble${RUN_NUM}.out"
    echo "Logging to: $LOG"
    LOGGING=1 nohup bash "$0" "$@" > "$LOG" 2>&1 &
    echo "Started in background (PID $!). Follow with: tail -f $LOG"
    exit
fi

# ============================================================
# CONFIGURABLE VARIABLES
# ============================================================

QTYPE="yesno"       # yesno / factoid / list
DATASET="dev"       # dev / batch01 / batch01-phaseB

# Output filename — auto-set to ensemble_<QTYPE>, saved to dev/outputs/ensemble_exact/
OUTPUT_NAME="ensemble_${QTYPE}"

# ------------------------------------------------------------
# INPUT FILES
# List the filenames (without .json) you want to ensemble.
# These must exist in dev/outputs/exact/ (or batch01/outputs/exact/).
# Just the filename — no path, no extension.
# ------------------------------------------------------------
INPUT_FILES=(
    "deepseek-deepseek-v3-2_p4_snippets_yesno.json"                 # 0.9328
    "Qwen-Qwen3-5-27B_p4_snippets_yesno.json"                       # 0.9328
    "openai-gpt-oss-20b_p2_snippets_yesno.json"
    # "stepfun-step-3-5-flash:free_p4_snippets_yesno.json"            # 0.8712
    # "openai-gpt-oss-120b_p4_snippets_yesno.json"                    # 0.8712
    # "qwen-qwen3-32b_p4_snippets_yesno.json"                         # 0.8583
    # "xiaomi-mimo-v2-flash_p4_snippets_yesno.json"                   # 0.8132
    # "arcee-ai-trinity-large-preview:free_p4_snippets_yesno.json"    # 0.8132
    # "nvidia-nemotron-3-super-120b-a12b:free_p4_snippets_yesno.json" # 0.7984
    # "moonshotai-kimi-k2-5_p4_snippets_yesno.json"                   # 0.7984
    # "minimax-minimax-m2-5_p4_snippets_yesno.json"                   # 0.7984
)

# ============================================================
# ENSEMBLE STRATEGY SETTINGS
# ============================================================

THRESHOLD=0.5   # used for list — see explanation below
RRF_K=60        # used for factoid — see explanation below

# QTYPE="yesno"
#   Strategy: MAJORITY VOTE
#   Each input file votes "yes" or "no". The answer with more
#   votes wins. Ties go to "yes".
#   → No tunable parameters for yes/no.
#
# QTYPE="factoid"
#   Strategy: RECIPROCAL RANK FUSION (RRF)
#   Each input file provides a ranked list of candidate answers
#   (best guess first). RRF scores each candidate by summing
#   1/(rank + RRF_K) across all input files, then re-ranks.
#   Candidates that rank highly in multiple files score best.
#
#   RRF_K — smoothing constant that controls how much rank matters:
#     Range : 1 to 100+ (typical: 60)
#     Low  (e.g. 10) → rank matters a lot — rank-1 answers
#                       get a big boost over rank-2
#     High (e.g. 100) → rank matters less — an answer appearing
#                        at rank-5 in many files can beat one
#                        at rank-1 in only a few files
#     Default 60 is a well-established empirical value from
#     information retrieval research.
#
# QTYPE="list"
#   Strategy: FREQUENCY THRESHOLD
#   Each input file provides a set of entities as the answer.
#   An entity is included in the final answer only if it appears
#   in >= THRESHOLD fraction of the input files.
#
#   THRESHOLD — fraction of input files an entity must appear in:
#     Range : 0.0 to 1.0
#     0.0  → include everything any file mentions (max recall,
#             but many wrong entities — low precision)
#     0.5  → majority vote — entity must appear in >half the files
#             (balanced precision/recall)
#     1.0  → only include entities all files agree on (max
#             precision, but misses many correct entities)
#     Recommended starting point: 0.5
#     If your precision is too low: increase toward 0.7–0.8
#     If your recall is too low: decrease toward 0.3–0.4
#
# ============================================================

# ============================================================
# PATHS — do not edit
# ============================================================

if [ "$DATASET" = "dev" ]; then
    GROUND_TRUTH="${REPO_DIR}/../data/val_data/13B1_golden_documents.jsonl"
elif [ "$DATASET" = "batch01" ]; then
    GROUND_TRUTH="/home/ucloud/BioASQ13B/phaseA-reranker/runs_bioasq_format_hydrated/submission1-0.64_hydrated.jsonl"
elif [ "$DATASET" = "batch01-phaseB" ]; then
    GROUND_TRUTH="/home/ucloud/BioASQ13B/data/BioASQ-task14bPhaseB-testset1_hydrated.jsonl"
fi

INFERENCE_DIR="${REPO_DIR}/${DATASET}/outputs/exact"       # where run_exact.sh / run_local_grid.sh saves outputs
OUTPUT_DIR="${REPO_DIR}/${DATASET}/outputs/ensemble_exact"  # where ensemble output is saved
RESULTS_DIR="${REPO_DIR}/${DATASET}/results/ensemble_exact"
OUTPUT="${OUTPUT_DIR}/${OUTPUT_NAME}.json"
REPORT="${RESULTS_DIR}/${OUTPUT_NAME}.json"

mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR"

# Build full paths from filenames
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

# ============================================================
# STEP 1 — Ensemble
# ============================================================

echo "========================================"
echo "  Type      : ${QTYPE}"
echo "  Dataset   : ${DATASET}"
echo "  Output    : ${OUTPUT}"
echo "  Inputs    :"
for f in "${RESOLVED_INPUTS[@]}"; do echo "    $f"; done
echo "========================================"

uv run python "${REPO_DIR}/inference/ensemble_exact.py" \
    --inputs    "${RESOLVED_INPUTS[@]}" \
    --output    "$OUTPUT" \
    --qtype     "$QTYPE" \
    --threshold "$THRESHOLD" \
    --rrf-k     "$RRF_K"

echo "[1/2] Ensemble done."

# ============================================================
# STEP 2 — Evaluation (dev only)
# ============================================================

if [ "$DATASET" = "dev" ]; then
    uv run python "${REPO_DIR}/evaluation/evaluation_exact.py" \
        --predictions  "$OUTPUT" \
        --ground-truth "$GROUND_TRUTH" \
        --output       "$REPORT" \
        --types        "$QTYPE"
    echo "[2/2] Evaluation done. Report: ${REPORT}"
else
    echo "[2/2] Skipping evaluation (batch01 — no gold answers)."
fi

echo "========================================"