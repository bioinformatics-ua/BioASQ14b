#!/bin/bash
# =============================================================================
#  Pipeline: Generate → Judge → Select / Merge
#
#  Three independent steps. Run any step standalone or the full pipeline.
#  Uncomment/comment sections as needed.
# =============================================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

# Dataset
INPUT_DATA="../data/BioASQ-task14bPhaseB-testset1_hydrated.jsonl"
DATASET="batch01"
CONTEXT_SOURCE="abstracts"

# Judge / Merge model
JUDGE_MODEL="anthropic/claude-sonnet-4-6"
MERGE_MODEL="anthropic/claude-sonnet-4-6"
BACKEND="openrouter"
REQUEST_DELAY=0.0      # 4.0 for free-tier rate limits

# Judge settings (ideal narrative + JSON scores)
JUDGE_MAX_TOKENS=2048
JUDGE_TEMPERATURE=0.0

# Merge settings
MERGE_MAX_TOKENS=1000
MERGE_TEMPERATURE=0.0
TOP_K=3                # top-K answers to show the merger

# Question types to process
TYPES="yesno factoid list"

# Paths (relative to phaseB-alex/)
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
EXACT_DIR="${REPO_DIR}/${DATASET}/outputs/exact"
JUDGE_DIR="${REPO_DIR}/${DATASET}/outputs/judged"
SELECT_DIR="${REPO_DIR}/${DATASET}/outputs/selected"
MERGE_DIR="${REPO_DIR}/${DATASET}/outputs/merged"

cd "$REPO_DIR"

# Load .env if present
if [ -f ../.env ]; then
    set -a; source ../.env; set +a
fi

# ── Helper ───────────────────────────────────────────────────────────────────

header() {
    echo ""
    echo "======================================================"
    echo "  $1"
    echo "======================================================"
    echo ""
}

# ── Step 1: Generate ─────────────────────────────────────────────────────────
# (Run this step separately using run_exact.sh with different models/prompts)
#
# Example: generate with 3 models × all prompts:
#
#   MODEL="anthropic/claude-sonnet-4-6" PROMPT_IDS="all" bash run_exact.sh
#   MODEL="google/gemini-2.0-flash-001" PROMPT_IDS="all" bash run_exact.sh
#   MODEL="meta-llama/llama-3.1-70b-instruct" PROMPT_IDS="all" bash run_exact.sh
#
# This produces files in ${EXACT_DIR}/ like:
#   anthropic-claude-sonnet-4-6_p1_abstracts_yesno.json
#   anthropic-claude-sonnet-4-6_p1_abstracts_factoid.json
#   ...

# ── Step 2: Judge ────────────────────────────────────────────────────────────

header "Step 2: Judge all prediction files"

PRED_FILES=(${EXACT_DIR}/*.json)
if [ ${#PRED_FILES[@]} -eq 0 ]; then
    echo "ERROR: No prediction files found in ${EXACT_DIR}/"
    echo "Run Step 1 (generate) first."
    exit 1
fi

echo "Found ${#PRED_FILES[@]} prediction file(s) to judge"

mkdir -p "${JUDGE_DIR}"

uv run python inference/judge_answers.py \
    --inputs "${PRED_FILES[@]}" \
    --input-data "${INPUT_DATA}" \
    --output "${JUDGE_DIR}/scores.json" \
    --types ${TYPES} \
    --model "${JUDGE_MODEL}" \
    --backend "${BACKEND}" \
    --context-source "${CONTEXT_SOURCE}" \
    --max-tokens ${JUDGE_MAX_TOKENS} \
    --temperature ${JUDGE_TEMPERATURE} \
    --request-delay ${REQUEST_DELAY}

# ── Step 3a: Select best (no LLM, instant) ───────────────────────────────────

header "Step 3a: Select best answer per question"

mkdir -p "${SELECT_DIR}"

uv run python inference/select_best.py \
    --scores "${JUDGE_DIR}/scores.json" \
    --inputs "${PRED_FILES[@]}" \
    --output "${SELECT_DIR}/best.json" \
    --metric overall

# ── Step 3b: Merge best via LLM ──────────────────────────────────────────────

header "Step 3b: Merge top answers via LLM"

mkdir -p "${MERGE_DIR}"

uv run python inference/merge_answers.py \
    --scores "${JUDGE_DIR}/scores.json" \
    --inputs "${PRED_FILES[@]}" \
    --input-data "${INPUT_DATA}" \
    --output "${MERGE_DIR}/final.json" \
    --top-k ${TOP_K} \
    --types ${TYPES} \
    --model "${MERGE_MODEL}" \
    --backend "${BACKEND}" \
    --context-source "${CONTEXT_SOURCE}" \
    --max-tokens ${MERGE_MAX_TOKENS} \
    --temperature ${MERGE_TEMPERATURE} \
    --request-delay ${REQUEST_DELAY}

# ── Done ─────────────────────────────────────────────────────────────────────

header "Pipeline complete"

echo "Outputs:"
echo "  Judge scores: ${JUDGE_DIR}/scores.json"
echo "  Selected:     ${SELECT_DIR}/best.json"
echo "  Merged:       ${MERGE_DIR}/final.json"
echo ""
echo "To evaluate (on dev set with gold answers):"
echo "  uv run python evaluation/evaluation_exact.py \\"
echo "    --predictions ${MERGE_DIR}/final.json \\"
echo "    --ground-truth ../data/val_data/13B1_golden_documents.jsonl"
