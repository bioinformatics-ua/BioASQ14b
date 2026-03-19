#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
CONVERTER="${REPO_DIR}/bioasq_format_converter.py"

TESTSET="/home/ucloud/BioASQ13B/phaseA-BM25/Batch01/BioASQ-task14bPhaseA-testset1"

SWEEP_OUT="/home/ucloud/BioASQ13B/phaseB/dev/sweep_outputs/submission1-0.64_hydrated"
SWEEP_SYNTH="/home/ucloud/BioASQ13B/phaseB/dev/sweep_synthesis"
SUBMISSIONS="/home/ucloud/BioASQ13B/phaseB-alex/batch01/submission"

OUT_DIR="${REPO_DIR}/batch01/submission"
mkdir -p "$OUT_DIR"

source "${REPO_DIR}/.venv/bin/activate"

# ── system0 ───────────────────────────────────────────────────────────────────
# ideal : medgemma sweep_outputs 5_6
# fallback: same
# exact  : Submission3 (openai-gpt-oss-120b)
uv run python "$CONVERTER" \
    "$TESTSET" \
    "${SWEEP_OUT}/medgemma-27b-text-it_abstracts_5_6.json" \
    "${OUT_DIR}/system0.json" \
    --fallback-ideal-answer "${SWEEP_OUT}/medgemma-27b-text-it_abstracts_5_6.json" \
    --exact-answer "${SUBMISSIONS}/Submission3_openai-gpt-oss-120b.json"

echo "[OK] system0.json"

# ── system1 ───────────────────────────────────────────────────────────────────
# ideal : sweep_synth opensource_to_opensource medgemma 4_2
# fallback: medgemma sweep_outputs 5_6 (system0 ideal)
# exact  : Submission1 (gemini-2.5-flash + qwen3-max-thinking)
uv run python "$CONVERTER" \
    "$TESTSET" \
    "${SWEEP_SYNTH}/opensource_to_opensource_medgemma-27b-text-it_4_2.json" \
    "${OUT_DIR}/system1.json" \
    --fallback-ideal-answer "${SWEEP_OUT}/medgemma-27b-text-it_abstracts_5_6.json" \
    --exact-answer "${SUBMISSIONS}/Submission1_gemini_2-5_flash_qwen3_max_thinking.json"

echo "[OK] system1.json"

# ── system2 ───────────────────────────────────────────────────────────────────
# ideal : claude-opus-4.6 sweep_outputs 10_6
# fallback: same
# exact  : Submission2 (gemini-2.5-flash)
uv run python "$CONVERTER" \
    "$TESTSET" \
    "${SWEEP_OUT}/claude-opus-4.6_abstracts_10_6.json" \
    "${OUT_DIR}/system2.json" \
    --fallback-ideal-answer "${SWEEP_OUT}/claude-opus-4.6_abstracts_10_6.json" \
    --exact-answer "${SUBMISSIONS}/Submission2_gemini_2-5_flash.json"

echo "[OK] system2.json"

# ── system3 ───────────────────────────────────────────────────────────────────
# ideal : sweep_synth proprietary_to_proprietary claude-sonnet-4-6 4_2
# fallback: claude-opus-4.6 (same as system2)
# exact  : Submission4 (gemini-2.5-flash + qwen3-max-thinking v2)
uv run python "$CONVERTER" \
    "$TESTSET" \
    "${SWEEP_SYNTH}/proprietary_to_proprietary_claude-sonnet-4-6_4_2.json" \
    "${OUT_DIR}/system3.json" \
    --fallback-ideal-answer "${SWEEP_OUT}/claude-opus-4.6_abstracts_10_6.json" \
    --exact-answer "${SUBMISSIONS}/Submission4_gemini_2-5_flash_qwen3_max_thinking_(2).json"

echo "[OK] system3.json"

# ── system4 ───────────────────────────────────────────────────────────────────
# ideal : sweep_synth opensource_to_proprietary claude-opus-4-6 4_2
# fallback: claude-opus-4.6 (same as system2)
# exact  : Submission5 (gemini-2.5-flash v2)
uv run python "$CONVERTER" \
    "$TESTSET" \
    "${SWEEP_SYNTH}/opensource_to_proprietary_claude-opus-4-6_4_2.json" \
    "${OUT_DIR}/system4.json" \
    --fallback-ideal-answer "${SWEEP_OUT}/claude-opus-4.6_abstracts_10_6.json" \
    --exact-answer "${SUBMISSIONS}/Submission5_gemini_2-5_flash_(2).json"

echo "[OK] system4.json"

echo ""
echo "All done. Submissions in: ${OUT_DIR}"
