#!/bin/bash


ROOT_DIR="$(pwd)"
DATA="${ROOT_DIR}/data/batch03/phaseb/BioASQ-task14bPhaseB-testset3.hydrated.jsonl"
OUTPUT_DIR="${ROOT_DIR}/data/batch03/phaseb/generation/quorum"
GEMMA4B_IP="http://127.0.0.1:8002/v1"
GEMMA2B_IP="http://127.0.0.1:8003/v1"
QWEN_IP="http://127.0.0.1:8004/v1"
QWEN3_6_IP="http://127.0.0.1:8001/v1"
NEMOTRON_IP="http://127.0.0.1:8005/v1"

cd src/bioasq/phase_b/quorum

export OPENROUTER_API_KEY=sk-or-v1-XXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Hydration
# uv run src/bioasq/data/hydration.py "data/batch03/phaseb/BioASQ-task14bPhaseB-testset3.json"

# Submissions
OUTPUT_DIR="${ROOT_DIR}/data/batch03/phaseb/generation/quorum"
for v in {0..4}; do
  uv run src/bioasq/common/merge_ab.py main \
    "${DATA}" \
    "${OUTPUT_DIR}/v${v}.final.json" \
    -o "${OUTPUT_DIR}/v${v}.bioasq.json"
done

uv run run.py run \
  -d "$DATA" \
  -o "$OUTPUT_DIR/v0.jsonl" \
  -m "external|${GEMMA4B_IP}|google/gemma-4-E4B-it" \
  -m "external|${GEMMA2B_IP}|google/gemma-4-E2B-it" \
  -m "external|${NEMOTRON_IP}|nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16" \
  --synthesizer-model "external|${GEMMA4B_IP}|google/gemma-4-E4B-it" 2>&1 > "$OUTPUT_DIR/v0.log" &

uv run run.py run \
  -d "$DATA" \
  -o "$OUTPUT_DIR/v1.jsonl" \
  -m "openrouter|google/gemma-4-26b-a4b-it" \
  -m "openrouter|google/gemma-4-31b-it" \
  -m "openrouter|nvidia/nemotron-3-nano-30b-a3b" \
  -m "openrouter|qwen/qwen3.5-flash-02-23" \
  --synthesizer-model "openrouter|google/gemma-4-26b-a4b-it" 2>&1 > "$OUTPUT_DIR/v1.log" &

uv run run.py run \
  -d "$DATA" \
  -o "$OUTPUT_DIR/v2.jsonl" \
  -m "openrouter|nvidia/nemotron-3-super-120b-a12b:nitro" \
  -m "openrouter|x-ai/grok-4.1-fast" \
  -m "openrouter|google/gemini-3-flash-preview" \
  -m "openrouter|google/gemini-2.5-flash" \
  -m "openrouter|openai/gpt-5-mini" \
  --synthesizer-model "openrouter|google/gemini-2.5-flash" 2>&1 > "$OUTPUT_DIR/v2.log" &

uv run run.py run \
  -d "$DATA" \
  -o "$OUTPUT_DIR/v3.jsonl" \
  -m "openrouter|google/gemma-4-31b-it" \
  -m "openrouter|qwen/qwen3.5-flash-02-23" \
  -m "openrouter|mistralai/mistral-small-2603" \
  -m "openrouter|mistralai/mistral-large-2512" \
  --synthesizer-model "openrouter|mistralai/mistral-small-2603" 2>&1 > "$OUTPUT_DIR/v3.log" &

uv run run.py run \
  -d "$DATA" \
  -o "$OUTPUT_DIR/v4_0_20.jsonl" \
  -m "openrouter|google/gemma-4-26b-a4b-it" \
  -m "external|${QWEN3_6_IP}|Qwen/Qwen3.6-35B-A3B" \
  --start-index 0 --end-index 20 \
  --synthesizer-model "external|${QWEN3_6_IP}|Qwen/Qwen3.6-35B-A3B" 2>&1 > "$OUTPUT_DIR/v4_0_20.log" &

uv run run.py run \
  -d "$DATA" \
  -o "$OUTPUT_DIR/v4_20_40.jsonl" \
  -m "openrouter|google/gemma-4-26b-a4b-it" \
  -m "external|${QWEN3_6_IP}|Qwen/Qwen3.6-35B-A3B" \
  --start-index 20 --end-index 40 \
  --synthesizer-model "external|${QWEN3_6_IP}|Qwen/Qwen3.6-35B-A3B" 2>&1 > "$OUTPUT_DIR/v4_20_40.log" &

uv run run.py run \
  -d "$DATA" \
  -o "$OUTPUT_DIR/v4_40_60.jsonl" \
  -m "openrouter|google/gemma-4-26b-a4b-it" \
  -m "external|${QWEN3_6_IP}|Qwen/Qwen3.6-35B-A3B" \
  --start-index 40 --end-index 60 \
  --synthesizer-model "external|${QWEN3_6_IP}|Qwen/Qwen3.6-35B-A3B" 2>&1 > "$OUTPUT_DIR/v4_40_60.log" &
