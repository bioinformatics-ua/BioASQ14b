#!/bin/bash
# ============================================================================
# Snippet Extraction Pipeline
# ============================================================================
#
# End-to-end pipeline for training and running the snippet extraction model.
#
# Steps:
#   1. Prepare training data (join gold snippets with document texts)
#   2. Generate rationales via large model API (optional, costs ~$5)
#   3. Format for LoRA training (chat template)
#   4. Train QLoRA adapter
#   5. Extract snippets on new data
#   6. Evaluate snippet quality
#
# Usage:
#   ./scripts/run_snippets.sh prepare     # Steps 1-3
#   ./scripts/run_snippets.sh train       # Step 4
#   ./scripts/run_snippets.sh extract     # Step 5
#   ./scripts/run_snippets.sh evaluate    # Step 6
#   ./scripts/run_snippets.sh all         # Steps 1-6
# ============================================================================

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# ----------------------------------------
# CONFIGURABLE VARIABLES
# ----------------------------------------

# Data paths
TRAINING_JSON="data/training14b/training14b.json"
INFLATED_JSONL="data/quality/training14b_inflated_clean_wContents.jsonl"
SNIPPET_DATA_DIR="data/training/snippet_extraction"

# Rationale generation
RATIONALE_BACKEND="local"         # local (vLLM) or openrouter
RATIONALE_MODEL="Qwen/Qwen3.5-27B"
RATIONALE_TOKENIZER="Qwen/Qwen3.5-27B"        # needed for GGUF
RATIONALE_HF_CONFIG="Qwen/Qwen3.5-27B"        # needed for GGUF
TENSOR_PARALLEL_SIZE=2            # GPUs for vLLM
RATIONALE_BATCH_SIZE=64           # vLLM batch size
RATIONALE_DELAY=0.00001          # seconds between API calls (openrouter only)

# LoRA training (Unsloth — default)
BASE_MODEL="unsloth/gemma-4-31B"
CHAT_TEMPLATE="gemma-4-thinking"  # "gemma-4" for standard
LORA_R=16
LORA_ALPHA=32
EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=8
LR=2e-4
MAX_SEQ_LEN=2048

# Inference
ADAPTER_PATH="${SNIPPET_DATA_DIR}/lora_output/final_adapter"
EXTRACT_INPUT="data/val_data/13B1_golden_documents.jsonl"
EXTRACT_OUTPUT="data/snippets/extracted_snippets.jsonl"
EXTRACT_BACKEND="local"  # local or openrouter

# ----------------------------------------
# FUNCTIONS
# ----------------------------------------

prepare() {
    echo "=== Step 1: Prepare training data ==="
    uv run python -m bioasq.snippets.prepare_training_data \
        --training-json "$TRAINING_JSON" \
        --inflated-jsonl "$INFLATED_JSONL" \
        --output "${SNIPPET_DATA_DIR}/gold_pairs.jsonl"

    echo ""
    echo "=== Step 2: Generate rationales ==="
    uv run python -m bioasq.snippets.generate_rationales \
        --input "${SNIPPET_DATA_DIR}/gold_pairs.jsonl" \
        --output "${SNIPPET_DATA_DIR}/gold_pairs_with_rationale.jsonl" \
        --backend "$RATIONALE_BACKEND" \
        --model "$RATIONALE_MODEL" \
        --tokenizer "$RATIONALE_TOKENIZER" \
        --hf-config-path "$RATIONALE_HF_CONFIG" \
        --language-model-only \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --batch-size "$RATIONALE_BATCH_SIZE" \
        --resume

    echo ""
    echo "=== Step 3: Format for training ==="
    uv run python -m bioasq.snippets.format_for_training \
        --input "${SNIPPET_DATA_DIR}/gold_pairs_with_rationale.jsonl" \
        --output-dir "${SNIPPET_DATA_DIR}/" \
        --val-fraction 0.1
}

train() {
    echo "=== Step 4: Train LoRA (Unsloth) ==="
    uv run python -m bioasq.snippets.train_unsloth \
        --base-model "$BASE_MODEL" \
        --train-data "${SNIPPET_DATA_DIR}/chat_train.jsonl" \
        --val-data "${SNIPPET_DATA_DIR}/chat_val.jsonl" \
        --output-dir "${SNIPPET_DATA_DIR}/lora_output" \
        --chat-template "$CHAT_TEMPLATE" \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --gradient-accumulation "$GRAD_ACCUM" \
        --lr "$LR" \
        --max-seq-length "$MAX_SEQ_LEN"
}

train_legacy() {
    echo "=== Step 4 (legacy): Train QLoRA (transformers+peft) ==="
    uv run python -m bioasq.snippets.train_lora \
        --base-model "google/gemma-4-31B" \
        --train-data "${SNIPPET_DATA_DIR}/chat_train.jsonl" \
        --val-data "${SNIPPET_DATA_DIR}/chat_val.jsonl" \
        --output-dir "${SNIPPET_DATA_DIR}/lora_output" \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --gradient-accumulation "$GRAD_ACCUM" \
        --lr "$LR" \
        --max-seq-length "$MAX_SEQ_LEN"
}

extract() {
    echo "=== Step 5: Extract snippets ==="
    uv run python -m bioasq.snippets.extract \
        --input "$EXTRACT_INPUT" \
        --output "$EXTRACT_OUTPUT" \
        --base-model "$BASE_MODEL" \
        --adapter-path "$ADAPTER_PATH" \
        --backend "$EXTRACT_BACKEND" \
        --max-new-tokens 500 \
        --temperature 0.1
}

evaluate() {
    echo "=== Step 6: Evaluate ==="
    uv run python -m bioasq.snippets.evaluate \
        --predictions "$EXTRACT_OUTPUT" \
        --gold "$TRAINING_JSON" \
        --inflated "$INFLATED_JSONL"
}

# ----------------------------------------
# MAIN
# ----------------------------------------

case "${1:-all}" in
    prepare)       prepare ;;
    train)         train ;;
    train-legacy)  train_legacy ;;
    extract)       extract ;;
    evaluate)      evaluate ;;
    all)
        prepare
        train
        extract
        evaluate
        ;;
    *)
        echo "Usage: $0 {prepare|train|train-legacy|extract|evaluate|all}"
        exit 1
        ;;
esac

echo ""
echo "Done."
