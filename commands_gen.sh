cd phaseB

declare -A models=(
  ["qwen/qwen3.5-27b"]="local"
  ["nvidia/nemotron-3-super-120b-a12b"]="openrouter"
  ["mistralai/mistral-small-2603"]="openrouter"
  ["anthropic/claude-sonnet-4-6"]="openrouter"
  ["anthropic/claude-opus-4.6"]="openrouter"
)

for submission in ../phaseA-reranker/runs_bioasq_format_hydrated/submission*.json; do
  for model in "${!models[@]}"; do
    echo "Running inference for ${model} with backend ${models[$model]}"
    uv run inference/run.py \
      --data-path   ${submission} \
      --output-dir  outputs/ \
      --model       ${model} \
      --backend     ${models[$model]} \
      --input-type  abstracts \
      --num-support 3,5 \
      --prompt-ids  1,2,3,4,5,6,7
  done
done
