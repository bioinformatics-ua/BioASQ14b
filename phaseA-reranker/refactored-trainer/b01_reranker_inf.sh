#!/usr/bin/env bash
# Inference on the biggest checkpoint of every run in outputs/,
# with load balancing across GPU 0 and GPU 1.
# Output: batch01/runs/{run_name}-{checkpoint_name}.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/outputs"
QUESTIONS="$SCRIPT_DIR/batch01/b01.jsonl"
RUNS_DIR="$SCRIPT_DIR/batch01/runs"

mkdir -p "$RUNS_DIR"

# ── Collect (run_dir → biggest checkpoint) ────────────────────────────────────
declare -A biggest_num   # run_dir → highest step int
declare -A biggest_path  # run_dir → full checkpoint path

while IFS= read -r ckpt; do
    run_dir="$(dirname "$ckpt")"
    step="${ckpt##*checkpoint-}"
    prev="${biggest_num[$run_dir]:-0}"
    if (( step > prev )); then
        biggest_num["$run_dir"]="$step"
        biggest_path["$run_dir"]="$ckpt"
    fi
done < <(find "$MODELS_DIR" -type d -name "checkpoint-*" | sort)

# ── Run inference with true 2-GPU load balancing ──────────────────────────────
# As soon as any GPU finishes, the next job goes straight to that GPU.
# Requires bash 5.1+ for: wait -n -p finished_pid
declare -A pid_to_gpu   # pid → gpu index
free_gpus=(0 1)         # initially both GPUs are free

_submit() {
    local ckpt_path="$1" out_file="$2" gpu="$3"
    local run_name ckpt_name
    run_name="$(basename "$(dirname "$ckpt_path")")"
    ckpt_name="$(basename "$ckpt_path")"
    echo "[GPU $gpu] $run_name / $ckpt_name"
    CUDA_VISIBLE_DEVICES=$gpu uv run main.py inference \
        --model-name  "$ckpt_path" \
        --questions-path "$QUESTIONS" \
        --output-path "$out_file" \
        &
    pid_to_gpu[$!]=$gpu
}

for run_dir in "${!biggest_path[@]}"; do
    ckpt_path="${biggest_path[$run_dir]}"
    run_name="$(basename "$run_dir")"
    ckpt_name="$(basename "$ckpt_path")"
    out_file="$RUNS_DIR/${run_name}-${ckpt_name}.json"

    if [[ -f "$out_file" ]]; then
        echo "[skip] $out_file already exists"
        continue
    fi

    # If no GPU is free, wait for the next job to finish and reclaim its GPU
    if (( ${#free_gpus[@]} == 0 )); then
        finished_pid=""
        wait -n -p finished_pid
        freed_gpu="${pid_to_gpu[$finished_pid]}"
        unset "pid_to_gpu[$finished_pid]"
        free_gpus+=("$freed_gpu")
        echo "[GPU $freed_gpu freed] (pid $finished_pid done)"
    fi

    # Take the first free GPU and submit
    gpu="${free_gpus[0]}"
    free_gpus=("${free_gpus[@]:1}")
    _submit "$ckpt_path" "$out_file" "$gpu"
done

# Drain remaining jobs
echo "Waiting for remaining jobs..."
wait

echo "All done. Results in $RUNS_DIR"
