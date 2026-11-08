#!/usr/bin/env bash
# Data curve (decision 17): same config as run1_5ep, only the train data changes.
# iters = 5 epochs of each subset with batch 4, as decision 15: ceil(5 * examples / 4).
#   session_25: 1351 examples -> 1689 iters
#   session_50: 2708 examples -> 3385 iters
# Run from the repo root. Sequential: 25 first.
set -euo pipefail
cd "$(dirname "$0")/.."

train() {  # $1 = subset percent, $2 = iters
  uv run mlx_lm.lora \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --train \
    --data "data/dataset/session_$1" \
    --mask-prompt \
    --adapter-path "adapters/data$1" \
    --iters "$2" \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --num-layers 16 \
    --max-seq-length 2048 \
    --save-every 100 \
    --steps-per-eval 100 \
    --val-batches 25 \
    --seed 0 \
    2>&1 | tee "logs/adapters_data$1.log"
}

train 25 1689
train 50 3385
