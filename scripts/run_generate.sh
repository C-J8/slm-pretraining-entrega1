#!/usr/bin/env bash
set -euo pipefail

python src/generate.py \
  --checkpoint checkpoints/ckpt_best.pt \
  --prompt "The future of artificial intelligence is" \
  --max_new_tokens 100 \
  --temperature 0.8 \
  --top_k 50 \
  --output outputs/samples_best_60k.txt
