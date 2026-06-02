#!/usr/bin/env bash
set -euo pipefail

python src/train.py --config configs/pretrain_llama_100m_rtx3060_60k.yaml
