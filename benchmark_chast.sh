#!/usr/bin/env bash

set -euo pipefail

# Defaults (override with env vars if needed)
DATA_PATH="${DATA_PATH:-/home/berkay/Desktop/research/datasets/NeoRadiumTDLdataset/test/TDLA}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SNR="${SNR:-20}"
WARMUP_BATCHES="${WARMUP_BATCHES:-5}"
DEVICE="${DEVICE:-cuda:0}"

echo "Running benchmark on DATA_PATH=${DATA_PATH}"

cmd=(
  python3 benchmark_chast.py
  --data_path "${DATA_PATH}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --snr "${SNR}"
  --warmup_batches "${WARMUP_BATCHES}"
  --device "${DEVICE}"
)

"${cmd[@]}"
