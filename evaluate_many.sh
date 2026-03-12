#!/usr/bin/env bash

set -euo pipefail

DATA_ROOT="/opt/shared/datasets/NeoRadiumTDLdataset/test"

# Iterate over experiment folders exp1 to exp5
for exp in {1..5}; do
  # Iterate over all TDL test sets: TDLA to TDLE
  for letter in A B C D E; do
    lower_letter=${letter,,}
    data_path="${DATA_ROOT}/TDL${letter}/"

    # (pilot_symbols, checkpoint suffix) pairs
    # 2         -> _2
    # "2 3"     -> _23
    # "2 7 11"  -> _2711
    for combo in "2|2" "2 3|23" "2 7 11|2711"; do
      pilot_symbols="${combo%%|*}"
      suffix="${combo##*|}"
      checkpoint="runs/exp${exp}/tdl${lower_letter}_${suffix}/best.pt"

      echo "Running eval: exp${exp}, TDL${letter}, pilot_symbols=${pilot_symbols}, ckpt_suffix=${suffix}"
      python3 evaluate.py \
        --data_path "${data_path}" \
        --checkpoint "${checkpoint}" \
        --pilot_symbols ${pilot_symbols}
    done
  done
done