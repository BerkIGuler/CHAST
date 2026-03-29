#!/usr/bin/env bash
# Run all continuation configs with the same out_dir layout as train_many.sh so
# `best.pt`, `load_checkpoint`, and TensorBoard (`out_dir/tb/`) stay aligned.
#
# Use the same DEVICE and EXP as the original train_many run (e.g. exp1 -> runs/exp1/<run_name>/).
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"

CONFIGS=(
  "configs/cont_train_chast_tdla_2.yaml"
  "configs/cont_train_chast_tdlb_2.yaml"
  "configs/cont_train_chast_tdlc_2.yaml"
  "configs/cont_train_chast_tdld_2.yaml"
  "configs/cont_train_chast_tdle_2.yaml"

  "configs/cont_train_chast_tdla_23.yaml"
  "configs/cont_train_chast_tdlb_23.yaml"
  "configs/cont_train_chast_tdlc_23.yaml"
  "configs/cont_train_chast_tdld_23.yaml"
  "configs/cont_train_chast_tdle_23.yaml"

  "configs/cont_train_chast_tdla_2711.yaml"
  "configs/cont_train_chast_tdlb_2711.yaml"
  "configs/cont_train_chast_tdlc_2711.yaml"
  "configs/cont_train_chast_tdld_2711.yaml"
  "configs/cont_train_chast_tdle_2711.yaml"
)

# Same experiment / seed convention as train_many.sh (maps expN -> seed N).
for EXP in exp1; do
  SEED="${EXP#exp}"
  for CFG in "${CONFIGS[@]}"; do
    RUN_NAME="$(basename "${CFG}" .yaml)"
    RUN_NAME="${RUN_NAME#cont_train_chast_}"
    OUT_DIR="runs/${EXP}/${RUN_NAME}"

    echo "=== continue | ${EXP} | seed=${SEED} | ${CFG} -> ${OUT_DIR} ==="
    python3 train.py "${CFG}" --device "${DEVICE}" --seed "${SEED}" --out_dir "${OUT_DIR}"
  done
done
