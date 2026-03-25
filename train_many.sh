#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"

CONFIGS=(
  "configs/train_chast_tdla_2.yaml"
  "configs/train_chast_tdlb_2.yaml"
  "configs/train_chast_tdlc_2.yaml"
  "configs/train_chast_tdld_2.yaml"
  "configs/train_chast_tdle_2.yaml"

  "configs/train_chast_tdla_23.yaml"
  "configs/train_chast_tdlb_23.yaml"
  "configs/train_chast_tdlc_23.yaml"
  "configs/train_chast_tdld_23.yaml"
  "configs/train_chast_tdle_23.yaml"

  "configs/train_chast_tdla_2711.yaml"
  "configs/train_chast_tdlb_2711.yaml"
  "configs/train_chast_tdlc_2711.yaml"
  "configs/train_chast_tdld_2711.yaml"
  "configs/train_chast_tdle_2711.yaml"
)

# Four experiment sets: exp2, exp3, exp4, exp5.
# We map expN -> seed N so each set is a different seed.
for EXP in exp1exp2 exp3 exp4 exp5; do
  SEED="${EXP#exp}" # exp2 -> 2
  for CFG in "${CONFIGS[@]}"; do
    RUN_NAME="$(basename "${CFG}" .yaml)"
    RUN_NAME="${RUN_NAME#train_chast_}"
    OUT_DIR="runs/${EXP}/${RUN_NAME}"

    echo "=== ${EXP} | seed=${SEED} | ${CFG} -> ${OUT_DIR} ==="
    python3 train.py "${CFG}" --device "${DEVICE}" --seed "${SEED}" --out_dir "${OUT_DIR}"
  done
done