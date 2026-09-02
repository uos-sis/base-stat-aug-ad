#!/usr/bin/env bash
# AER test pipeline: compute AUC + SHAP feature importances from the TEST SPLIT
# for every saved best model (per dataset x feature source x seed).
# No wandb required. Results are written to results/shap/<dataset>-<feets>/.
#   bash test_all.sh
#   DATASETS="mooc" FEATS="stats" SEEDS="0 1" bash test_all.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATASETS="${DATASETS:-reddit mooc wikipedia}"
FEATS="${FEATS:-raw stats statsdim}"
SEEDS="${SEEDS:-0 1 2 3}"
DATA_MODE="${DATA_MODE:-real_synth}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
VAL_RATIO="${VAL_RATIO:-0.15}"
SHAP_MAX_SAMPLES="${SHAP_MAX_SAMPLES:-2000}"
SHAP_BACKGROUND_SAMPLES="${SHAP_BACKGROUND_SAMPLES:-40}"
SHAP_NSAMPLES="${SHAP_NSAMPLES:-20}"

FAILED=0
run_one() {
  local desc="$1"; shift
  echo "=== START $desc: $* ==="
  if python "$@" 2>&1 | tee "log/${desc}.out"; then
    echo "=== OK ${desc} ==="
  else
    echo "=== FAILED ${desc} (exit ${PIPESTATUS[0]}) ===" >&2
    FAILED=$((FAILED+1))
  fi
}

mkdir -p log results/shap

for data in ${DATASETS}; do
  for feats in ${FEATS}; do
    run_one "${data}_${feats}" calcshape.py -d "${data}" --feets "${feats}" \
      --data_mode "${DATA_MODE}" --train_ratio "${TRAIN_RATIO}" --val_ratio "${VAL_RATIO}" \
      --seeds ${SEEDS} \
      --shap_max_samples "${SHAP_MAX_SAMPLES}" \
      --shap_background_samples "${SHAP_BACKGROUND_SAMPLES}" \
      --shap_nsamples "${SHAP_NSAMPLES}"
  done
done

echo "TOTAL_FAILURES=${FAILED}"
[ "${FAILED}" -eq 0 ]