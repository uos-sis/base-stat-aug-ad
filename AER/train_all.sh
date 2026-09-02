#!/usr/bin/env bash
# Train the AER model on every dataset variant (reddit, mooc, wikipedia x
# raw/stats/statsdim) for the seeds 0 1 2 3.
# Set WANDB_LOGGING=true to enable wandb logging (off by default).
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATASETS=(reddit mooc wikipedia)
FEATS=(raw stats statsdim)
SEEDS="${SEEDS:-0 1 2 3}"
WANDB_LOGGING="${WANDB_LOGGING:-false}"
FAILED=0

case "${WANDB_LOGGING}" in
  true|1|yes|on) WANDB_ARGS=(--wandb-logging) ;;
  *)             WANDB_ARGS=() ;;
esac

run_run() {
  local desc="$1"; shift
  echo "=== START $desc: $* ==="
  if python "$@" 2>&1 | tee "log/${desc}.out"; then
    echo "=== OK ${desc} ==="
  else
    echo "=== FAILED ${desc} (exit ${PIPESTATUS[0]}) ===" >&2
    FAILED=$((FAILED+1))
  fi
}

mkdir -p log

for ds in "${DATASETS[@]}"; do
  for feets in "${FEATS[@]}"; do
    for seed in ${SEEDS}; do
      run_run "${ds}_seed${seed}_${feets}" main.py -d "${ds}" \
        --data_mode real_synth \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --seed "${seed}" \
        --feets "${feets}" \
        "${WANDB_ARGS[@]}"
    done
  done
done

echo "TOTAL_FAILURES=${FAILED}"
[ "${FAILED}" -eq 0 ]