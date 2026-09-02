#!/usr/bin/env bash
# Train GraphSAGE on every dataset variant (reddit, mooc, wikipedia x
# raw/stats/stats_mlp) for the seeds 0 1 2 3.  Runs from the GraphSAGE dir.
# Requires preprocessed snapshots (preprocess_all.sh).  Hidden dims follow the
# original sweep: 64 for reddit*/wikipedia*, 8 for mooc*.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SEEDS="${SEEDS:-0 1 2 3}"
FAILED=0

run_run() {
  local desc="$1"; shift
  echo "=== START $desc: $* ==="
  if python -m graphsage.torch_train "$@" 2>&1 | tee "log/${desc}.out"; then
    echo "=== OK ${desc} ==="
  else
    echo "=== FAILED ${desc} (exit ${PIPESTATUS[0]}) ===" >&2
    FAILED=$((FAILED+1))
  fi
}

mkdir -p log

for dim in 64 8; do
  if [ "$dim" = "64" ]; then
    DS_LIST=(reddit reddit_stats reddit_stats_mlp_ks_64 \
             wikipedia wikipedia_stats wikipedia_stats_mlp_ks_64)
  else
    DS_LIST=(mooc mooc_stats mooc_stats_mlp_ks_8)
  fi
  for ds in "${DS_LIST[@]}"; do
    for seed in ${SEEDS}; do
      run_run "${ds}_seed${seed}" \
        --data_set "${ds}" --seed "${seed}" --dim_1 "${dim}" --dim_2 "${dim}"
    done
  done
done

echo "TOTAL_FAILURES=${FAILED}"
[ "${FAILED}" -eq 0 ]