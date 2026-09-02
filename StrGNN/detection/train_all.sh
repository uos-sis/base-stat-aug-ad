#!/bin/bash
# Train StrGNN on all dataset variants for the accumulated (Main.py) and
# time-evolving (Main_statistic.py) graphs.  Hyperparameter groups per dataset
# are preserved from the original train_strgnn.sh sweep.
#   - train_all.sh                      # time-evolving only (default)
#   - GRAPHS="acc sta" bash train_all.sh   # both graph types
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

GRAPHS="${GRAPHS:-sta}"
SEEDS="${SEEDS:-1 2 3}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.0001}"
MAX_NODES_PER_HOP="${MAX_NODES_PER_HOP:-20}"
WANDB_PROJECT="${WANDB_PROJECT:-CNA2026_StrGNN6}"
NPARALLEL="${NPARALLEL:-4}"

HIGH_DIM_DS="reddit reddit_stats wikipedia wikipedia_stats"
MID_DIM_DS="reddit_stats_mlp_ks_64 wikipedia_stats_mlp_ks_64"
LOW_DIM_DS="mooc mooc_stats mooc_stats_mlp_ks_8"

# run at most NPARALLEL training calls concurrently
sim() {
  "$@" &
  while (( $(jobs -pr | wc -l) >= NPARALLEL )); do
    wait -n
  done
}

# args: graph ds_list hidden latent edge_proj dense mlp_dropout
run_group() {
  local graph="$1" ds_list="$2" hidden="$3" latent="$4" edge_proj="$5" dense="$6" mlp_dropout="$7"
  for ds in ${ds_list}; do
    for seed in ${SEEDS}; do
      if [ "$graph" = "acc" ]; then
        sim python -u Main.py --graph=acc_${ds}.npy --split=${ds} \
          --hidden "$hidden" --latent-dim "$latent" --edge-proj-dim "$edge_proj" \
          --dense-dim "$dense" --mlp-dropout "$mlp_dropout" \
          --num-epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
          --learning-rate "$LR" --max-nodes-per-hop "$MAX_NODES_PER_HOP" \
          --seed "$seed" --wandb-project "$WANDB_PROJECT" --no-progress-bar \
          --wandb-run-name acc_${ds}_seed${seed}
      else
        sim python -u Main_statistic.py --graph=sta_${ds}.npy --split=${ds} \
          --hidden "$hidden" --latent-dim "$latent" --edge-proj-dim "$edge_proj" \
          --dense-dim "$dense" --mlp-dropout "$mlp_dropout" \
          --num-epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
          --learning-rate "$LR" --max-nodes-per-hop "$MAX_NODES_PER_HOP" \
          --seed "$seed" --wandb-project "$WANDB_PROJECT" --no-progress-bar \
          --wandb-run-name sta_${ds}_seed${seed}
      fi
    done
  done
}

mkdir -p logs
for graph in ${GRAPHS}; do
  run_group "$graph" "$HIGH_DIM_DS" 128 32-32-16-1 64 256 0.2
  run_group "$graph" "$MID_DIM_DS"  128 32-32-16-1 64 256 0.2
  run_group "$graph" "$LOW_DIM_DS"  64  8-8-8-1    16 128 0.2
done
wait
echo "done" | tee -a logs/run_all.log