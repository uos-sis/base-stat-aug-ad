#!/bin/bash
# TGN node classification training on every dataset variant (reddit, mooc,
# wikipedia x raw/stats/stats_mlp) for the seeds 0 1 2 3.  Hyperparameters
# preserved from the original train_tgn.sh sweep (uniform across datasets).
# Run from the DyGLib directory:  bash tgn_train_all.sh
#SBATCH --job-name=tgn_node_classification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=slurm-tgn-%j.out
#SBATCH --error=slurm-tgn-%j.err
set -euo pipefail
SEEDS="${SEEDS:-0 1 2 3}"

cd "$(dirname "${BASH_SOURCE[0]}")"

NPARALLEL=3
sim() {
  "$@" &
  while (( $(jobs -pr | wc -l) >= NPARALLEL )); do
    wait -n
  done
}

TIME_FEAT_DIM=16
NUM_NEIGHBORS=20
NUM_LAYERS=2
NUM_HEADS=3
DROPOUT=0.3
WEIGHT_DECAY=0.0005
SAMPLE_NEIGHBOR_STRATEGY=recent
WANDB_PROJECT=CNA2026_TGN

DATASETS="
  reddit reddit_stats reddit_stats_mlp_ks_64
  wikipedia wikipedia_stats wikipedia_stats_mlp_ks_64
  mooc mooc_stats mooc_stats_mlp_ks_8
"

for seed in ${SEEDS}; do
  for ds in ${DATASETS}; do
    sim python -u train_node_classification.py --dataset_name "$ds" --model_name TGN \
      --num_neighbors $NUM_NEIGHBORS --num_layers $NUM_LAYERS --num_heads $NUM_HEADS \
      --dropout $DROPOUT --weight_decay $WEIGHT_DECAY \
      --sample_neighbor_strategy $SAMPLE_NEIGHBOR_STRATEGY --time_feat_dim "$TIME_FEAT_DIM" \
      --wandb_project "$WANDB_PROJECT" --seed "$seed" --num_runs 1
  done
done

wait