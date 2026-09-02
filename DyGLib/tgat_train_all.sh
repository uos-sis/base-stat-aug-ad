#!/bin/bash
# TGAT node classification training on every dataset variant (reddit, mooc,
# wikipedia x raw/stats/stats_mlp) for the seeds 0 1 2 3.  Per-dataset
# hyperparameters are preserved from the original train_tgat.sh sweep:
#   - wikipedia/wikipedia_stats use the 'uniform' neighbor strategy
#   - wikipedia_stats_mlp_ks_64 uses 1 layer (uniform)
#   - all others use 'recent' with 2 layers
# Run from the DyGLib directory:  bash tgat_train_all.sh
#SBATCH --job-name=tgat_node_classification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=slurm-tgat-%j.out
#SBATCH --error=slurm-tgat-%j.err
set -euo pipefail
SEEDS="${SEEDS:-0 1 2 3}"

# run at most NPARALLEL training calls concurrently
NPARALLEL=1
sim() {
  "$@" &
  while (( $(jobs -pr | wc -l) >= NPARALLEL )); do
    wait -n
  done
}

TIME_FEAT_DIM=16
NUM_NEIGHBORS=50
NUM_HEADS=3
DROPOUT=0.3
WEIGHT_DECAY=0.0005
SAMPLE_NEIGHBOR_STRATEGY=recent
WIKI_SAMPLE_NEIGHBOR_STRATEGY=uniform
WANDB_PROJECT=CNA2026_TGAT2

# <dataset> <sample strategy> <num layers>
DATASET_CONFIGS="
  reddit recent 2
  reddit_stats recent 2
  reddit_stats_mlp_ks_64 recent 2
  wikipedia uniform 2
  wikipedia_stats uniform 2
  wikipedia_stats_mlp_ks_64 uniform 1
  mooc recent 2
  mooc_stats recent 2
  mooc_stats_mlp_ks_8 recent 2
"

for seed in ${SEEDS}; do
  while read -r ds strategy layers; do
    [ -n "$ds" ] || continue
    sim python -u train_node_classification.py --dataset_name "$ds" --model_name TGAT \
      --num_neighbors $NUM_NEIGHBORS --num_layers "$layers" --num_heads $NUM_HEADS \
      --dropout $DROPOUT --weight_decay $WEIGHT_DECAY \
      --sample_neighbor_strategy "$strategy" --time_feat_dim "$TIME_FEAT_DIM" \
      --wandb_project "$WANDB_PROJECT" --seed "$seed" --num_runs 1
  done <<< "$DATASETS"
done

wait