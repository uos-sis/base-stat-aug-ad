#!/bin/bash
# DyGFormer node classification training on every dataset variant (reddit,
# mooc, wikipedia x raw/stats/stats_mlp) for the seeds 0 1 2 3.  Per-dataset
# hyperparameters are preserved from the original train_dygformer.sh sweep:
#   - reddit*:    dropout 0.2, channel_embedding_dim 48
#   - wikipedia*: dropout 0.1, channel_embedding_dim 48
#   - mooc*:      dropout 0.1, channel_embedding_dim 36
# Run from the DyGLib directory:  bash dygformer_train_all.sh
#SBATCH --job-name=dygformer_node_classification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=slurm-dygformer-%j.out
#SBATCH --error=slurm-dygformer-%j.err
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
NUM_LAYERS=2
NUM_HEADS=2
MAX_INPUT_SEQUENCE_LENGTH=64
PATCH_SIZE=2
WANDB_PROJECT=CNA2026_DyGFormer2

# <dataset> <dropout> <channel_embedding_dim>
DATASETS="
  reddit 0.2 48
  reddit_stats 0.2 48
  reddit_stats_mlp_ks_64 0.2 48
  wikipedia 0.1 48
  wikipedia_stats 0.1 48
  wikipedia_stats_mlp_ks_64 0.1 48
  mooc 0.1 36
  mooc_stats 0.1 36
  mooc_stats_mlp_ks_8 0.1 36
"

for seed in ${SEEDS}; do
  while read -r ds dropout channel; do
    [ -n "$ds" ] || continue
    sim python -u train_node_classification.py --dataset_name "$ds" --model_name DyGFormer \
      --num_layers $NUM_LAYERS --num_heads $NUM_HEADS \
      --max_input_sequence_length $MAX_INPUT_SEQUENCE_LENGTH --patch_size $PATCH_SIZE \
      --dropout "$dropout" --time_feat_dim "$TIME_FEAT_DIM" \
      --channel_embedding_dim "$channel" --wandb_project "$WANDB_PROJECT" \
      --seed "$seed" --num_runs 1
  done <<< "$DATASETS"
done

wait