#!/bin/bash
# Compute per-dimension SHAP importance for every checkpoint whose dataset
# contains "_stats" (i.e. *_stats and *_stats_mlp_ks_*), mirroring the DyGLib
# train-script layout.  Run from the detection directory:
#
#     bash test_all.sh          # or: sbatch test_all.sh
#
# Results (importance.class_{all,ano,normal}.mean_abs_shap_per_feature) are
# written into each checkpoint dir's result.json by detection/test.py.
#
# SLURM: 1 GPU, 16 CPUs on the default (sis) partition, 3 runs in parallel.
#SBATCH --job-name=strgnn_shap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=cnode1,cnode2,cnode3,cnode4,cnode5,cnode6
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=slurm-strgnn-shap-%j.out
#SBATCH --error=slurm-strgnn-shap-%j.err

# run at most NPARALLEL training calls concurrently
NPARALLEL=3
sim() {
  "$@" &
  while (( $(jobs -pr | wc -l) >= NPARALLEL )); do
    wait -n
  done
}

CHECKPOINT_ROOT=./checkpoints
GPU=0
SHAP_MAX_SAMPLES=200
SHAP_BACKGROUND_SAMPLES=50
SHAP_NSAMPLES=5

for d in "$CHECKPOINT_ROOT"/*stats*; do
  [ -d "$d" ] || continue
  sim python -u test.py --dirs "$(basename "$d")" --compute-shap \
    --gpu "$GPU" --shap-max-samples "$SHAP_MAX_SAMPLES" \
    --shap-background-samples "$SHAP_BACKGROUND_SAMPLES" \
    --shap-nsamples "$SHAP_NSAMPLES"
done

wait
