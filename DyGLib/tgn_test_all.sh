#!/bin/bash
#SBATCH --job-name=tgn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=cnode1,cnode2,cnode3,cnode4,cnode5,cnode6
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=slurm-tgn-%j.out
#SBATCH --error=slurm-tgn-%j.err
set -euo pipefail

MODEL=TGN
SEEDS="${SEEDS:-0 1 2 3}"
DATASETS="${DATASETS:-reddit reddit_stats reddit_stats_mlp_ks_64 wikipedia wikipedia_stats wikipedia_stats_mlp_ks_64 mooc mooc_stats mooc_stats_mlp_ks_8}"
SPLIT_DIM="${SPLIT_DIM:-}"
RAW_DATA_DIR="${RAW_DATA_DIR:-../raw_data/}"
# new per-model edge-feature SHAP pipeline (shap_tgn.py) is used on the
# *_stats / *_stats_mlp_* datasets; this caps the number of explained samples.
SHAP_MAX_SAMPLES="${SHAP_MAX_SAMPLES:-100}"
SHAP_SCRIPT=shap_tgn.py

for D in ${DATASETS}; do
  for S in ${SEEDS}; do
    DIR="checkpoints/node_classification_${MODEL}_${D}_seed${S}"

    if [[ ! -d "${DIR}" ]]; then
      echo "[skip] ${DIR}"
      continue
    fi

    # baseline evaluation (roc_auc / pr_auc) for every dataset
    # ARGS=(--dirs "${DIR}")
    # echo "===================="
    # echo "Test ${MODEL}: ${DIR}"
    # echo "===================="
    # python -u test.py "${ARGS[@]}"

    # dedicated edge-feature SHAP pipeline for the stats datasets
    if [[ "${D}" == *"_stats"* ]]; then
      ARGS=(--dirs "${DIR}" --shap_max_samples "${SHAP_MAX_SAMPLES}" --raw_data_dir "${RAW_DATA_DIR}")
      if [[ -n "${SPLIT_DIM}" ]]; then
        ARGS+=(--split_dim "${SPLIT_DIM}")
      fi
      echo "===================="
      echo "Edge-feature SHAP ${MODEL}: ${DIR}"
      echo "===================="
      python -u "${SHAP_SCRIPT}" "${ARGS[@]}"
    fi
  done
done
echo "TGN test completed."
