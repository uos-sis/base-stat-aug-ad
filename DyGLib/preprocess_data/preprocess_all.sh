#!/usr/bin/env bash
# Preprocess all dataset variants (reddit, mooc, wikipedia x raw/stats/stats_mlp).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

for DS in \
  mooc mooc_stats mooc_stats_mlp_ks_8 \
  wikipedia wikipedia_stats wikipedia_stats_mlp_ks_64 \
  reddit reddit_stats reddit_stats_mlp_ks_64; do
  python preprocess_data.py --dataset_name "${DS}"
done