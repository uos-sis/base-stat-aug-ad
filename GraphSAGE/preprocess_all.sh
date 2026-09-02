#!/usr/bin/env bash
# Preprocess all dataset variants (reddit, mooc, wikipedia x raw/stats/stats_mlp)
# into JODIE snapshots consumed by the GraphSAGE training script.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RAW_DATA_DIR="${RAW_DATA_DIR:-../raw_data}"
OUT_DIR="${OUT_DIR:-./data}"
WINDOW_SIZE="${WINDOW_SIZE:-2000}"

for DS in \
  mooc mooc_stats mooc_stats_mlp_ks_8 \
  wikipedia wikipedia_stats wikipedia_stats_mlp_ks_64 \
  reddit reddit_stats reddit_stats_mlp_ks_64; do
  python3 prepare_jodie_snapshots.py \
    --raw_data_dir "${RAW_DATA_DIR}" \
    --data_set "${DS}" \
    --out_dir "${OUT_DIR}" \
    --window_size "${WINDOW_SIZE}"
done