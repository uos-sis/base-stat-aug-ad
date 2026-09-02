#!/usr/bin/env bash
# Preprocess all dataset variants (reddit, mooc, wikipedia x raw/stats/stats_mlp)
# into the StrGNN snapshot format and sync them into detection/data and data_sta.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python prepare_modus2_data.py --sync-repo