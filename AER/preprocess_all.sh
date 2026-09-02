#!/usr/bin/env bash
# Preprocess all dataset variants (reddit, mooc, wikipedia x raw/stats/statsdim).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python preprocess.py --datasets reddit mooc wikipedia \
    --feets raw stats statsdim --validation-rate 0.7 --test-rate 0.85