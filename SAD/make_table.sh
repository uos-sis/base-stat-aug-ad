#!/usr/bin/env bash
# Generate the LaTeX comparison table (see make_table.py) for the SAD runs.
# Mirrors GraphSAGE/make_table.py; just a thin shell wrapper that saves the
# table text to a .tex file.  Run from the SAD directory (anomaly_detection/SAD):
#
#     # default: use ./checkpoints, write ./make_table.tex
#     bash make_table.sh
#
#     # also merge the GraphSAGE runs so each model gets its own row
#     bash make_table.sh --roots ./checkpoints ../GraphSAGE/checkpoints
#
#     # change the output file
#     bash make_table.sh -o /tmp/feature_augmentation.tex
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="$SCRIPT_DIR/make_table.tex"
ROOTS=("$SCRIPT_DIR/checkpoints")

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)
      OUTPUT="$2"
      shift 2
      ;;
    *)
      ROOTS+=("$1")
      shift
      ;;
  esac
done

python3 "$SCRIPT_DIR/make_table.py" --checkpoint_root "${ROOTS[@]}" | tee "$OUTPUT"
echo
echo "saved LaTeX table to: $OUTPUT"