#!/usr/bin/env bash
#set -euo pipefail

# Verzeichnis dieses Skripts (anomaly_detection/SAD)
#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Projektwurzel
#ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Datenordner mit ml2_*.csv / ml2_*.npy
#DATA_DIR="$ROOT_DIR/Datensaetze/SAD/JODIE"

#python "$SCRIPT_DIR/train.py" \
#  --dir_data "$DATA_DIR" \
#  --data_set reddit \
#  --anomaly_alpha 1e-1 \
#  --supc_alpha 5e-3 \
#  --mask_label \
#  --mask_ratio 0#.5
#python "$SCRIPT_DIR/train.py" \
#  --dir_data "$DATA_DIR" \
#  --data_set reddit \
#  --anomaly_alpha 1e-1 \
#  --supc_alpha 5e-3 \
#  --mask_label \
#  --mask_ratio 0.5

#python "$SCRIPT_DIR/train.py" \
#  --dir_data "$DATA_DIR" \
#  --data_set mooc \
#  --anomaly_alpha 1e-1 \
#  --supc_alpha 5e-3 \
#  --mask_label \
#  --mask_ratio 0.5
#python "$SCRIPT_DIR/train.py" \
#  --dir_data "$DATA_DIR" \
#  --data_set mooc \
#  --anomaly_alpha 1e-1 \
#  --supc_alpha 5e-3 \
#  --mask_label \
#  --mask_ratio 0.5

#python "$SCRIPT_DIR/train.py" \
#  --dir_data "$DATA_DIR" \
#  --data_set wikipedia \
#  --anomaly_alpha 1e-1 \
#  --supc_alpha 5e-3 \
#  --mask_label \
#  --mask_ratio 0.5
#python "$SCRIPT_DIR/train.py" \
#  --dir_data "$DATA_DIR" \
#  --data_set wikipedia \
#  --anomaly_alpha 1e-1 \
#  --supc_alpha 5e-3 \
#  --mask_label \
#  --mask_ratio 0.5
#echo "Training completed."





#hier stats in paper
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

COMMON_ARGS=(
  --mode sad
  --learning_rate 5e-4
  --batch_size 256
  --n_layer 2
  --n_heads 2
  --hidden_dim 128
  --n_neighbors 20
  --memory_size 4000
  --sample_size 1000
  --anomaly_alpha 1e-1
  --supc_alpha 1e-2
  --mask_label
  --mask_ratio 0.5
  --train_split 0.7
  --val_split 0.15
  --test_split 0.15
)

SEEDS="${SEEDS:-0 1 2 3}"
#Training: dataset=mooc, anomaly_per=0.01, seed=5
for SEED in ${SEEDS}; do
  echo "===================="
  echo "Run with seed=${SEED}"
  echo "===================="

  python "$SCRIPT_DIR/train.py" --dir_data "$DATA_DIR" --data_set reddit "${COMMON_ARGS[@]}" --seed "$SEED"
  python "$SCRIPT_DIR/train.py" --dir_data "$DATA_DIR" --data_set mooc "${COMMON_ARGS[@]}" --seed "$SEED"
  python "$SCRIPT_DIR/train.py" --dir_data "$DATA_DIR" --data_set wikipedia "${COMMON_ARGS[@]}" --seed "$SEED"
done