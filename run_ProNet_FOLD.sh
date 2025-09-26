#!/bin/bash
set -euo pipefail

# -----------------------------
# Hardware
# -----------------------------
DEVICE=0               # GPU id passed to the script 

# -----------------------------
# Dataset
# -----------------------------
DATASET='fold'
DATASET_PATH="./dataset/data"   # <- remove trailing slash; run_ProNet.py appends '/HomologyTAPE'
CUTOFF=10.0

# -----------------------------
# Model & Training config
# -----------------------------
LEVEL='allatom'      # 'aminoacid' , 'backbone', 'allatom'
NUM_BLOCKS=4
HIDDEN_CHANNELS=128
OUT_CHANNELS=1195

EPOCHS=1000
LR=0.0005
LR_DECAY_STEP=150
LR_DECAY_FACTOR=0.5

# -----------------------------
# Augmentation & Optimization
# -----------------------------
MASK_AATYPE=0.2
DROPOUT=0.3
BATCH_SIZE=32
EVAL_BATCH_SIZE=32
NUM_WORKERS=5

# -----------------------------
# SCHull evaluation controls
# -----------------------------
USE_SCHULL=1                      # 1 = evaluate per size-bucket & report best, 0 = classic whole-split
CT_LIST="50,100,150,200,250,300,400,500,2000"  # bucket upper bounds (like in the supplementary materials given in: https://openreview.net/forum?id=OIvg3MqWX2&noteId=UlvPDZvECD)
SIZE_MODE="residues"              # 'residues' | 'graph_nodes' | 'ca_atoms' (dataset decides)

# -----------------------------
# Launch
# -----------------------------
CMD=(
  python run_ProNet.py
  --device "$DEVICE"
  --dataset "$DATASET"
  --dataset_path "$DATASET_PATH"
  --cutoff "$CUTOFF"
  --batch_size "$BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --level "$LEVEL"
  --num_blocks "$NUM_BLOCKS"
  --hidden_channels "$HIDDEN_CHANNELS"
  --out_channels "$OUT_CHANNELS"
  --epochs "$EPOCHS"
  --lr "$LR"
  --lr_decay_step_size "$LR_DECAY_STEP"
  --lr_decay_factor "$LR_DECAY_FACTOR"
  --mask_aatype "$MASK_AATYPE"
  --dropout "$DROPOUT"
  --num_workers "$NUM_WORKERS"
  --mask --noise --deform --euler_noise --data_augment_eachlayer
)

# add SCHull flags only when requested
if [[ "$USE_SCHULL" == "1" ]]; then
  CMD+=( --use_schull_buckets --ct_lst "$CT_LIST" )
  # Optional: if you also added --size_mode in your dataset init path:
  CMD+=( --size_mode "$SIZE_MODE" )
fi

echo "Running:"
printf ' %q' "${CMD[@]}"; echo
"${CMD[@]}"
