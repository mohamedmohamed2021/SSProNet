#!/bin/bash

# =========================
# Run SSProNet / ProNet on LBA dataset
# =========================

# GPU device
DEVICE="cuda:0"

# Dataset (LBA)
DATASET="lba"
DATASET_PATH="./dataset/data/LBA/data"         # LMDB data
DSSP_PATH="./dataset/data/LBA/dssp_pocket"     # DSSP features
CUTOFF=10.0

# Model & Training config
LEVEL="aminoacid"   
NUM_BLOCKS=4
HIDDEN_CHANNELS=128
OUT_CHANNELS=1        # regression → single value pK

EPOCHS=400
LR=0.0005
LR_DECAY_STEP=60
LR_DECAY_FACTOR=0.5

# Augmentation & Optimization
MASK_AATYPE=0.2
DROPOUT=0.3
BATCH_SIZE=16
EVAL_BATCH_SIZE=32
NUM_WORKERS=5

# -------------------------
# Launch the training
# -------------------------
python run_ProNet_LBA.py \
  --device $DEVICE \
  --dataset_path $DATASET_PATH \
  --dssp_path $DSSP_PATH \
  --cutoff $CUTOFF \
  --batch_size $BATCH_SIZE \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --level $LEVEL \
  --num_blocks $NUM_BLOCKS \
  --hidden_channels $HIDDEN_CHANNELS \
  --out_channels $OUT_CHANNELS \
  --epochs $EPOCHS \
  --lr $LR \
  --weight_decay 0.0 \
  --mask_aatype $MASK_AATYPE \
  --dropout $DROPOUT \
  --mask --noise --deform
