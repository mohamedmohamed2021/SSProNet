#!/bin/bash

# ===== User settings =====
DEVICE=0

# Point to the PARENT folder that contains "ProtFunct/"
# Example: /workspace/workspace/CD_Conv_Plus_ProNet/dataset/data
DATASET_PATH="/workspace/workspace/CD_Conv_Plus_ProNet/dataset/data"

# ===== Dataset & geometry =====
DATASET='func'          # EC / ProtFunct
CUTOFF=10.0

# ===== Model =====
LEVEL='backbone'        # 'aminoacid' , 'backbone', 'allatom'
NUM_BLOCKS=4
HIDDEN_CHANNELS=128
OUT_CHANNELS=384        # *** EC has 384 classes ***

# ===== Training =====
EPOCHS=400
LR=0.0005
LR_DECAY_STEP=60
LR_DECAY_FACTOR=0.5
DROPOUT=0.3

# ===== Augmentation & loader =====
MASK_AATYPE=0.2
BATCH_SIZE=32
EVAL_BATCH_SIZE=32
NUM_WORKERS=5

python run_ProNet.py \
  --device $DEVICE \
  --dataset $DATASET \
  --dataset_path $DATASET_PATH \
  --cutoff $CUTOFF \
  --batch_size $BATCH_SIZE \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --level $LEVEL \
  --num_blocks $NUM_BLOCKS \
  --hidden_channels $HIDDEN_CHANNELS \
  --out_channels $OUT_CHANNELS \
  --epochs $EPOCHS \
  --lr $LR \
  --lr_decay_step_size $LR_DECAY_STEP \
  --lr_decay_factor $LR_DECAY_FACTOR \
  --mask_aatype $MASK_AATYPE \
  --dropout $DROPOUT \
  --num_workers $NUM_WORKERS \
  --mask --noise --deform --euler_noise --data_augment_eachlayer
