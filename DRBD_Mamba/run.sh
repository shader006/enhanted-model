#!/bin/bash

# Define arguments
# DATA_PATH="./TrainingData/"
DATA_PATH="/home/cuc.buithi/BRATS/BraTS2021_Training_Data"
# FILE_PATH="./Folds_JSON/brats2023_val.json"
FILE_PATH="../brats23_split_70_10_20.json"
# FILE_PATH="./Folds_JSON/IMFUSE_UPDATED_JSON_CM.json"
MODALITIES="t1n t2w t1c t2f"
# NOTE: INPUT_SIZE and BATCH_SIZE are managed in BRATS23/settings.py
# NFS clusters can emit .nfs cleanup errors with multiprocessing workers.
NUM_WORKERS=0
RESUME="--resume"  # Add this flag if you want to resume training; remove it if you are starting from scratch
VQVAETRAINING="--vqvae_training"
LDMTRAINING="--ldmtraining"
CHECKPOINT_DIR="./model/checkpoints"  # Directory for saving and loading checkpoints
VQVAE="--VQVAE"
LDM="--LDM"
COND="--COND"
CONDTRAINING="--cond_training"
LMUNET="--LMUNET"
LMUNETTRAINING="--lmunet_training"

# Run the Python script with the arguments
    python main.py \
    --data_path $DATA_PATH \
    --file_path $FILE_PATH \
    --modalities $MODALITIES \
    --num_workers $NUM_WORKERS \
    --checkpoint_dir $CHECKPOINT_DIR \
    $VQVAE \
    $VQVAETRAINING \
    # $RESUME \
    # # #--checkpoint_dir $CHECKPOINT_DIR
