#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Specify which GPU to use
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging

# Create timestamp for unique run identification
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory if it doesn't exist
mkdir -p logs

# Run configuration
DATASET="imagenet"
BACKBONE="dino_vitb16"
BATCH_SIZE=32
NUM_WORKERS=8
EPOCHS=30
LR=1e-4

# Run the training script with nohup
nohup python -u feature_train.py \
    --dataset $DATASET \
    --backbone $BACKBONE \
    --pretrained \
    --batch $BATCH_SIZE \
    --mixed-precision \
    --gradient-accumulation 4 \
    --grad-checkpoint \
    --num-workers $NUM_WORKERS \
    --data-dir /home/tanmoy/research/data \
    --epochs $EPOCHS \
    --lr $LR \
    > logs/training_${TIMESTAMP}.log 2>&1 &

# Save the process ID
echo $! > logs/training_${TIMESTAMP}.pid

echo "Training started with PID $(cat logs/training_${TIMESTAMP}.pid)"
echo "Log file: logs/training_${TIMESTAMP}.log"
echo "Monitor the training with: tail -f logs/training_${TIMESTAMP}.log" 