#!/bin/bash

# Memory-optimized training script for Vision Transformer
# This script uses smaller batch sizes and memory-efficient settings

echo "Starting memory-optimized Vision Transformer training..."

# Clear GPU cache before starting
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Run training with memory-optimized parameters
python feature_train.py \
    --dataset imagenet \
    --ood cifar100,svhn,textures \
    --backbone vit_base_patch16_224 \
    --pretrained \
    --epochs 30 \
    --batch 8 \
    --lr 0.0001 \
    --pixel-radius 0.03 \
    --feature-radius 0.1 \
    --pixel-weight 0.3 \
    --feature-weight 0.5 \
    --cross-weight 0.2 \
    --grad-checkpoint \
    --num-workers 2 \
    --strategy-batch-size 2 \
    --empty-cache-freq 3 \
    --mixed-precision \
    --gradient-accumulation 8 \
    --data-dir /home/tanmoy/research/data \
    --checkpoint-dir checkpoints/vit_base_patch16_224_optimized \
    --output-dir results/vit_base_patch16_224_optimized \
    --runs-dir runs/vit_base_patch16_224_optimized

echo "Training completed!" 