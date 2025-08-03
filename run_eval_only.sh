#!/bin/bash

# Evaluation-only script - much faster than full training+evaluation
# This script loads a pre-trained checkpoint and only runs evaluation

echo "Starting evaluation-only mode..."

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 checkpoints/vit_base_patch16_224_optimized/final_model_vit_base_patch16_224.pt"
    exit 1
fi

CHECKPOINT_PATH="$1"
BACKBONE="vit_base_patch16_224"  # Adjust if using different model

# Clear GPU cache before starting
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Run evaluation only
python evaluate_only.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --backbone "$BACKBONE" \
    --ood cifar100,svhn,textures \
    --batch 32 \
    --data-dir /home/tanmoy/research/data \
    --output-dir eval_results_"$BACKBONE" \
    --use-cache

echo "Evaluation completed!" 