#!/bin/bash

# CIFAR-10/100 OOD Detection Evaluation Script
# This script runs comprehensive evaluations on CIFAR datasets

echo "============================================="
echo "CIFAR OOD Detection Evaluation Runner"
echo "============================================="

# Default parameters
DATASET="cifar10"
BACKBONE="resnet18"
BATCH_SIZE=32
DATA_DIR="/home/tanmoy/research/data"
OUTPUT_DIR="cifar_results"
PRETRAINED=false
OOD_DATASETS="cifar100,svhn,textures"
METHODS="energy,mahalanobis,msp,odin,ensemble"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --backbone|-b)
            BACKBONE="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --pretrained)
            PRETRAINED=true
            shift
            ;;
        --ood)
            OOD_DATASETS="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --dataset DATASET       CIFAR dataset (cifar10/cifar100) [default: cifar10]"
            echo "  --backbone BACKBONE     Model backbone [default: resnet18]"
            echo "  --batch BATCH_SIZE      Batch size [default: 32]"
            echo "  --data-dir DIR          Data directory [default: /home/tanmoy/research/data]"
            echo "  --output-dir DIR        Output directory [default: cifar_results]"
            echo "  --pretrained            Use pretrained weights"
            echo "  --ood DATASETS          OOD datasets (comma-separated) [default: cifar100,svhn,textures]"
            echo "  --methods METHODS       Detection methods [default: energy,mahalanobis,msp,odin,ensemble]"
            echo ""
            echo "Examples:"
            echo "  $0 --dataset cifar10 --backbone resnet18 --pretrained"
            echo "  $0 --dataset cifar100 --backbone vit_base_patch16_224 --ood 'cifar10,svhn'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build the command
CMD="python evaluate_cifar.py"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --backbone $BACKBONE"
CMD="$CMD --batch $BATCH_SIZE"
CMD="$CMD --data-dir '$DATA_DIR'"
CMD="$CMD --output-dir '$OUTPUT_DIR'"
CMD="$CMD --ood '$OOD_DATASETS'"
CMD="$CMD --methods '$METHODS'"

if [ "$PRETRAINED" = true ]; then
    CMD="$CMD --pretrained"
fi

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Backbone: $BACKBONE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Data Directory: $DATA_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  OOD Datasets: $OOD_DATASETS"
echo "  Detection Methods: $METHODS"
echo "  Pretrained: $PRETRAINED"
echo ""

echo "Running command:"
echo "$CMD"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the evaluation
eval $CMD

echo ""
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR" 