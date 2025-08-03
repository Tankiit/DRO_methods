#!/bin/bash

# Base directories
DATA_DIR="/home/tanmoy/research/data"
CHECKPOINT_DIR="checkpoints"
OUTPUT_DIR="results"
RUNS_DIR="runs"

# Create necessary directories
mkdir -p $CHECKPOINT_DIR $OUTPUT_DIR $RUNS_DIR logs

# List of architectures to train
declare -a ARCHITECTURES=(
    "vit_base_patch16_224"
    "vit_large_patch16_224"
    "dino_vits16"
    "dino_vitb16"
    "resnet50"
)

# List of OOD datasets to evaluate against
declare -a OOD_DATASETS=("cifar100" "svhn" "textures")

# Common arguments for all runs
COMMON_ARGS="--data-dir $DATA_DIR \
             --mixed-precision \
             --grad-checkpoint \
             --empty-cache-freq 5"

# Function to determine optimal batch size based on model and available GPU memory
get_batch_size() {
    local MODEL=$1
    local GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
    
    case $MODEL in
        "vit_large"*)
            if [ $GPU_MEM -lt 16000 ]; then
                echo 8
            elif [ $GPU_MEM -lt 24000 ]; then
                echo 16
            else
                echo 32
            fi
            ;;
        "vit_base"*)
            if [ $GPU_MEM -lt 12000 ]; then
                echo 16
            elif [ $GPU_MEM -lt 20000 ]; then
                echo 32
            else
                echo 64
            fi
            ;;
        "dino_vitb"*)
            if [ $GPU_MEM -lt 12000 ]; then
                echo 16
            elif [ $GPU_MEM -lt 20000 ]; then
                echo 32
            else
                echo 64
            fi
            ;;
        "dino_vits"*)
            if [ $GPU_MEM -lt 10000 ]; then
                echo 32
            elif [ $GPU_MEM -lt 16000 ]; then
                echo 48
            else
                echo 96
            fi
            ;;
        *)
            if [ $GPU_MEM -lt 8000 ]; then
                echo 32
            elif [ $GPU_MEM -lt 16000 ]; then
                echo 64
            else
                echo 128
            fi
            ;;
    esac
}

# Function to determine gradient accumulation steps
get_grad_accum() {
    local MODEL=$1
    local BATCH_SIZE=$2
    
    case $MODEL in
        "vit_large"*)
            if [ $BATCH_SIZE -lt 16 ]; then
                echo 8
            elif [ $BATCH_SIZE -lt 32 ]; then
                echo 4
            else
                echo 2
            fi
            ;;
        "vit_base"*|"dino_vitb"*)
            if [ $BATCH_SIZE -lt 32 ]; then
                echo 4
            else
                echo 2
            fi
            ;;
        *)
            if [ $BATCH_SIZE -lt 64 ]; then
                echo 2
            else
                echo 1
            fi
            ;;
    esac
}

# Function to determine learning rate
get_learning_rate() {
    local MODEL=$1
    local BATCH_SIZE=$2
    local GRAD_ACCUM=$3
    local EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM))
    
    case $MODEL in
        "vit_large"*)
            if [ $EFFECTIVE_BATCH -lt 64 ]; then
                echo "5e-5"
            else
                echo "1e-4"
            fi
            ;;
        "vit_base"*|"dino_vitb"*)
            if [ $EFFECTIVE_BATCH -lt 64 ]; then
                echo "1e-4"
            else
                echo "2e-4"
            fi
            ;;
        *)
            if [ $EFFECTIVE_BATCH -lt 128 ]; then
                echo "5e-4"
            else
                echo "1e-3"
            fi
            ;;
    esac
}

# Function to train a single architecture
train_architecture() {
    local ARCH=$1
    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
    # Create architecture-specific directories
    local ARCH_CHECKPOINT_DIR="${CHECKPOINT_DIR}/${ARCH}"
    local ARCH_OUTPUT_DIR="${OUTPUT_DIR}/${ARCH}"
    local ARCH_RUNS_DIR="${RUNS_DIR}/${ARCH}"
    mkdir -p "$ARCH_CHECKPOINT_DIR" "$ARCH_OUTPUT_DIR" "$ARCH_RUNS_DIR"
    
    # Get optimal hyperparameters
    local BATCH_SIZE=$(get_batch_size $ARCH)
    local GRAD_ACCUM=$(get_grad_accum $ARCH $BATCH_SIZE)
    local LR=$(get_learning_rate $ARCH $BATCH_SIZE $GRAD_ACCUM)
    
    echo "==========================================="
    echo "Starting training for $ARCH"
    echo "Batch Size: $BATCH_SIZE"
    echo "Gradient Accumulation: $GRAD_ACCUM"
    echo "Learning Rate: $LR"
    echo "==========================================="
    
    # Training log file
    local LOG_FILE="logs/train_${ARCH}_${TIMESTAMP}.log"
    
    # Build comma-separated OOD list (e.g. "cifar100,svhn,textures")
    local OOD_LIST=$(IFS=','; echo "${OOD_DATASETS[*]}")

    # Single training run followed by evaluation on all OOD datasets
    python feature_train.py $COMMON_ARGS \
        --backbone $ARCH \
        --dataset imagenet \
        --ood $OOD_LIST \
        --batch $BATCH_SIZE \
        --gradient-accumulation $GRAD_ACCUM \
        --lr $LR \
        --pretrained \
        --epochs 30 \
        --checkpoint-dir "$ARCH_CHECKPOINT_DIR" \
        --output-dir "$ARCH_OUTPUT_DIR" \
        --runs-dir "$ARCH_RUNS_DIR" > "$LOG_FILE" 2>&1

    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo "Training completed successfully for $ARCH (evaluated on $OOD_LIST)"
        echo "Results saved in:"
        echo "- Checkpoints: $ARCH_CHECKPOINT_DIR"
        echo "- Outputs: $ARCH_OUTPUT_DIR"
        echo "- Tensorboard: $ARCH_RUNS_DIR"
        echo "- Log: $LOG_FILE"
    else
        echo "Training failed for $ARCH. Check $LOG_FILE for details"
    fi
}

# Function to monitor GPU usage
monitor_gpu() {
    local LOG_FILE="logs/gpu_monitoring_$(date +%Y%m%d).log"
    while true; do
        echo "----------------------------------------" >> "$LOG_FILE"
        date >> "$LOG_FILE"
        nvidia-smi >> "$LOG_FILE"
        sleep 60
    done
}

# Start GPU monitoring
monitor_gpu &
MONITOR_PID=$!

# Train each architecture sequentially
echo "Starting sequential training of architectures..."
for ARCH in "${ARCHITECTURES[@]}"; do
    train_architecture $ARCH
    
    # Optional: Wait between architectures to let GPU cool down
    sleep 300  # 5 minutes cooldown
done

# Stop GPU monitoring
kill $MONITOR_PID

# Combine results
echo "All training completed!"
echo "Results are organized by architecture in:"
echo "- $CHECKPOINT_DIR"
echo "- $OUTPUT_DIR"
echo "- $RUNS_DIR"
echo "GPU monitoring logs available in: logs/gpu_monitoring_*.log"

# Create a summary of results
SUMMARY_FILE="${OUTPUT_DIR}/training_summary_$(date +%Y%m%d).txt"
echo "Training Summary" > "$SUMMARY_FILE"
echo "=================" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for ARCH in "${ARCHITECTURES[@]}"; do
    echo "Architecture: $ARCH" >> "$SUMMARY_FILE"
    echo "-------------------" >> "$SUMMARY_FILE"
    echo "Checkpoints: ${CHECKPOINT_DIR}/${ARCH}" >> "$SUMMARY_FILE"
    echo "Results: ${OUTPUT_DIR}/${ARCH}" >> "$SUMMARY_FILE"
    echo "Tensorboard: ${RUNS_DIR}/${ARCH}" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
done

echo "Summary saved to: $SUMMARY_FILE" 