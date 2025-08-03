#!/bin/bash

# Baseline Comparison Examples for CIFAR OOD Detection
# This script demonstrates various ways to run the baseline comparison

echo "🔬 CIFAR OOD Detection - Baseline Comparison Examples"
echo "======================================================"

# Example 1: Quick baseline comparison with latest checkpoint
echo ""
echo "Example 1: Quick Baseline Comparison (Latest Checkpoint)"
echo "--------------------------------------------------------"
echo "Command:"
echo "python run_baseline_comparison.py --quick-mode"
echo ""

# Example 2: Full baseline comparison with specific checkpoint
echo "Example 2: Full Baseline Comparison (Specific Checkpoint)"
echo "---------------------------------------------------------"
echo "Command:"
echo "python run_baseline_comparison.py --checkpoint checkpoints/best_model_20250726_120417.pt --dataset cifar10 --backbone resnet18"
echo ""

# Example 3: Compare only specific methods
echo "Example 3: Compare Specific Methods Only"
echo "----------------------------------------"
echo "Command:"
echo "python run_baseline_comparison.py --methods MSP ODIN Energy H-DRO --quick-mode"
echo ""

# Example 4: CIFAR-100 comparison with ViT
echo "Example 4: CIFAR-100 with Vision Transformer"
echo "--------------------------------------------"
echo "Command:"
echo "python run_baseline_comparison.py --dataset cifar100 --backbone vit_base_patch16_224 --checkpoint-dir checkpoints/vit_base_patch16_224"
echo ""

# Example 5: Custom output directory
echo "Example 5: Custom Output Directory"
echo "----------------------------------"
echo "Command:"
echo "python run_baseline_comparison.py --output-dir my_baseline_results --run-name resnet18_comparison"
echo ""

# Usage instructions
echo ""
echo "📋 USAGE INSTRUCTIONS:"
echo "======================"
echo ""
echo "1. Basic Usage:"
echo "   python run_baseline_comparison.py [OPTIONS]"
echo ""
echo "2. Key Options:"
echo "   --checkpoint       : Path to specific checkpoint file"
echo "   --checkpoint-dir   : Directory containing checkpoints (default: checkpoints)"
echo "   --dataset          : cifar10, cifar100 (default: cifar10)"
echo "   --backbone         : resnet18, resnet50, vit_base_patch16_224 (default: resnet18)"
echo "   --quick-mode       : Run with fewer methods for faster execution"
echo "   --methods          : Specific methods to test (e.g., MSP ODIN Energy)"
echo "   --output-dir       : Output directory (default: baseline_comparison_results)"
echo "   --run-name         : Custom name for the run"
echo ""
echo "3. Available Methods:"
echo "   📊 STANDARD METHODS: MSP, ODIN, Mahalanobis, Energy"
echo "   🏆 RECENT SOTA: KNN, ViM, ASH, GradNorm"
echo "   🔄 VIRTUAL SAMPLE METHODS: VOS, NPOS, CSI, DREAM_OOD"
echo "   💪 DRO METHODS: Wasserstein_DRO, Group_DRO, CVaR_DRO"
echo "   ⚡ ENERGY-BASED: JEM, EBGAN, EBM_OOD"
echo "   🎯 OUR METHOD: H-DRO"
echo ""
echo "4. Output Structure:"
echo "   output_dir/"
echo "   ├── visualizations/"
echo "   │   ├── performance_heatmap.png"
echo "   │   ├── ranking_charts.png"
echo "   │   ├── statistical_analysis.png"
echo "   │   └── improvement_analysis.png"
echo "   ├── tables/"
echo "   │   ├── performance_comparison.csv"
echo "   │   ├── performance_comparison.tex"
echo "   │   ├── efficiency_comparison.csv"
echo "   │   └── efficiency_comparison.tex"
echo "   └── comprehensive_baseline_comparison.json"
echo ""
echo "5. Generated Analyses:"
echo "   📈 Performance Ranking by AUROC"
echo "   ⚡ Efficiency Ranking by Throughput"
echo "   📊 Statistical Significance Tests"
echo "   🎯 H-DRO Improvement Analysis"
echo "   🔥 Comprehensive Visualizations"
echo ""

# Interactive mode
echo "🚀 READY TO RUN BASELINE COMPARISON?"
echo "===================================="
echo ""
echo "Choose an example to run:"
echo "1) Quick comparison with latest checkpoint"
echo "2) Full comparison with specific checkpoint"
echo "3) Compare only standard methods (MSP, ODIN, Energy, H-DRO)"
echo "4) Custom configuration"
echo "5) Exit"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "Running quick baseline comparison..."
        python run_baseline_comparison.py --quick-mode
        ;;
    2)
        echo "Enter specific checkpoint path:"
        read -p "Checkpoint path: " checkpoint_path
        if [ -f "$checkpoint_path" ]; then
            echo "Running full comparison with specific checkpoint..."
            python run_baseline_comparison.py --checkpoint "$checkpoint_path"
        else
            echo "❌ Checkpoint file not found: $checkpoint_path"
        fi
        ;;
    3)
        echo "Running comparison with standard methods only..."
        python run_baseline_comparison.py --methods MSP ODIN Mahalanobis Energy H-DRO --quick-mode
        ;;
    4)
        echo "Enter custom configuration:"
        read -p "Dataset (cifar10/cifar100): " dataset
        read -p "Backbone (resnet18/resnet50/vit_base_patch16_224): " backbone
        read -p "Checkpoint directory (default: checkpoints): " checkpoint_dir
        read -p "Quick mode? (y/n): " quick_mode
        
        cmd="python run_baseline_comparison.py --dataset $dataset --backbone $backbone"
        
        if [ ! -z "$checkpoint_dir" ]; then
            cmd="$cmd --checkpoint-dir $checkpoint_dir"
        fi
        
        if [ "$quick_mode" = "y" ]; then
            cmd="$cmd --quick-mode"
        fi
        
        echo "Running custom analysis..."
        echo "Command: $cmd"
        eval $cmd
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        ;;
esac

echo ""
echo "✅ Baseline comparison completed! Check the output directory for results." 