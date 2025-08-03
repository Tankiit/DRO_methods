#!/bin/bash

# Comprehensive Analysis Examples for CIFAR OOD Detection
# This script demonstrates various ways to run ablation studies and efficiency analysis

echo "🔬 CIFAR OOD Detection - Comprehensive Analysis Examples"
echo "======================================================"

# Example 1: Quick ablation study on CIFAR-10 with ResNet-18
echo ""
echo "Example 1: Quick Ablation Study (CIFAR-10 + ResNet-18)"
echo "------------------------------------------------------"
echo "Command:"
echo "python run_comprehensive_analysis.py --dataset cifar10 --backbone resnet18 --pretrained --analysis-type ablation --quick-mode"
echo ""

# Example 2: Full efficiency analysis on CIFAR-100 with ViT
echo "Example 2: Efficiency Analysis (CIFAR-100 + ViT)"
echo "------------------------------------------------"
echo "Command:"
echo "python run_comprehensive_analysis.py --dataset cifar100 --backbone vit_base_patch16_224 --pretrained --analysis-type efficiency"
echo ""

# Example 3: Complete analysis with custom parameters
echo "Example 3: Complete Analysis (Both Ablation + Efficiency)"
echo "--------------------------------------------------------"
echo "Command:"
echo "python run_comprehensive_analysis.py --dataset cifar10 --backbone resnet50 --pretrained --analysis-type both --epochs 10 --batch-size 64"
echo ""

# Example 4: Architecture comparison study
echo "Example 4: Architecture Comparison Study"
echo "---------------------------------------"
echo "Commands (run separately):"
echo "python run_comprehensive_analysis.py --dataset cifar10 --backbone resnet18 --pretrained --output-dir arch_comparison_resnet18"
echo "python run_comprehensive_analysis.py --dataset cifar10 --backbone resnet50 --pretrained --output-dir arch_comparison_resnet50"
echo "python run_comprehensive_analysis.py --dataset cifar10 --backbone vit_base_patch16_224 --pretrained --output-dir arch_comparison_vit"
echo ""

# Example 5: Dataset comparison study
echo "Example 5: Dataset Comparison Study"
echo "----------------------------------" 
echo "Commands (run separately):"
echo "python run_comprehensive_analysis.py --dataset cifar10 --backbone resnet18 --pretrained --output-dir dataset_comparison_cifar10"
echo "python run_comprehensive_analysis.py --dataset cifar100 --backbone resnet18 --pretrained --output-dir dataset_comparison_cifar100"
echo ""

# Example 6: Training Log Analysis
echo "Example 6: Training Log Analysis"
echo "-------------------------------"
echo "Commands:"
echo "python training_log_analyzer.py --log-dir runs --plot-types loss_curves accuracy_curves lr_schedule"
echo "python training_log_analyzer.py --training-log logs/train_resnet50_20250727_135101.log --plot-types dashboard loss_components"
echo "python training_log_analyzer.py --log-dir runs --training-log training_fixed.log --plot-types all"
echo ""

# Usage instructions
echo ""
echo "📋 USAGE INSTRUCTIONS:"
echo "======================"
echo ""
echo "1. Basic Usage:"
echo "   python run_comprehensive_analysis.py [OPTIONS]"
echo ""
echo "2. Key Options:"
echo "   --dataset        : cifar10, cifar100"
echo "   --backbone       : resnet18, resnet50, vit_base_patch16_224, efficientnet_b0"
echo "   --analysis-type  : ablation, efficiency, both"
echo "   --pretrained     : Use pretrained weights (recommended)"
echo "   --quick-mode     : Fast mode with reduced parameters"
echo "   --epochs         : Number of training epochs (default: 5)"
echo "   --batch-size     : Batch size (default: 32)"
echo "   --output-dir     : Output directory (default: comprehensive_analysis_results)"
echo ""
echo "3. Output Structure:"
echo "   output_dir/"
echo "   ├── ablation_study/"
echo "   │   ├── plots/                 # Performance plots"
echo "   │   ├── tables/                # LaTeX tables"
echo "   │   ├── logs/                  # Detailed logs"
echo "   │   └── complete_ablation_results.json"
echo "   ├── efficiency_analysis/"
echo "   │   ├── plots/                 # Memory/timing plots"  
echo "   │   ├── tables/                # Efficiency tables"
echo "   │   ├── logs/                  # Analysis logs"
echo "   │   └── complete_efficiency_results.json"
echo "   └── combined_results_summary.json"
echo ""
echo "4. Generated Analyses:"
echo ""
echo "   📊 ABLATION STUDY:"
echo "   • Virtual Sample Strategy Comparison"
echo "   • Progressive Training Analysis"
echo "   • Hyperparameter Sensitivity"
echo "   • Architecture Generalization"
echo ""
echo "   ⚡ EFFICIENCY ANALYSIS:"
echo "   • Memory Usage vs Batch Size"
echo "   • Training Time Breakdown"
echo "   • Scalability Analysis"
echo "   • Baseline Method Comparison"
echo ""
echo "5. For Paper Writing:"
echo "   • Use LaTeX tables from tables/ directories"
echo "   • Include plots from plots/ directories"
echo "   • Reference detailed results in JSON files"
echo ""
echo "6. Python API Usage:"
echo "   from ablation_analysis import run_ablation_study, run_efficiency_analysis"
echo "   config = {'dataset': 'cifar10', 'backbone': 'resnet18', ...}"
echo "   ablation_results = run_ablation_study(config)"
echo "   efficiency_results = run_efficiency_analysis(config)"
echo ""

# Interactive mode
echo "🚀 READY TO RUN ANALYSIS?"
echo "========================="
echo ""
echo "Choose an example to run:"
echo "1) Quick ablation study (CIFAR-10 + ResNet-18)"
echo "2) Efficiency analysis (CIFAR-10 + ResNet-18)"  
echo "3) Complete analysis (both ablation + efficiency)"
echo "4) Training log analysis (TensorBoard + text logs)"
echo "5) Baseline comparison (OOD detection methods)"
echo "6) Custom configuration"
echo "7) Exit"
echo ""

read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        echo "Running quick ablation study..."
        python run_comprehensive_analysis.py --dataset cifar10 --backbone resnet18 --pretrained --analysis-type ablation --quick-mode
        ;;
    2)
        echo "Running efficiency analysis..."
        python run_comprehensive_analysis.py --dataset cifar10 --backbone resnet18 --pretrained --analysis-type efficiency --quick-mode
        ;;
    3)
        echo "Running complete analysis..."
        python run_comprehensive_analysis.py --dataset cifar10 --backbone resnet18 --pretrained --analysis-type both --quick-mode
        ;;
    4)
        echo "Running training log analysis..."
        echo "Available options:"
        echo "a) Analyze TensorBoard logs"
        echo "b) Analyze text training logs"
        echo "c) Analyze both"
        read -p "Choose option (a/b/c): " log_choice
        
        case $log_choice in
            a)
                python training_log_analyzer.py --log-dir runs --plot-types loss_curves accuracy_curves lr_schedule dashboard
                ;;
            b)
                python training_log_analyzer.py --training-log training_fixed.log --plot-types loss_curves accuracy_curves dashboard
                ;;
            c)
                python training_log_analyzer.py --log-dir runs --training-log training_fixed.log --plot-types all
                ;;
            *)
                echo "Invalid choice. Using default TensorBoard analysis..."
                python training_log_analyzer.py --log-dir runs --plot-types loss_curves accuracy_curves lr_schedule
                ;;
        esac
        ;;
    5)
        echo "Running baseline comparison..."
        python run_baseline_comparison.py --methods MSP ODIN Energy H-DRO --quick-mode
        ;;
    6)
        echo "Enter custom configuration:"
        read -p "Analysis type (ablation/efficiency/both/training_logs/baseline): " analysis_type
        
        case $analysis_type in
            training_logs)
                read -p "Log directory (default: runs): " log_dir
                read -p "Training log file (optional): " training_log
                log_dir=${log_dir:-runs}
                cmd="python training_log_analyzer.py --log-dir $log_dir"
                if [ ! -z "$training_log" ]; then
                    cmd="$cmd --training-log $training_log"
                fi
                cmd="$cmd --plot-types all"
                echo "Running custom training log analysis..."
                eval $cmd
                ;;
            baseline)
                read -p "Methods (space-separated, default: MSP ODIN Energy H-DRO): " methods
                methods=${methods:-"MSP ODIN Energy H-DRO"}
                echo "Running custom baseline comparison..."
                python run_baseline_comparison.py --methods $methods --quick-mode
                ;;
            *)
                read -p "Dataset (cifar10/cifar100): " dataset
                read -p "Backbone (resnet18/resnet50/vit_base_patch16_224): " backbone
                read -p "Use pretrained? (y/n): " pretrained
                
                if [ "$pretrained" = "y" ]; then
                    pretrained_flag="--pretrained"
                else
                    pretrained_flag=""
                fi
                
                echo "Running custom analysis..."
                python run_comprehensive_analysis.py --dataset $dataset --backbone $backbone --analysis-type $analysis_type $pretrained_flag --quick-mode
                ;;
        esac
        ;;
    7)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        ;;
esac

echo ""
echo "✅ Analysis completed! Check the output directory for results." 