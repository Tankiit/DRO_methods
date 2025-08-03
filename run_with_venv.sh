#!/bin/bash

# Comprehensive Analysis Runner with Virtual Environment Support
# This script ensures the analysis runs in the correct conda environment

echo "🔬 CIFAR OOD Detection - Comprehensive Analysis (Virtual Environment)"
echo "====================================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not available. Please install conda first."
    exit 1
fi

# Activate the torch-multimodal environment
echo "🔄 Activating torch-multimodal environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch-multimodal

# Verify environment is active
if [[ "$CONDA_DEFAULT_ENV" != "torch-multimodal" ]]; then
    echo "❌ Error: Failed to activate torch-multimodal environment"
    echo "Current environment: $CONDA_DEFAULT_ENV"
    exit 1
fi

echo "✅ Virtual environment activated: $CONDA_DEFAULT_ENV"
echo ""

# Check Python and required packages
echo "🔍 Checking Python and package availability..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'❌ PyTorch not available: {e}')
    sys.exit(1)

try:
    import torchvision
    print(f'Torchvision version: {torchvision.__version__}')
except ImportError as e:
    print(f'❌ Torchvision not available: {e}')
    sys.exit(1)

try:
    import timm
    print(f'TIMM version: {timm.__version__}')
except ImportError as e:
    print(f'❌ TIMM not available: {e}')
    sys.exit(1)

try:
    import pandas as pd
    print(f'Pandas version: {pd.__version__}')
except ImportError as e:
    print(f'❌ Pandas not available: {e}')
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print(f'Matplotlib available')
except ImportError as e:
    print(f'❌ Matplotlib not available: {e}')
    sys.exit(1)

try:
    import seaborn as sns
    print(f'Seaborn available')
except ImportError as e:
    print(f'❌ Seaborn not available: {e}')
    sys.exit(1)

print('✅ All required packages available!')
"

# Check if the check was successful
if [ $? -ne 0 ]; then
    echo "❌ Package check failed. Please install missing packages."
    exit 1
fi

echo ""
echo "🚀 Ready to run comprehensive analysis!"
echo ""

# Parse command line arguments or run interactively
if [ $# -eq 0 ]; then
    # Interactive mode
    echo "Choose analysis type:"
    echo "1) Quick ablation study (CIFAR-10 + ResNet-18)"
    echo "2) Efficiency analysis (CIFAR-10 + ResNet-18)"  
    echo "3) Complete analysis (both ablation + efficiency)"
    echo "4) Custom configuration"
    echo "5) Test basic functionality"
    echo "6) Exit"
    echo ""
    
    read -p "Enter your choice (1-6): " choice
    
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
            echo "Enter custom configuration:"
            read -p "Dataset (cifar10/cifar100): " dataset
            read -p "Backbone (resnet18/resnet50/vit_base_patch16_224): " backbone
            read -p "Analysis type (ablation/efficiency/both): " analysis_type
            read -p "Use pretrained? (y/n): " pretrained
            
            if [ "$pretrained" = "y" ]; then
                pretrained_flag="--pretrained"
            else
                pretrained_flag=""
            fi
            
            echo "Running custom analysis..."
            python run_comprehensive_analysis.py --dataset $dataset --backbone $backbone --analysis-type $analysis_type $pretrained_flag --quick-mode
            ;;
        5)
            echo "Testing basic functionality..."
            python -c "
from ablation_analysis import AblationStudyRunner, EfficiencyAnalyzer
from feature_train import get_model
import torch

print('✅ Imports successful')

# Test basic model creation
try:
    class MockArgs:
        def __init__(self):
            self.dataset = 'cifar10'
            self.backbone = 'resnet18'
            self.pretrained = True
            self.grad_checkpoint = False
    
    args = MockArgs()
    model = get_model(args)
    print('✅ Model creation successful')
    
    # Test basic forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    print('✅ Model forward pass successful')
    print(f'   Output shape: {output[0].shape if isinstance(output, tuple) else output.shape}')
    
    print('🎉 All basic functionality tests passed!')
    
except Exception as e:
    print(f'❌ Test failed: {e}')
    import traceback
    traceback.print_exc()
"
            ;;
        6)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice. Please run the script again."
            exit 1
            ;;
    esac
else
    # Pass all arguments to the analysis script
    echo "Running with provided arguments: $@"
    python run_comprehensive_analysis.py "$@"
fi

echo ""
echo "✅ Analysis completed!"
echo "📁 Results should be in the output directory"
echo "🔍 Check the generated plots, tables, and JSON files" 