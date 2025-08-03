#!/bin/bash

# Setup Script for Comprehensive Analysis in Virtual Environment
# This script ensures the torch-multimodal environment has all required packages

echo "🔧 Setting up Virtual Environment for Comprehensive Analysis"
echo "=========================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not available. Please install conda first."
    exit 1
fi

# Source conda configuration
source ~/miniconda3/etc/profile.d/conda.sh

# Check if torch-multimodal environment exists
if conda env list | grep -q "torch-multimodal"; then
    echo "✅ torch-multimodal environment found"
else
    echo "❌ torch-multimodal environment not found"
    echo "Creating torch-multimodal environment..."
    conda create -n torch-multimodal python=3.9 -y
fi

# Activate the environment
echo "🔄 Activating torch-multimodal environment..."
conda activate torch-multimodal

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" != "torch-multimodal" ]]; then
    echo "❌ Error: Failed to activate torch-multimodal environment"
    exit 1
fi

echo "✅ Environment activated: $CONDA_DEFAULT_ENV"

# Install/upgrade conda packages first
echo ""
echo "📦 Installing conda packages..."
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y pandas numpy matplotlib seaborn scipy scikit-learn -c conda-forge

# Install pip packages
echo ""
echo "📦 Installing pip packages..."
pip install --upgrade pip

# Install timm and other ML packages
pip install timm>=0.6.0
pip install tqdm psutil

# Install additional visualization packages
pip install plotly>=5.0.0

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python -c "
import sys
print(f'Python version: {sys.version}')
print('')

packages_to_check = [
    ('torch', 'PyTorch'),
    ('torchvision', 'Torchvision'), 
    ('timm', 'TIMM'),
    ('pandas', 'Pandas'),
    ('numpy', 'NumPy'),
    ('matplotlib', 'Matplotlib'),
    ('seaborn', 'Seaborn'),
    ('scipy', 'SciPy'),
    ('sklearn', 'Scikit-learn'),
    ('tqdm', 'TQDM'),
    ('psutil', 'PSUtil')
]

all_good = True
for package, name in packages_to_check:
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {name}: {version}')
    except ImportError:
        print(f'❌ {name}: Not installed')
        all_good = False

if all_good:
    print('')
    print('🎉 All packages installed successfully!')
    
    # Test PyTorch CUDA
    import torch
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠️  CUDA not available (CPU-only mode)')
else:
    print('')
    print('❌ Some packages are missing. Please check the installation.')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Virtual environment setup completed successfully!"
    echo ""
    echo "🚀 You can now run the comprehensive analysis:"
    echo "   ./run_with_venv.sh"
    echo ""
    echo "Or directly:"
    echo "   conda activate torch-multimodal"
    echo "   python run_comprehensive_analysis.py --help"
else
    echo ""
    echo "❌ Setup failed. Please check the error messages above."
    exit 1
fi 