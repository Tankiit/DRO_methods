# 🔬 Comprehensive CIFAR OOD Detection Analysis

This directory contains a complete analysis framework for studying CIFAR-10/100 out-of-distribution (OOD) detection using hierarchical Distributionally Robust Optimization (DRO) with virtual samples.

## 🎯 Features

### 📊 **Ablation Studies**
- **Virtual Sample Strategy Analysis** - Compare different virtual sample generation approaches
- **Progressive Training Analysis** - Study different training schedules and convergence patterns  
- **Hyperparameter Sensitivity** - Systematic analysis of key parameters
- **Architecture Generalization** - Performance across different model architectures

### ⚡ **Efficiency Analysis**
- **Memory Usage Analysis** - Training and inference memory requirements
- **Training Time Breakdown** - Component-wise timing analysis
- **Scalability Analysis** - Performance vs dataset size
- **Baseline Comparisons** - Against standard OOD detection methods

---

## 🚀 Quick Start

### 1. **Set Up Virtual Environment**

First, ensure your `torch-multimodal` conda environment has all required packages:

```bash
# Run the setup script (one-time setup)
./setup_venv_analysis.sh
```

This script will:
- ✅ Check/create the `torch-multimodal` conda environment
- ✅ Install PyTorch, torchvision, and other ML packages
- ✅ Install analysis dependencies (pandas, matplotlib, seaborn, etc.)
- ✅ Verify all installations

### 2. **Run Analysis with Virtual Environment**

Use the environment-aware runner:

```bash
# Interactive mode (recommended for first-time users)
./run_with_venv.sh

# Or with specific parameters
./run_with_venv.sh --dataset cifar10 --backbone resnet18 --pretrained --analysis-type both --quick-mode
```

### 3. **Alternative: Manual Environment Activation**

```bash
# Activate environment manually
conda activate torch-multimodal

# Run analysis directly
python run_comprehensive_analysis.py --dataset cifar10 --backbone resnet18 --pretrained --analysis-type both
```

---

## 📁 File Structure

```
Virtual_Outlier_Exposure/
├── 🔬 Analysis Framework
│   ├── ablation_analysis.py              # Core analysis classes (1,744 lines)
│   ├── run_comprehensive_analysis.py     # Main runner script
│   └── multi_scoring.py                  # OOD detection methods
│
├── 🚀 Virtual Environment Support
│   ├── setup_venv_analysis.sh           # One-time environment setup
│   ├── run_with_venv.sh                 # Environment-aware runner
│   └── requirements_analysis.txt        # Package dependencies
│
├── 📊 Evaluation Scripts
│   ├── evaluate_cifar.py                # CIFAR-specific evaluation
│   ├── run_cifar_evaluation.sh          # CIFAR evaluation wrapper
│   └── run_analysis_examples.sh         # Interactive examples
│
├── 🏗️ Core Framework
│   ├── feature_train.py                 # Main training/evaluation code
│   └── README_COMPREHENSIVE_ANALYSIS.md # This file
```

---

## 🔧 Usage Examples

### **Quick Ablation Study**
```bash
./run_with_venv.sh --dataset cifar10 --backbone resnet18 --pretrained --analysis-type ablation --quick-mode
```

### **Memory & Efficiency Analysis**
```bash
./run_with_venv.sh --dataset cifar10 --backbone resnet18 --pretrained --analysis-type efficiency
```

### **Complete Analysis (Both)**
```bash
./run_with_venv.sh --dataset cifar10 --backbone resnet18 --pretrained --analysis-type both
```

### **Architecture Comparison**
```bash
# Compare different architectures
./run_with_venv.sh --dataset cifar10 --backbone resnet18 --pretrained --output-dir results_resnet18
./run_with_venv.sh --dataset cifar10 --backbone resnet50 --pretrained --output-dir results_resnet50
./run_with_venv.sh --dataset cifar10 --backbone vit_base_patch16_224 --pretrained --output-dir results_vit
```

### **Dataset Comparison**
```bash
# Compare CIFAR-10 vs CIFAR-100
./run_with_venv.sh --dataset cifar10 --backbone resnet18 --pretrained --output-dir results_cifar10
./run_with_venv.sh --dataset cifar100 --backbone resnet18 --pretrained --output-dir results_cifar100
```

---

## 📊 Generated Outputs

### **Directory Structure**
```
comprehensive_analysis_results/
├── ablation_study/
│   ├── plots/
│   │   ├── virtual_strategy_comparison.png
│   │   ├── progressive_training_analysis.png
│   │   ├── hyperparameter_sensitivity.png
│   │   └── architecture_analysis.png
│   ├── tables/
│   │   ├── virtual_strategy_table.tex        # LaTeX table
│   │   ├── progressive_training_table.tex    # LaTeX table
│   │   ├── architecture_table.tex            # LaTeX table
│   │   └── sensitivity_summary_table.tex     # LaTeX table
│   ├── logs/
│   │   └── ablation_YYYYMMDD_HHMMSS.log
│   └── complete_ablation_results.json
├── efficiency_analysis/
│   ├── plots/
│   │   ├── memory_analysis.png
│   │   └── timing_analysis.png
│   ├── tables/
│   │   ├── memory_table.tex
│   │   ├── batch_efficiency.csv
│   │   └── baseline_comparison.csv
│   ├── logs/
│   │   └── efficiency_YYYYMMDD_HHMMSS.log
│   └── complete_efficiency_results.json
└── combined_results_summary.json
```

### **📈 Plots Generated**
- **Virtual Strategy Comparison** - AUROC and FPR@95% for different approaches
- **Progressive Training Analysis** - Convergence curves and final performance
- **Hyperparameter Sensitivity** - Parameter sweep results with stability analysis
- **Architecture Analysis** - Performance vs complexity trade-offs
- **Memory Usage Curves** - Training/inference memory vs batch size
- **Training Time Breakdown** - Component-wise timing analysis

### **📋 LaTeX Tables (Ready for Papers)**
- Virtual sample strategy comparison
- Progressive training schedule results
- Architecture generalization analysis
- Hyperparameter sensitivity summary
- Memory usage analysis
- Efficiency comparison table

---

## 🔍 Analysis Components

### 📊 **Ablation Study Details**

#### **1. Virtual Sample Strategy Analysis**
- `no_virtual` - ERM baseline (no virtual samples)
- `pixel_only` - Pixel-level virtual samples only
- `feature_only` - Feature-level virtual samples only  
- `combined_basic` - Simple combination approach
- `combined_full` - Full hierarchical approach

#### **2. Progressive Training Analysis**
- `standard_erm` - Standard ERM training
- `pixel_only` - Pixel-only progressive schedule
- `progressive_full` - Full progressive training
- `simultaneous` - Multi-level simultaneous training

#### **3. Hyperparameter Sensitivity**
- **Pixel radius** - [0.01, 0.03, 0.05, 0.1, 0.15]
- **Feature radius** - [0.05, 0.1, 0.15, 0.2, 0.3]
- **Loss weights** - [0.1, 0.3, 0.5, 0.7, 0.9]
- **Learning rate** - [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

#### **4. Architecture Generalization**
- **ResNet-18/50** - CNN architectures
- **ViT-Base** - Vision Transformer
- **EfficientNet-B0** - Efficient scaling

### ⚡ **Efficiency Analysis Details**

#### **1. Memory Usage Analysis**
- Training memory vs batch size [4, 8, 16, 32, 64, 128]
- Inference memory requirements
- Memory efficiency (MB per sample)

#### **2. Training Time Breakdown**
- Data loading time
- Forward pass time
- Virtual sample generation time
- Backward pass time
- Optimizer step time

#### **3. Scalability Analysis**
- Training time vs dataset size [100, 200, 500, 1000, 2000]
- Throughput analysis (samples/second)
- Resource utilization trends

#### **4. Baseline Comparisons**
- **H-DRO** (Our method)
- **ERM** (Standard training)
- **Outlier Exposure**
- **Virtual Outlier Synthesis**
- **Energy-based methods**

---

## 🐍 Python API Usage

```python
from ablation_analysis import run_ablation_study, run_efficiency_analysis

# Configuration
config = {
    'dataset': 'cifar10',
    'backbone': 'resnet18', 
    'pretrained': True,
    'data_dir': '/home/tanmoy/research/data',
    'num_classes': 10,
    'batch_size': 32,
    'lr': 1e-3,
    'num_epochs': 5
}

# Run analyses
ablation_results = run_ablation_study(config)
efficiency_results = run_efficiency_analysis(config)

# Access results
best_strategy = max(ablation_results['virtual_strategies'].items(), 
                   key=lambda x: x[1]['evaluation_metrics']['avg_auroc'])
print(f"Best strategy: {best_strategy[0]}")
```

---

## 🛠️ Options and Configuration

### **Command Line Arguments**
```bash
python run_comprehensive_analysis.py --help

Options:
  --dataset           : cifar10, cifar100
  --backbone          : resnet18, resnet50, vit_base_patch16_224, efficientnet_b0
  --analysis-type     : ablation, efficiency, both
  --pretrained        : Use pretrained weights (recommended)
  --quick-mode        : Fast mode with reduced parameters
  --epochs           : Number of training epochs (default: 5)
  --batch-size       : Batch size (default: 32)
  --lr               : Learning rate (default: 1e-3)
  --data-dir         : Data directory (default: /home/tanmoy/research/data)
  --output-dir       : Output directory (default: comprehensive_analysis_results)
```

### **Environment Variables**
```bash
# Optional: Set data directory
export DATA_DIR="/path/to/your/data"

# Optional: Set output directory  
export OUTPUT_DIR="/path/to/results"
```

---

## 🐛 Troubleshooting

### **Environment Issues**
```bash
# If environment activation fails
conda env remove -n torch-multimodal
./setup_venv_analysis.sh

# If packages are missing
conda activate torch-multimodal
pip install -r requirements_analysis.txt
```

### **Memory Issues**
```bash
# Use smaller batch sizes
./run_with_venv.sh --batch-size 16 --quick-mode

# Enable CPU-only mode if GPU memory is insufficient
CUDA_VISIBLE_DEVICES="" ./run_with_venv.sh
```

### **Data Directory Issues**
```bash
# Specify custom data directory
./run_with_venv.sh --data-dir /path/to/your/data
```

---

## 📚 References

This analysis framework supports the research in:

- **Hierarchical Distributionally Robust Optimization** for OOD detection
- **Virtual Sample Generation** strategies for robustness
- **Multi-level OOD Detection** approaches
- **Comprehensive Evaluation** of detection methods

For more details, see the paper and related documentation.

---

## 🤝 Contributing

To extend the analysis framework:

1. **Add new virtual sample strategies** in `ablation_analysis.py`
2. **Implement new efficiency metrics** in the `EfficiencyAnalyzer` class
3. **Add new architectures** in `feature_train.py`
4. **Extend visualization functions** for new analysis types

---

## 📄 License

This project is part of the CIFAR OOD Detection research framework.

---

**Happy Analyzing! 🔬✨** 