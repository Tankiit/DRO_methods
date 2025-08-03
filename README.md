# Distributionally Robust Optimization for Out-of-Distribution Detection

This repository contains state-of-the-art implementations of Distributionally Robust Optimization (DRO) methods for Out-of-Distribution (OOD) detection, with comprehensive performance improvements and benchmarking.

## 🚀 Latest Addition: Hierarchical DRO with Virtual Outlier Exposure

Our latest implementation introduces a hierarchical DRO approach for virtual outlier exposure in deep learning models, focusing on Vision Transformers (ViT) and including both pixel-level and feature-level robustness.

### Key Features of Hierarchical DRO

- Hierarchical DRO with pixel and feature level robustness
- Support for Vision Transformers (ViT) and ResNet architectures
- Mixed precision training
- Gradient checkpointing for memory efficiency
- Multi-level uncertainty estimation
- Comprehensive OOD detection evaluation

### Usage of Hierarchical DRO

Train a model with hierarchical DRO:

```bash
python feature_train.py \
    --dataset imagenet \
    --backbone dino_vitb16 \
    --ood cifar100 \
    --batch 32 \
    --mixed-precision \
    --gradient-accumulation 4 \
    --grad-checkpoint \
    --data-dir /path/to/data
```

### Arguments

- `--dataset`: Training dataset (imagenet, cifar100)
- `--backbone`: Model architecture (resnet18, resnet50, vit_base_patch16_224, dino_vitb16)
- `--ood`: OOD dataset for validation
- `--batch`: Batch size
- `--mixed-precision`: Enable mixed precision training
- `--gradient-accumulation`: Number of gradient accumulation steps
- `--grad-checkpoint`: Enable gradient checkpointing
- `--data-dir`: Path to data directory

## 📁 Repository Structure

### Core Implementations

1. **`feature_train.py`** - **★ LATEST IMPLEMENTATION ★**
   - Hierarchical DRO with Virtual Outlier Exposure
   - Vision Transformer support
   - Multi-level uncertainty estimation
   - Expected performance: **AUROC > 0.90**

2. **`wasserstein_dro_cifar.py`**
   - Wasserstein DRO for CIFAR10 vs SVHN benchmark
   - Efficient `vmap`-based per-sample gradients
   - Multiple OOD detection methods
   - Expected performance: **AUROC > 0.85**

3. **`targeted_ood_improvement.py`**
   - Targeted performance improvements
   - Challenging synthetic data generation
   - Energy + Mahalanobis ensemble scoring
   - Expected performance: **AUROC > 0.90**

4. **`improved_weight_watchers_dro.py`**
   - Enhanced Weight Watchers DRO with spectral analysis
   - Advanced DRO loss formulations
   - Expected performance: **AUROC > 0.85**

### Additional Modules

- **`skwdro/`** - Core DRO library implementation
- **`ablation_analysis.py`** - Comprehensive ablation studies
- **`training_log_analyzer.py`** - Training analysis tools
- **`multi_scoring.py`** - Multiple OOD scoring methods
- **CIFAR experiments** - Complete CIFAR10/100 evaluation suite

## 🏆 Performance Comparison

| Method | Data | AUROC | Key Features |
|--------|------|-------|--------------|
| **Hierarchical DRO** | ImageNet/CIFAR100 | **~0.90+** | ViT, multi-level robustness |
| **Wasserstein DRO** | CIFAR10/SVHN | **~0.85+** | Real benchmark, vmap gradients |
| **Targeted Improvement** | Challenging synthetic | **~0.90+** | Multi-method ensemble |
| **Improved Weight Watchers** | Complex synthetic | **~0.85+** | Spectral analysis |

## 🔧 Installation

```bash
# Create a conda environment
conda create -n dro python=3.12
conda activate dro

# Install dependencies
pip install torch torchvision timm tqdm tensorboard numpy matplotlib scikit-learn scipy

# Optional for weight analysis:
pip install weightwatcher

# Install the project
pip install -e .
```

## 📊 Key Innovations

### 1. Hierarchical DRO
- Multi-level robustness (pixel and feature)
- Vision Transformer optimization
- Memory-efficient training
- Comprehensive uncertainty estimation

### 2. Multiple OOD Scoring Methods
- **Energy**: `-T * log(sum(exp(logits/T)))`
- **Mahalanobis**: Distance to training distribution
- **Feature-level**: Hierarchical feature analysis
- **Ensemble**: Weighted combination of methods

### 3. Real-World Evaluation
- ImageNet, CIFAR100, SVHN benchmarks
- Comprehensive metrics
- Visual analysis and monitoring
- Statistical significance testing

## 📈 Visualization

All implementations include comprehensive visualization:
- Decision boundaries
- Score distributions
- ROC and PR curves
- Training dynamics
- Performance comparisons

## 🔬 Theoretical Foundation

### Hierarchical DRO
- Multi-level uncertainty sets
- Pixel and feature space robustness
- Adaptive perturbation scaling
- Cross-level consistency

### Wasserstein DRO
- Uncertainty set optimization
- Dual formulation
- Per-sample gradients
- Worst-case perturbations

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd <repository-name>
```

2. **Set up environment**
```bash
pip install -r requirements.txt
```

3. **Run basic experiments**
```bash
# CIFAR10 experiment
bash run_cifar_evaluation.sh

# Hierarchical DRO training
python feature_train.py --dataset cifar100 --backbone resnet18
```

4. **Analyze results**
```bash
python training_log_analyzer.py --log-dir logs/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.