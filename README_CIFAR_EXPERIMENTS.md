# CIFAR ResNet Experiments with Hierarchical DRO

This collection of scripts provides a comprehensive framework for training and evaluating ResNet models on CIFAR datasets with hierarchical Distributionally Robust Optimization (DRO) and multi-scoring OOD detection.

## 🎯 Overview

The framework includes:

1. **Core Experiment Script** (`cifar_resnet_experiments.py`) - Main experiment runner
2. **Batch Runner** (`run_cifar_experiments.py`) - Run multiple experiments automatically
3. **Single Experiment Runner** (`run_single_cifar.py`) - Simple interface for individual experiments
4. **Feature Training Module** (`feature_train.py`) - Contains hierarchical DRO implementation
5. **Multi-scoring Module** (`multi_scoring.py`) - OOD detection methods

## 🚀 Quick Start

### 1. Run a Single Experiment

```bash
# Basic CIFAR-10 experiment with ResNet-18
python run_single_cifar.py --dataset cifar10 --model resnet18 --epochs 50

# CIFAR-100 with ResNet-50 and hierarchical DRO
python run_single_cifar.py --dataset cifar100 --model resnet50 --epochs 100

# Standard training (no hierarchical DRO)
python run_single_cifar.py --dataset cifar10 --model resnet18 --no-hierarchical-dro

# Use pretrained weights
python run_single_cifar.py --dataset cifar10 --model resnet50 --pretrained
```

### 2. Run Batch Experiments

```bash
# Quick comparison (50 epochs, key configurations)
python run_cifar_experiments.py --mode quick

# Full comparison (100 epochs, all configurations)
python run_cifar_experiments.py --mode full

# Ablation study focusing on hierarchical DRO
python run_cifar_experiments.py --mode ablation

# Custom configuration
python run_cifar_experiments.py --mode custom --datasets cifar10 --models resnet18 resnet50 --epochs 200
```

### 3. Evaluation Only

```bash
# Evaluate a pre-trained model
python run_single_cifar.py --eval-only --model-path ./checkpoints/model.pt --dataset cifar10 --model resnet18
```

## 📋 Detailed Usage

### Core Experiment Script (`cifar_resnet_experiments.py`)

This is the main experiment script that handles training and evaluation:

```bash
python cifar_resnet_experiments.py [OPTIONS]
```

**Key Arguments:**
- `--dataset`: Choose between `cifar10` or `cifar100`
- `--model`: ResNet variant (`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.1)
- `--batch-size`: Batch size (default: 128)
- `--no-hierarchical-dro`: Disable hierarchical DRO (use standard training)
- `--pretrained`: Use pretrained ImageNet weights
- `--data-dir`: Directory containing datasets (default: `./data`)
- `--output-dir`: Directory for saving results (default: `./cifar_results`)

**Example:**
```bash
python cifar_resnet_experiments.py \
    --dataset cifar10 \
    --model resnet50 \
    --epochs 100 \
    --lr 0.1 \
    --batch-size 128 \
    --data-dir /path/to/data \
    --output-dir ./my_results
```

### Batch Experiment Runner (`run_cifar_experiments.py`)

Run multiple experiments with different configurations automatically:

**Modes:**
- `quick`: Fast comparison with key configurations (50 epochs)
- `full`: Comprehensive comparison with all configurations (100 epochs)  
- `ablation`: Focused study on hierarchical DRO effectiveness
- `custom`: User-defined configurations

**Examples:**
```bash
# Quick comparison
python run_cifar_experiments.py --mode quick --data-dir ./data

# Full comparison with custom output directory
python run_cifar_experiments.py --mode full --output-dir ./comprehensive_results

# Custom experiment grid
python run_cifar_experiments.py --mode custom \
    --datasets cifar10 cifar100 \
    --models resnet18 resnet50 \
    --epochs 200

# Analyze existing results without running experiments
python run_cifar_experiments.py --analyze-only --output-dir ./existing_results
```

### Single Experiment Runner (`run_single_cifar.py`)

Simplified interface for running individual experiments:

```bash
# Basic usage
python run_single_cifar.py --dataset cifar10 --model resnet18

# With custom parameters
python run_single_cifar.py \
    --dataset cifar100 \
    --model resnet50 \
    --epochs 200 \
    --lr 0.01 \
    --batch-size 64 \
    --pretrained
```

## 📊 Output Structure

Each experiment creates a comprehensive output structure:

```
output_directory/
├── checkpoints/           # Model checkpoints
│   ├── best_model_*.pt
│   └── final_model_*.pt
├── results/              # Evaluation results
│   ├── ood_results_*.csv
│   ├── ood_results_*.xlsx
│   └── ood_results_*.json
├── intermediate_scores/   # Individual score files
│   ├── scores_*.pkl
│   └── ...
└── plots/                # Visualizations
    ├── auroc_heatmap_*.png
    ├── fpr95_heatmap_*.png
    ├── method_comparison_*.png
    └── training_history_*.png
```

## 🎛️ Configuration Options

### Training Configuration

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--dataset` | Dataset to use | `cifar10` | `cifar10`, `cifar100` |
| `--model` | ResNet architecture | `resnet18` | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` |
| `--epochs` | Training epochs | `100` | Any positive integer |
| `--lr` | Learning rate | `0.1` | Any positive float |
| `--batch-size` | Batch size | `128` | Any positive integer |
| `--pretrained` | Use pretrained weights | `False` | Flag |
| `--no-hierarchical-dro` | Disable hierarchical DRO | `False` | Flag |

### Evaluation Configuration

The framework evaluates multiple OOD detection methods:

- **Energy**: Energy-based OOD detection
- **Mahalanobis**: Mahalanobis distance in feature space
- **MSP**: Maximum Softmax Probability
- **ODIN**: Out-of-DIstribution detector for Neural networks
- **Ensemble**: Weighted combination of multiple methods

### OOD Datasets

For CIFAR-10 training, OOD evaluation includes:
- CIFAR-100
- SVHN
- Textures (DTD)

For CIFAR-100 training, OOD evaluation includes:
- CIFAR-10
- SVHN
- Textures (DTD)

## 📈 Results Analysis

### Saved Metrics

Each experiment saves comprehensive metrics:

- **AUROC**: Area Under ROC Curve
- **FPR@95**: False Positive Rate at 95% True Positive Rate  
- **AUPR**: Area Under Precision-Recall Curve
- **Detection Error**: Minimum classification error
- **Score Statistics**: Mean, std, separation for ID/OOD scores

### Visualizations

Automatic generation of:
- AUROC heatmaps by method and OOD dataset
- FPR@95 comparison charts
- Method performance bar plots
- Training history plots (loss, accuracy, learning rate)

### Example Results Access

```python
import pandas as pd
import pickle

# Load detailed results
df = pd.read_csv('results/ood_results_cifar10_resnet18_20241201_120000.csv')

# Load individual scores for analysis
with open('intermediate_scores/scores_cifar10_resnet18_energy_cifar100.pkl', 'rb') as f:
    scores_data = pickle.load(f)
    id_scores = scores_data['id_scores']
    ood_scores = scores_data['ood_scores']
```

## 🔧 Advanced Usage

### Custom Data Directory Structure

Ensure your data directory has the following structure:
```
data/
├── cifar-10-batches-py/    # CIFAR-10 (downloaded automatically)
├── cifar-100-python/       # CIFAR-100 (downloaded automatically)
├── SVHN/                   # SVHN (downloaded automatically)
└── DTD/                    # Describable Textures Dataset (downloaded automatically)
```

### Memory Optimization

For large-scale experiments or limited GPU memory:

```bash
# Reduce batch size
python run_single_cifar.py --batch-size 64

# Use gradient checkpointing (handled automatically for larger models)
python run_single_cifar.py --model resnet152
```

### Continuing Interrupted Experiments

The framework automatically saves intermediate results and checkpoints. To continue:

1. Check the `checkpoints/` directory for saved models
2. Use `--eval-only` mode with the checkpoint path for evaluation only

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 64` or `--batch-size 32`
   - Use smaller model: `--model resnet18`

2. **Dataset Download Issues**
   - Ensure internet connection for automatic downloads
   - Check disk space (datasets require ~500MB total)

3. **Long Training Times**
   - Use fewer epochs for testing: `--epochs 50`
   - Try quick mode: `python run_cifar_experiments.py --mode quick`

### Debug Mode

For debugging, use the single experiment runner with verbose output:

```bash
python run_single_cifar.py --dataset cifar10 --model resnet18 --epochs 5
```

## 📝 Examples

### Example 1: Quick CIFAR-10 vs CIFAR-100 Comparison

```bash
# Train on CIFAR-10, evaluate against CIFAR-100
python run_single_cifar.py --dataset cifar10 --model resnet18 --epochs 50

# Train on CIFAR-100, evaluate against CIFAR-10  
python run_single_cifar.py --dataset cifar100 --model resnet18 --epochs 50
```

### Example 2: Hierarchical DRO Ablation Study

```bash
# Standard training
python run_single_cifar.py --dataset cifar10 --model resnet50 --no-hierarchical-dro --epochs 100

# Hierarchical DRO training
python run_single_cifar.py --dataset cifar10 --model resnet50 --epochs 100
```

### Example 3: Model Size Comparison

```bash
# Small model
python run_single_cifar.py --dataset cifar10 --model resnet18 --epochs 100

# Medium model  
python run_single_cifar.py --dataset cifar10 --model resnet34 --epochs 100

# Large model
python run_single_cifar.py --dataset cifar10 --model resnet50 --epochs 100
```

### Example 4: Batch Processing for Publication

```bash
# Run comprehensive experiments for paper
python run_cifar_experiments.py --mode full --output-dir ./paper_results

# Generate analysis plots
python run_cifar_experiments.py --analyze-only --output-dir ./paper_results
```

## 🤝 Contributing

To extend the framework:

1. **Add new OOD detection methods**: Modify `multi_scoring.py`
2. **Add new datasets**: Extend the dataset loading in `cifar_resnet_experiments.py`
3. **Add new architectures**: Update model choices in argument parsers
4. **Customize training**: Modify the training loop in `CIFARResNetExperiment.train_model()`

## 📄 File Dependencies

- `feature_train.py`: Contains `HierarchicalDROWithMultiScoring`, `TimmFeatureExtractor`
- `multi_scoring.py`: Contains `MultiScoreOODDetector`
- Standard PyTorch, timm, torchvision, scikit-learn, pandas, matplotlib

## 🎯 Expected Runtime

Approximate training times on modern GPU (RTX 3090):

| Dataset | Model | Epochs | Standard Training | Hierarchical DRO |
|---------|-------|--------|------------------|------------------|
| CIFAR-10 | ResNet-18 | 100 | ~30 minutes | ~45 minutes |
| CIFAR-10 | ResNet-50 | 100 | ~45 minutes | ~70 minutes |
| CIFAR-100 | ResNet-18 | 100 | ~30 minutes | ~45 minutes |
| CIFAR-100 | ResNet-50 | 100 | ~45 minutes | ~70 minutes |

Evaluation time: ~5-10 minutes per experiment depending on OOD datasets.

---

For questions or issues, please check the troubleshooting section or examine the detailed error messages in the output logs. 