# Virtual Outlier Exposure with Hierarchical DRO

This repository implements a hierarchical Distributionally Robust Optimization (DRO) approach for virtual outlier exposure in deep learning models. The implementation focuses on vision transformers (ViT) and includes both pixel-level and feature-level robustness.

## Features

- Hierarchical DRO with pixel and feature level robustness
- Support for Vision Transformers (ViT) and ResNet architectures
- Mixed precision training
- Gradient checkpointing for memory efficiency
- Multi-level uncertainty estimation
- Comprehensive OOD detection evaluation

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd Virtual_Outlier_Exposure

# Create a conda environment
conda create -n voe python=3.12
conda activate voe

# Install dependencies
pip install torch torchvision timm tqdm tensorboard
```

## Usage

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

## Model Architecture

The implementation supports:
- Vision Transformers (ViT)
- ResNet models
- DINO pre-trained models

## Training Details

- Uses ViT-specific learning rate schedule
- Implements warmup and cosine annealing
- Supports gradient accumulation for large models
- Includes memory optimization techniques

## Results

The model achieves strong OOD detection performance using multiple scoring methods:
- Energy-based scoring
- Mahalanobis distance
- Maximum softmax probability
- ODIN
- Ensemble methods

## Citation

If you find this code useful for your research, please cite:

```bibtex
@misc{virtual_outlier_exposure,
  author = {[Author]},
  title = {Virtual Outlier Exposure with Hierarchical DRO},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[repository-url]}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 