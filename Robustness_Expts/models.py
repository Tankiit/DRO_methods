# ============================================================================
# RQ1: Dataset-Agnostic Efficient DRO Framework
# ============================================================================

import argparse
import os
from typing import Dict, Any, Optional

try:
    # `timm` greatly expands the list of available models. It is optional – we only
    # rely on it for architectures that are not shipped with torchvision.
    import timm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – harmless when `timm` is absent
    timm = None  # type: ignore

try:
    import open_clip  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    open_clip = None  # type: ignore

import torch
import torchvision.models as tv_models


class DatasetConfig:
    """Configuration for different datasets"""
    
    SUPPORTED_DATASETS = {
        'cifar10': {
            'num_classes': 10,
            'input_shape': (3, 32, 32),
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2023, 0.1994, 0.2010),
            'ood_datasets': ['cifar100', 'svhn', 'textures']
        },
        'cifar100': {
            'num_classes': 100,
            'input_shape': (3, 32, 32),
            'mean': (0.5071, 0.4867, 0.4408),
            'std': (0.2675, 0.2565, 0.2761),
            'ood_datasets': ['cifar10', 'svhn', 'places365']
        },
        'imagenet': {
            'num_classes': 1000,
            'input_shape': (3, 224, 224),
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'ood_datasets': ['places365', 'sun', 'inaturalist']
        }
    }
    
    @classmethod
    def get_config(cls, dataset_name):
        """Get configuration for specific dataset"""
        return cls.SUPPORTED_DATASETS.get(dataset_name.lower())

class ModelConfig:
    """Configuration for different model architectures"""
    
    SUPPORTED_MODELS = {
        'resnet18': {
            'architecture': 'resnet',
            'depth': 18,
            'feature_dim': 512,
            'efficiency_profile': 'fast'
        },
        'wideresnet28': {
            'architecture': 'wideresnet',
            'depth': 28,
            'width': 10,
            'feature_dim': 640,
            'efficiency_profile': 'balanced'
        },
        'vit_base': {
            'architecture': 'vision_transformer',
            'patch_size': 16,
            'feature_dim': 768,
            'efficiency_profile': 'accurate'
        },
        'efficientnet_b0': {
            'architecture': 'efficientnet',
            'version': 'b0',
            'feature_dim': 1280,
            'efficiency_profile': 'fast'
        },
        'vit_small': {
            'architecture': 'vision_transformer',
            'patch_size': 16,
            'feature_dim': 384,
            'efficiency_profile': 'balanced'
        },
        'vit_large': {
            'architecture': 'vision_transformer',
            'patch_size': 16,
            'feature_dim': 1024,
            'efficiency_profile': 'accurate'
        },
        'openclip_vit_b32': {
            'architecture': 'openclip',
            'variant': 'vit_b32',
            'feature_dim': 512,
            'efficiency_profile': 'balanced'
        }
    }
    
    @classmethod
    def get_config(cls, model_name):
        """Get configuration for specific model"""
        return cls.SUPPORTED_MODELS.get(model_name.lower())


def build_model(model_name: str, num_classes: int) -> torch.nn.Module:  # noqa: D401
    """Return a PyTorch model instance that matches *model_name*.

    Parameters
    ----------
    model_name:
        One of the keys in :pyattr:`ModelConfig.SUPPORTED_MODELS`.
    num_classes:
        Number of output classes for the classifier head.

    Notes
    -----
    • Models shipped with torchvision are used where available.
    • For WiderResNet-28, ViT, etc. we fall back to ``timm`` if it is installed.
    • An informative ``ImportError`` is raised if a required backend is missing.
    """
    name = model_name.lower()
    if name == "resnet18":
        return tv_models.resnet18(weights=None, num_classes=num_classes)

    if name == "efficientnet_b0":
        return tv_models.efficientnet_b0(weights=None, num_classes=num_classes)

    # Architectures that are only available via the "timm" library ----------------
    if timm is None:
        raise ImportError(
            f"Model '{model_name}' requires the optional dependency 'timm'. "
            "Install it with `pip install timm` or choose a different model."
        )

    if name == "wideresnet28":
        # 28-10 matches the depth & width in the config above.
        return timm.create_model("wide_resnet28_10", pretrained=False, num_classes=num_classes)

    if name == "vit_base":
        # ViT-B/16 is the canonical base Vision Transformer.
        return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)

    if name == "vit_small":
        return timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=num_classes)

    if name == "vit_large":
        return timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=num_classes)

    if name == "openclip_vit_b32":
        if open_clip is None:
            raise ImportError(
                "Model 'openclip_vit_b32' requires the 'open_clip_torch' package. "
                "Install it with `pip install open_clip_torch`."
            )
        oc_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
        embed_dim = getattr(oc_model, "embed_dim", 512)

        class OpenCLIPClassifier(torch.nn.Module):
            def __init__(self, backbone: torch.nn.Module, dim: int, n_cls: int):
                super().__init__()
                self.backbone = backbone
                self.classifier = torch.nn.Linear(dim, n_cls)

            def forward(self, x):
                feats = self.backbone(x)
                if isinstance(feats, tuple):  # open_clip may return (features, ...)
                    feats = feats[0]
                return self.classifier(feats)

        return OpenCLIPClassifier(oc_model.visual, embed_dim, num_classes)

    # -----------------------------------------------------------------------------
    raise ValueError(f"Unsupported model: {model_name}. Update build_model() if necessary.")






# -----------------------------------------------------------------------------
# Argument parsing helpers
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return a namespace."""
    parser = argparse.ArgumentParser(
        description="Dataset-Agnostic Efficient DRO Framework (RQ1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset / model selections --------------------------------------------------
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DatasetConfig.SUPPORTED_DATASETS.keys()),
        default="cifar10",
        help="Dataset to use for training & evaluation.",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=list(ModelConfig.SUPPORTED_MODELS.keys()),
        default="resnet18",
        help="Neural network architecture.",
    )

    # General training hyper-parameters ------------------------------------------
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size.")
    parser.add_argument("--lr", "--learning-rate", type=float, default=0.1, help="Learning rate.")

    # DRO-specific options --------------------------------------------------------
    parser.add_argument(
        "--dro-method",
        type=str,
        choices=["erm", "group_dro", "irm", "dro"],
        default="erm",
        help="Distributionally-robust optimisation technique.",
    )

    # Data / workflow paths -------------------------------------------------------
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/mukher74/research/data",
        help="Root directory that contains (or will contain) the datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Where to save checkpoints, logs, etc.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main entrypoint (stub) -------------------------------------------------------
# -----------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Resolve configs ------------------------------------------------------------
    dataset_cfg: Optional[Dict[str, Any]] = DatasetConfig.get_config(args.dataset)
    model_cfg: Optional[Dict[str, Any]] = ModelConfig.get_config(args.model)

    if dataset_cfg is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    if model_cfg is None:
        raise ValueError(f"Unknown model: {args.model}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Dataset config:")
    for k, v in dataset_cfg.items():
        print(f"  {k}: {v}")

    print("\n[INFO] Model config:")
    for k, v in model_cfg.items():
        print(f"  {k}: {v}")

    print("\n[INFO] Instantiating model ...")
    model = build_model(args.model, num_classes=dataset_cfg["num_classes"])
    print(model)

    # NOTE: The rest of the training / evaluation pipeline (data loaders,
    # optimiser, DRO wrappers, etc.) is intentionally left as future work.
    # This stub provides a solid starting point while keeping the example
    # self-contained.


if __name__ == "__main__":
    main()