import os
import random
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from feature_train import HierarchicalDROWithMultiScoring


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


class HDROModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.h_psi = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        self.g_phi = backbone.fc
        # Add head attribute for compatibility with FastVirtualGenerator
        self.head = backbone.fc

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.h_psi(x)
        z = torch.flatten(z, 1)
        logits = self.g_phi(z)
        return logits, z


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_backbone(cifar_adjust: bool = True) -> nn.Module:
    backbone = torchvision.models.resnet18(pretrained=False)
    if cifar_adjust:
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Linear(512, 10)
    return backbone


def create_dataloaders(dataset_root: str, batch_size: int, num_workers: int = 2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=dataset_root, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=dataset_root, train=False, download=True, transform=transform_test
    )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)
    return trainloader, testloader


def evaluate_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / max(1, total)


def train_hdro(
    dataset_root: str = "/home/tanmoy/research/data",
    epochs: int = 50,
    batch_size: int = 128,
    lr_backbone: float = 1e-3,
    lr_hyper: float = 1e-2,
    pixel_radius: float = 0.03,
    feature_radius: float = 0.1,
    log_dir: str = "runs/hdro_train",
    save_dir: str = "checkpoints",
    num_workers: int = 2,
    seed: int = 42,
    device_str: Optional[str] = None,
):
    set_seed(seed)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

    backbone = build_backbone(cifar_adjust=True)
    model = HDROModel(backbone).to(device)

    hdro_trainer = HierarchicalDROWithMultiScoring(
        model=model,
        num_classes=10,
        device=str(device),
        pixel_radius=pixel_radius,
        feature_radius=feature_radius,
    )

    # Data
    trainloader, testloader = create_dataloaders(dataset_root, batch_size, num_workers)

    # Delegate training to built-in trainer (handles its own optimizers, writer, scheduler)
    hdro_trainer.train_hierarchical_dro(
        train_loader=trainloader,
        num_epochs=epochs,
        lr=lr_backbone,
        val_loader=testloader,
        config=None,
        checkpoint_dir=save_dir,
    )

    # Save final model checkpoint
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"hdro_final.pt")
    torch.save({
        "model_state": model.state_dict(),
    }, ckpt_path)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train HDRO-aware CIFAR-10 model")
    parser.add_argument("--data-root", type=str, default="/home/tanmoy/research/data", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-backbone", type=float, default=1e-3)
    parser.add_argument("--lr-hyper", type=float, default=1e-2)
    parser.add_argument("--pixel-radius", type=float, default=0.03)
    parser.add_argument("--feature-radius", type=float, default=0.1)
    parser.add_argument("--log-dir", type=str, default="runs/hdro_train")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    resolved_device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    train_hdro(
        dataset_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_hyper=args.lr_hyper,
        pixel_radius=args.pixel_radius,
        feature_radius=args.feature_radius,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
        seed=args.seed,
        device_str=resolved_device,
    )


