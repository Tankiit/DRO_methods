"""margin_detection.py
Margin-based OOD Detection
==========================
Key empirical observation (Liu & Qin, ICML 2024)
------------------------------------------------
In well-trained classifiers:
• In-distribution (ID) samples tend to reside *farther* from the decision boundaries.
• Out-of-distribution (OOD) samples are typically *closer* to those boundaries.

This distance – often called the *classification margin* – thus offers a natural
signal for OOD detection.

This module implements a lightweight, dependency-free version of that idea using
logit margins. Feel free to swap the metric with more sophisticated boundary
measures (gradient-based distances, DeepFool iterations, etc.).
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import argparse
import torchvision.transforms as T
import torchvision.datasets as tv_datasets
from torch.utils.data import DataLoader, random_split

# Reuse model factory from rq1
from rq1 import build_model, DatasetConfig


__all__ = [
    "compute_boundary_distance",
    "classify_sample",
    "classify_batch",
    "estimate_margin_threshold",
    "predict_ood_mask",
]


# ---------------------------------------------------------------------------
# Core helpers ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def compute_boundary_distance(logits: torch.Tensor) -> torch.Tensor:  # noqa: D401
    """Return the *margin* between top-1 and top-2 logits for each sample.

    A larger margin → sample is farther from the decision boundary.
    """
    # Ensure 2-D shape [batch, num_classes]
    if logits.ndim != 2:
        logits = logits.view(logits.shape[0], -1)

    top2 = torch.topk(logits, k=2, dim=1).values  # shape (B, 2)
    margin = top2[:, 0] - top2[:, 1]
    return margin  # shape (B,)


def classify_sample(
    logits: torch.Tensor,
    threshold: float,
) -> Literal["ID", "OOD"]:
    """Classify a *single* sample as ID/OOD based on margin threshold."""

    margin = compute_boundary_distance(logits.unsqueeze(0))[0].item()
    return "ID" if margin > threshold else "OOD"


def classify_batch(
    logits: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorised ID/OOD classification for a batch.

    Parameters
    ----------
    logits: torch.Tensor
        Tensor of shape (batch_size, num_classes).
    threshold: float
        Margin threshold; can be chosen via validation set AUROC or held-out OOD data.

    Returns
    -------
    Tuple[tensor, tensor]
        • `is_id` – boolean tensor where `True` denotes ID prediction.
        • `margins` – the computed margins for each sample (useful for scoring).
    """
    margins = compute_boundary_distance(logits)
    is_id = margins > threshold
    return is_id, margins


def estimate_margin_threshold(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    quantile: float = 0.05,
    device: str | torch.device = "auto",
) -> float:
    """Estimate a decision-boundary margin threshold from *ID* data.

    This utility sweeps the given *val_loader* (e.g. CIFAR-10 validation split),
    collects the margin for each sample, and returns the *quantile* value.

    A small quantile (e.g. 5%) means 95% of ID samples will have margin larger
    than the threshold → low false-negative rate.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    model.eval().to(device)

    margins: list[float] = []
    with torch.no_grad():
        for x, _ in tqdm(val_loader, desc="Estimating threshold", leave=False):
            x = x.to(device)
            logits = model(x)
            margins.extend(compute_boundary_distance(logits).cpu().tolist())

    threshold = float(torch.quantile(torch.tensor(margins), q=quantile))
    return threshold


def predict_ood_mask(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    threshold: float,
    device: str | torch.device = "auto",
) -> torch.Tensor:
    """Return boolean tensor marking OOD predictions for the given loader."""
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    model.eval().to(device)

    preds = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Predicting OOD", leave=False):
            x = x.to(device)
            logits = model(x)
            is_id, _ = classify_batch(logits, threshold)
            preds.append(~is_id)  # OOD mask

    return torch.cat(preds)


def _auto_device(dev: str | torch.device = "auto") -> torch.device:
    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(dev)


# ---------------------------------------------------------------------------
# CIFAR-10 Loader ------------------------------------------------------------
# ---------------------------------------------------------------------------


def _cifar10_loaders(data_root: str, batch_size: int, val_split: float = 0.1):
    cfg = DatasetConfig.get_config("cifar10")
    mean, std = cfg["mean"], cfg["std"]
    normalize = T.Normalize(mean, std)

    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
    test_tf = T.Compose([T.ToTensor(), normalize])

    train_set = tv_datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    test_set = tv_datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)

    val_len = int(len(train_set) * val_split)
    train_len = len(train_set) - val_len
    train_subset, val_subset = random_split(train_set, [train_len, val_len])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(
        description="Margin-based OOD detection on CIFAR-10 using models from rq1.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model", type=str, default="resnet18", choices=list(DatasetConfig.SUPPORTED_MODELS.keys()), help="Model architecture defined in rq1.py")
    parser.add_argument("--data-dir", type=str, default="/Users/mukher74/research/data", help="CIFAR data root directory")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--quantile", type=float, default=0.05, help="Quantile for threshold estimation")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, mps, cpu")

    args = parser.parse_args()

    device = _auto_device(args.device)
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading CIFAR-10 ...")
    _, val_loader, test_loader = _cifar10_loaders(args.data_dir, args.batch_size)

    print(f"[INFO] Building model '{args.model}' ...")
    model = build_model(args.model, num_classes=10).to(device)

    print("[INFO] Estimating margin threshold ...")
    threshold = estimate_margin_threshold(model, val_loader, quantile=args.quantile, device=device)
    print(f"[RESULT] Estimated threshold (quantile {args.quantile}): {threshold:.4f}")

    print("[INFO] Predicting OOD mask on CIFAR-10 test set ...")
    ood_mask = predict_ood_mask(model, test_loader, threshold, device=device)
    id_ratio = (~ood_mask).float().mean().item()
    print(f"[RESULT] Fraction predicted as ID on test set: {id_ratio * 100:.2f}%")

    