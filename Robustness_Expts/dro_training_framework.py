"""dro_training_framework.py
Comprehensive Distributionally-Robust Optimisation (DRO) training framework
===========================================================================
This standalone module implements:
• A flexible :class:`DROTrainingConfig` dataclass for hyper-parameters
• :class:`DROTrainingFramework` that orchestrates training, validation, and monitoring
• Supporting utilities such as :class:`EfficiencyMonitor`, :class:`PerformanceTracker`,
  and (placeholder) foundation-model adapters

Run the module directly to execute an example training loop with dummy data:
$ python dro_training_framework.py
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class DROTrainingConfig:
    """Hyper-parameters & settings for DRO training."""

    # DRO parameters
    dro_method: str = "chi_square"  # {chi_square, wasserstein, cvar}
    uncertainty_radius: float = 0.1
    robustness_weight: float = 1.0

    # Efficiency parameters
    efficiency_mode: str = "balanced"  # {fast, balanced, accurate}
    batch_size: int = 128
    low_rank_dim: int = 50
    enable_batch_processing: bool = True

    # Foundation-model parameters
    foundation_model: str = "none"  # {none, clip, dinov2}
    freeze_backbone: bool = True
    feature_adaptation: bool = True

    # Training parameters
    learning_rate: float = 1e-3
    num_epochs: int = 100
    validation_freq: int = 5
    early_stopping_patience: int = 10


# ---------------------------------------------------------------------------
# Core training framework ----------------------------------------------------
# ---------------------------------------------------------------------------

class DROTrainingFramework:
    """End-to-end DRO training with monitoring and optional foundation models."""

    def __init__(self, config: DROTrainingConfig, device: str | torch.device = "auto") -> None:
        """Create training framework.

        Parameters
        ----------
        device: str | torch.device
            • "auto" (default) → picks CUDA if available, otherwise MPS, else CPU.
            • "cuda", "mps", "cpu" or explicit torch.device are also accepted.
        """
        self.config = config

        def _select_device(dev):
            if dev == "auto":
                if torch.cuda.is_available():
                    return torch.device("cuda")
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    return torch.device("mps")
                return torch.device("cpu")
            return torch.device(dev)

        self.device = _select_device(device)

        self.dro_criterion = self._initialize_dro_criterion()
        self.efficiency_monitor = EfficiencyMonitor()
        self.performance_tracker = PerformanceTracker()
        self.foundation_adapter: Optional[BaseFoundationAdapter] = None

        if config.foundation_model != "none":
            self.foundation_adapter = self._initialize_foundation_adapter()

    # ---------------------------------------------------------------------
    # Internal helpers ----------------------------------------------------
    # ---------------------------------------------------------------------

    def _initialize_dro_criterion(self):
        """Select the appropriate DRO loss based on *config*."""

        if self.config.dro_method == "chi_square":
            if self.config.efficiency_mode == "fast" and RobustLoss is not None:
                return RobustLoss(geometry="chi-square", size=self.config.uncertainty_radius)
            # Full neural-network implementation
            return Chi2NNDRO(input_dim=512, output_dim=10, reg=self.config.uncertainty_radius)

        if self.config.dro_method == "wasserstein":
            return WNNDRO(input_dim=512, output_dim=10, reg=self.config.uncertainty_radius)

        raise ValueError(f"Unsupported DRO method: {self.config.dro_method}")

    def _initialize_foundation_adapter(self):
        if self.config.foundation_model == "clip":
            return CLIPFoundationAdapter(self.config.freeze_backbone, self.config.feature_adaptation)
        if self.config.foundation_model == "dinov2":
            return DINOv2FoundationAdapter(self.config.freeze_backbone, self.config.feature_adaptation)
        raise ValueError(f"Unsupported foundation model: {self.config.foundation_model}")

    # ---------------------------------------------------------------------
    # Public API ----------------------------------------------------------
    # ---------------------------------------------------------------------

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loaders_ood: Dict[str, DataLoader],
    ) -> Dict:
        """Main training loop with validation and OOD evaluation."""

        logging.info("🚀 Starting DRO training – %s", self.config)
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        best_metric = 0.0
        patience_ctr = 0
        results: Dict[str, List[Dict]] = {
            "efficiency_metrics": [],
            "performance_metrics": [],
        }

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            train_stats = self._train_epoch(model, train_loader, optimizer, epoch)
            epoch_train_time = time.time() - epoch_start

            if epoch % self.config.validation_freq == 0:
                val_stats = self._validate_epoch(model, val_loader, test_loaders_ood, epoch)
                results["efficiency_metrics"].append(
                    {
                        "epoch": epoch,
                        "train_time": epoch_train_time,
                        "memory_mb": self.efficiency_monitor.get_memory_usage(),
                    }
                )
                results["performance_metrics"].append({"epoch": epoch, **val_stats})

                current_metric = val_stats["avg_ood_auroc"]
                if current_metric > best_metric:
                    best_metric = current_metric
                    patience_ctr = 0
                    self._save_best_model(model, epoch, val_stats)
                else:
                    patience_ctr += 1

                self._log_epoch(epoch, train_stats, val_stats, epoch_train_time)

                if patience_ctr >= self.config.early_stopping_patience:
                    logging.info("Early stopping at epoch %d", epoch)
                    break

        results["best_auroc"] = best_metric
        return results

    # ------------------------------------------------------------------
    # Epoch routines ---------------------------------------------------
    # ------------------------------------------------------------------

    def _train_epoch(self, model, loader, optimizer, epoch) -> Dict:
        model.train()
        tot_loss = 0.0
        metric_batches: List[Dict] = []

        for batch_idx, (x, y) in enumerate(loader):
            t0 = time.time()
            x, y = x.to(self.device), y.to(self.device)
            if self.foundation_adapter:
                x = self.foundation_adapter.extract_features(x)

            optimizer.zero_grad()
            logits = model(x)

            loss = self._compute_dro_loss(logits, y)
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            metric_batches.append(
                {
                    "batch_time": time.time() - t0,
                    "batch_size": x.shape[0],
                }
            )

            if batch_idx % 100 == 0:
                logging.info("Epoch %d | Batch %d | loss %.4f", epoch, batch_idx, loss.item())

        avg_batch_time = np.mean([m["batch_time"] for m in metric_batches])
        return {"loss": tot_loss / len(loader), "avg_batch_time": avg_batch_time}

    def _validate_epoch(self, model, val_loader, test_loaders_ood, epoch) -> Dict:
        model.eval()
        tot_correct, tot_samples = 0, 0
        id_scores: List[float] = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                if self.foundation_adapter:
                    x = self.foundation_adapter.extract_features(x)
                logits = model(x)
                preds = logits.argmax(dim=1)
                tot_correct += (preds == y).sum().item()
                tot_samples += y.size(0)
                id_scores.extend(self._compute_ood_scores(logits).cpu().numpy())

        acc = tot_correct / tot_samples
        ood_res: Dict[str, float] = {}
        for name, ood_loader in test_loaders_ood.items():
            ood_scores: List[float] = []
            with torch.no_grad():
                for x, _ in ood_loader:
                    x = x.to(self.device)
                    if self.foundation_adapter:
                        x = self.foundation_adapter.extract_features(x)
                    logits = model(x)
                    ood_scores.extend(self._compute_ood_scores(logits).cpu().numpy())
            ood_res[name] = self._compute_auroc(id_scores, ood_scores)

        return {
            "accuracy": acc,
            "ood_results": ood_res,
            "avg_ood_auroc": float(np.mean(list(ood_res.values()))) if ood_res else 0.0,
        }

    # ------------------------------------------------------------------
    # Utility helpers --------------------------------------------------
    # ------------------------------------------------------------------

    def _compute_dro_loss(self, logits, targets):
        if RobustLoss is not None and isinstance(self.dro_criterion, RobustLoss):
            base_losses = F.cross_entropy(logits, targets, reduction="none")
            return self.dro_criterion(base_losses)
        if isinstance(self.dro_criterion, nn.Module):
            return self.dro_criterion(logits, targets)
        return F.cross_entropy(logits, targets)

    @staticmethod
    def _compute_ood_scores(logits):
        return -torch.logsumexp(logits, dim=1)

    @staticmethod
    def _compute_auroc(id_scores, ood_scores):
        from sklearn.metrics import roc_auc_score  # local import to avoid hard dep

        labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        scores = np.concatenate([-np.array(id_scores), -np.array(ood_scores)])
        return float(roc_auc_score(labels, scores))

    @staticmethod
    def _save_best_model(model, epoch, metrics):
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "metrics": metrics}, "best_dro_model.pth")

    @staticmethod
    def _log_epoch(epoch, train_stats, val_stats, elapsed):
        logging.info(
            "Epoch %d | loss %.4f | val acc %.4f | AUROC %.4f | time %.2fs",
            epoch,
            train_stats["loss"],
            val_stats["accuracy"],
            val_stats["avg_ood_auroc"],
            elapsed,
        )


# ---------------------------------------------------------------------------
# Supporting helper classes --------------------------------------------------
# ---------------------------------------------------------------------------

class EfficiencyMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.memory_snapshots: List[float] = []

    def get_memory_usage(self):
        mem_mb = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
        self.memory_snapshots.append(mem_mb)
        return mem_mb

    def get_summary(self):
        return {
            "total_time": time.time() - self.start_time,
            "peak_mem_mb": max(self.memory_snapshots) if self.memory_snapshots else 0.0,
            "avg_mem_mb": float(np.mean(self.memory_snapshots)) if self.memory_snapshots else 0.0,
        }


class PerformanceTracker:
    """Track performance metrics throughout training"""
    
    def __init__(self):
        self.metrics_history = []
    
    def add_metrics(self, epoch, metrics):
        """Add metrics for an epoch"""
        self.metrics_history.append({'epoch': epoch, **metrics})
    
    def get_summary(self):
        """Get performance summary"""
        if not self.metrics_history:
            return {}
        
        best_epoch = max(self.metrics_history, key=lambda x: x.get('avg_ood_auroc', 0))
        return {
            'best_epoch': best_epoch['epoch'],
            'best_performance': best_epoch.get('avg_ood_auroc', 0),
            'final_performance': self.metrics_history[-1].get('avg_ood_auroc', 0)
        }


# ---------------------------------------------------------------------------
# Foundation-model adapters (stubs) -----------------------------------------
# ---------------------------------------------------------------------------

class BaseFoundationAdapter(nn.Module):
    """Abstract adapter."""

    def __init__(self):
        super().__init__()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError


class CLIPFoundationAdapter(BaseFoundationAdapter):
    def __init__(self, freeze_backbone: bool, enable_adaptation: bool):
        super().__init__()
        # TODO: load CLIP backbone (omitted for brevity)
        self.enable_adaptation = enable_adaptation
        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x  # placeholder – replace with real CLIP embedding


class DINOv2FoundationAdapter(BaseFoundationAdapter):
    def __init__(self, freeze_backbone: bool, enable_adaptation: bool):
        super().__init__()
        # TODO: load DINOv2 backbone (omitted for brevity)
        self.enable_adaptation = enable_adaptation
        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x  # placeholder


# ---------------------------------------------------------------------------
# Example usage --------------------------------------------------------------
# ---------------------------------------------------------------------------

def _make_dummy_loader(num_batches: int, batch_size: int, input_dim: int, num_classes: int):
    dataset = [
        (
            torch.randn(batch_size, input_dim),
            torch.randint(0, num_classes, (batch_size,)),
        )
        for _ in range(num_batches)
    ]
    return DataLoader(dataset, batch_size=None)  # already batched tuples


def main_dro_training_example():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = DROTrainingConfig(num_epochs=5, validation_freq=1, batch_size=64)
    trainer = DROTrainingFramework(cfg)

    model = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10))

    train_loader = _make_dummy_loader(20, cfg.batch_size, 512, 10)
    val_loader = _make_dummy_loader(5, cfg.batch_size, 512, 10)
    ood_loader = _make_dummy_loader(5, cfg.batch_size, 512, 10)

    results = trainer.train(
        model,
        train_loader,
        val_loader,
        {"dummy_ood": ood_loader},
    )

    logging.info("Training finished – summary: %s", results)
    return results


if __name__ == "__main__":  # pragma: no cover
    main_dro_training_example() 