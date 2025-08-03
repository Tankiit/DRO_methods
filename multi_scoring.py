import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

class MultiScoreOODDetector:
    """Multi-scoring framework for OOD detection."""
    
    def __init__(self, model, num_classes=1000, device='cuda'):
        self.model = model
        self.num_classes = num_classes
        self.device = device
        self.feature_mean = None
        self.feature_cov = None
        self.class_means = None
        self.class_covs = None
        self.temperature = 1000.0  # For ODIN
        self.noise_magnitude = 0.0014  # For ODIN

    # ------------------------------------------------------------------
    # Helper: extract features regardless of backbone type
    # ------------------------------------------------------------------
    def get_features(self, x: torch.Tensor, model=None) -> torch.Tensor:
        """Return a (B, D) feature tensor for input *x*.

        The different models used in this repo expose features in different
        ways.  This utility tries, in order:

        1.  A dedicated ``get_features`` method on the backbone.
        2.  Vision-Transformer path: reconstruct CLS token embedding.
        3.  Fallback: assume forward returns ``(logits, features)``.
        """
        if model is None:
            model = self.model

        # 1. Explicit helper provided by the backbone
        if hasattr(model, "get_features") and callable(model.get_features):
            return model.get_features(x)

        # 2. ViT or DINO models (have patch_embed / blocks)
        if hasattr(model, "patch_embed") and hasattr(model, "blocks"):
            B = x.size(0)
            tokens = model.patch_embed(x)
            # prepend CLS token when present
            if hasattr(model, "cls_token") and model.cls_token is not None:
                cls = model.cls_token.expand(B, -1, -1)
                tokens = torch.cat((cls, tokens), dim=1)

            # positional embeddings (interpolate if needed)
            if getattr(model, "pos_embed", None) is not None:
                pos = model.pos_embed
                if tokens.shape[1] != pos.shape[1]:
                    cls_pos, pos_grid = pos[:, :1], pos[:, 1:]
                    gs_old = int((pos_grid.shape[1]) ** 0.5)
                    gs_new = int((tokens.shape[1]-1) ** 0.5)
                    pos_grid = pos_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
                    pos_grid = torch.nn.functional.interpolate(
                        pos_grid, size=(gs_new, gs_new), mode="bicubic", align_corners=False
                    ).permute(0, 2, 3, 1).reshape(1, -1, pos_grid.shape[1])
                    pos = torch.cat((cls_pos, pos_grid), dim=1)
                tokens = tokens + pos

            if hasattr(model, "pos_drop"):
                tokens = model.pos_drop(tokens)
            for blk in model.blocks:
                tokens = blk(tokens)
            if hasattr(model, "norm"):
                tokens = model.norm(tokens)
            return tokens[:, 0]  # CLS token

        # 3. Fallback: forward returns tuple (logits, features)
        out = model(x)
        if isinstance(out, tuple) and len(out) == 2:
            return out[1]

        raise AttributeError("Unable to extract features: please implement get_features() in backbone.")
    
    def fit_feature_statistics(self, id_loader):
        """Fit feature statistics on ID data."""
        print("Computing feature statistics...")
        self.model.eval()
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for data, labels in tqdm(id_loader, desc='Computing features'):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # Get features using the get_features method
                features = self.get_features(data, self.model)
                
                features_list.append(features.cpu())
                labels_list.append(labels.cpu())
        
        # Concatenate all features and labels
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        # Compute global statistics
        self.feature_mean = features.mean(0)
        self.feature_cov = torch.cov(features.T)
        
        # Compute class-wise statistics
        self.class_means = []
        self.class_covs = []
        for c in range(self.num_classes):
            idx = (labels == c)
            if idx.sum() > 0:
                class_features = features[idx]
                self.class_means.append(class_features.mean(0))
                self.class_covs.append(torch.cov(class_features.T))
            else:
                self.class_means.append(torch.zeros_like(self.feature_mean))
                self.class_covs.append(torch.eye(self.feature_cov.size(0), device=self.feature_cov.device, dtype=self.feature_cov.dtype))
        
        self.class_means = torch.stack(self.class_means)
        self.class_covs = torch.stack(self.class_covs)
        print("Feature statistics computed successfully")
    
    def compute_energy_score(self, data: torch.Tensor) -> torch.Tensor:
        """Compute energy-based score."""
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(data), 32):  # Process in batches
                batch = data[i:i+32].to(self.device)
                output = self.model(batch)
                # Handle both single output and tuple output
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                energy = -torch.logsumexp(logits, dim=1)
                scores.append(energy.cpu())
        
        return torch.cat(scores)
    
    def compute_mahalanobis_score(self, data: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance-based score."""
        self.model.eval()
        scores = []
        
        if self.feature_mean is None or self.feature_cov is None:
            raise ValueError("Must call fit_feature_statistics first!")
        
        inv_cov = torch.linalg.inv(self.feature_cov + 1e-6 * torch.eye(self.feature_cov.shape[0]))
        
        with torch.no_grad():
            for i in range(0, len(data), 32):
                batch = data[i:i+32].to(self.device)
                # Use unified extractor so the logic works for all backbones
                features = self.get_features(batch, self.model)
                
                # Compute Mahalanobis distance
                centered = features - self.feature_mean.to(self.device)
                scores_batch = torch.diag(
                    centered @ inv_cov.to(self.device) @ centered.T
                )
                scores.append(scores_batch.cpu())
        
        return torch.cat(scores)
    
    def compute_msp_score(self, data: torch.Tensor) -> torch.Tensor:
        """Compute Maximum Softmax Probability score."""
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(data), 32):
                batch = data[i:i+32].to(self.device)
                output = self.model(batch)
                # Handle both single output and tuple output
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                probs = F.softmax(logits, dim=1)
                scores.append(probs.max(dim=1)[0].cpu())
        
        return torch.cat(scores)
    
    def compute_odin_score(self, data: torch.Tensor) -> torch.Tensor:
        """Compute ODIN score with temperature scaling and input preprocessing."""
        scores = []
        self.model.eval()
        
        for i in range(0, len(data), 32):
            batch = data[i:i+32].to(self.device)
            batch.requires_grad_(True)
            
            # Temperature scaling
            output = self.model(batch)
            # Handle both single output and tuple output
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            scaled_logits = logits / self.temperature
            
            # Input preprocessing
            loss = torch.sum(-F.log_softmax(scaled_logits, dim=1))
            loss.backward()
            
            gradient = batch.grad.data
            gradient = torch.sign(gradient)
            batch_perturbed = batch - self.noise_magnitude * gradient
            batch_perturbed = torch.clamp(batch_perturbed, 0, 1)
            
            # Get final scores
            with torch.no_grad():
                output_perturbed = self.model(batch_perturbed)
                # Handle both single output and tuple output
                if isinstance(output_perturbed, tuple):
                    logits_perturbed = output_perturbed[0]
                else:
                    logits_perturbed = output_perturbed
                probs = F.softmax(logits_perturbed / self.temperature, dim=1)
                scores.append(probs.max(dim=1)[0].cpu())
            
            batch.grad = None
        
        return torch.cat(scores)
    
    def compute_ensemble_score(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ensemble of multiple scores."""
        energy_scores = self.compute_energy_score(data)
        mahalanobis_scores = self.compute_mahalanobis_score(data)
        msp_scores = self.compute_msp_score(data)
        
        # Normalize scores to [0,1] - handle edge case where all scores are identical
        def safe_normalize(scores):
            min_val, max_val = scores.min(), scores.max()
            if max_val - min_val == 0:
                return torch.zeros_like(scores)  # All values are identical
            return (scores - min_val) / (max_val - min_val)
        
        energy_scores = safe_normalize(energy_scores)
        mahalanobis_scores = safe_normalize(mahalanobis_scores)  
        msp_scores = safe_normalize(msp_scores)
        
        # Combine scores (weighted average)
        ensemble_scores = (
            0.4 * energy_scores + 
            0.4 * mahalanobis_scores + 
            0.2 * msp_scores
        )
        
        return ensemble_scores, energy_scores, mahalanobis_scores 