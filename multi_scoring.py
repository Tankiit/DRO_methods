import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

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
    
    def fit_feature_statistics(self, id_loader):
        """Fit feature statistics on ID data."""
        print("Computing feature statistics...")
        self.model.eval()
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for data, labels in id_loader:
                data = data.to(self.device)
                # Get features
                if hasattr(self.model, 'get_features'):
                    features = self.model.get_features(data)
                else:
                    _, features = self.model(data)
                features_list.append(features.cpu())
                labels_list.append(labels)
        
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        # Compute global statistics
        self.feature_mean = features.mean(0)
        self.feature_cov = torch.cov(features.T)
        
        # Compute class-conditional statistics
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
                self.class_covs.append(torch.eye_like(self.feature_cov))
        
        self.class_means = torch.stack(self.class_means)
        self.class_covs = torch.stack(self.class_covs)
        print("Feature statistics computed.")
    
    def compute_energy_score(self, data: torch.Tensor) -> torch.Tensor:
        """Compute energy-based score."""
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(data), 32):  # Process in batches
                batch = data[i:i+32].to(self.device)
                logits, _ = self.model(batch)
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
                if hasattr(self.model, 'get_features'):
                    features = self.model.get_features(batch)
                else:
                    _, features = self.model(batch)
                
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
                logits, _ = self.model(batch)
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
            logits, _ = self.model(batch)
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
                logits_perturbed, _ = self.model(batch_perturbed)
                probs = F.softmax(logits_perturbed / self.temperature, dim=1)
                scores.append(probs.max(dim=1)[0].cpu())
            
            batch.grad = None
        
        return torch.cat(scores)
    
    def compute_ensemble_score(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ensemble of multiple scores."""
        energy_scores = self.compute_energy_score(data)
        mahalanobis_scores = self.compute_mahalanobis_score(data)
        msp_scores = self.compute_msp_score(data)
        
        # Normalize scores to [0,1]
        energy_scores = (energy_scores - energy_scores.min()) / (energy_scores.max() - energy_scores.min())
        mahalanobis_scores = (mahalanobis_scores - mahalanobis_scores.min()) / (mahalanobis_scores.max() - mahalanobis_scores.min())
        msp_scores = (msp_scores - msp_scores.min()) / (msp_scores.max() - msp_scores.min())
        
        # Combine scores (weighted average)
        ensemble_scores = (
            0.4 * energy_scores + 
            0.4 * mahalanobis_scores + 
            0.2 * msp_scores
        )
        
        return ensemble_scores, energy_scores, mahalanobis_scores 