import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import timm
from timm.models.vision_transformer import VisionTransformer
import argparse
import json
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import os
import functools
import traceback

class TimmFeatureExtractor(nn.Module):
    """Feature extractor based on timm models with separate backbone and classifier."""
    def __init__(self, model_name='resnet18', num_classes=10, pretrained=False, use_checkpoint=False):
        super().__init__()
        # Create backbone without classifier
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,  # Remove classifier
            global_pool='avg'  # Use average pooling
        )
        
        # Enable gradient checkpointing if requested
        if use_checkpoint:
            self.backbone.set_grad_checkpointing(True)
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Create separate classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Store dimensions
        self.feature_dim = feature_dim
        self.num_classes = num_classes
    
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        if return_features:
            return features
        logits = self.classifier(features)
        return logits, features

    def get_features(self, x, model):
        """Extract features from the model"""
        if hasattr(model, 'get_features'):
            # Use get_features method if available
            return model.get_features(x)
        elif hasattr(model, 'blocks'):
            # For ViT models, use the output before the head
            B = x.shape[0]
            
            # Get patch embeddings
            x = model.patch_embed(x)  # Shape: B, N, D
            
            # Add CLS token
            cls_tokens = model.cls_token.expand(B, -1, -1)  # Shape: B, 1, D
            x = torch.cat((cls_tokens, x), dim=1)  # Shape: B, N+1, D
            
            # Add position embedding
            if model.pos_embed is not None:
                # Get the current sequence length (including CLS token)
                curr_L = x.shape[1]
                pos_L = model.pos_embed.shape[1]
                
                if curr_L != pos_L:
                    # Need to interpolate position embeddings
                    # First, remove CLS token embedding and reshape to 2D grid
                    pos_embed = model.pos_embed
                    cls_pos_embed = pos_embed[:, 0:1]
                    pos_embed_grid = pos_embed[:, 1:].reshape(1, int((pos_L-1)**0.5), int((pos_L-1)**0.5), -1)
                    
                    # Interpolate grid to new size
                    new_size = int((curr_L-1)**0.5)
                    pos_embed_grid = torch.nn.functional.interpolate(
                        pos_embed_grid.permute(0, 3, 1, 2),
                        size=(new_size, new_size),
                        mode='bicubic',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                    
                    # Flatten grid and add CLS token embedding back
                    pos_embed_grid = pos_embed_grid.reshape(1, -1, pos_embed.shape[-1])
                    pos_embed = torch.cat([cls_pos_embed, pos_embed_grid], dim=1)
                    
                    x = x + pos_embed
                else:
                    x = x + model.pos_embed
            
            x = model.pos_drop(x)
            
            # Apply transformer blocks
            for block in model.blocks:
                x = block(x)
            
            x = model.norm(x)
            # Use CLS token features
            features = x[:, 0]
            return features
        else:
            # For other models, assume the forward pass returns (logits, features)
            _, features = model(x)
            return features

    def get_logits(self, features):
        """Utility method to get logits from features."""
        return self.classifier(features)

def safe_normalize(x, eps=1e-8):
    """Safely normalize tensor values to prevent overflow/underflow"""
    if torch.is_tensor(x):
        x = x.clamp(-1e6, 1e6)  # Prevent extreme values
        mean = x.mean().detach()
        std = x.std().detach() + eps
        return ((x - mean) / std).clamp(-5, 5)  # Keep values in reasonable range
    return x

def safe_log(x, eps=1e-8):
    """Safely compute log to prevent numerical issues"""
    return torch.log(x.clamp(min=eps))

def safe_div(x, y, eps=1e-8):
    """Safely divide tensors"""
    return x / (y + eps)

class OptimizedHierarchicalDRO(nn.Module):
    """
    Optimized version of Hierarchical DRO with performance improvements.
    Reduces virtual sample generation frequency and simplifies computations.
    """
    def __init__(self, 
                 model,
                 num_classes: int = 10,
                 device: str = 'cpu',
                 # Pixel-level parameters
                 pixel_radius: float = 0.03,
                 pixel_steps: int = 7,
                 pixel_lr: float = 2e-3,
                 # Feature-level parameters  
                 feature_radius: float = 0.1,
                 feature_steps: int = 5,
                 feature_lr: float = 1e-2,
                 # Loss weighting
                 pixel_weight: float = 0.3,
                 feature_weight: float = 0.5,
                 cross_level_weight: float = 0.2):
        
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.device = device
        
        # Store hyperparameters
        self.pixel_radius = pixel_radius
        self.pixel_steps = pixel_steps
        self.pixel_lr = pixel_lr
        self.feature_radius = feature_radius
        self.feature_steps = feature_steps
        self.feature_lr = feature_lr
        
        # Initialize tensorboard writer and step counter
        self.writer = None
        self.global_step = 0
        self.current_epoch = 0
        
        # Optimization settings
        self.virtual_sample_frequency = 3  # Generate virtual samples every N steps
        self.current_step = 0
        self.batch_virtual_size = 4  # Size for batch processing virtual samples
        self.enable_value_checks = False  # Disable expensive checks by default
        self.log_value_ranges = False  # Disable expensive logging by default
        
        # Initialize loss weights as learnable parameters with constraints
        self.pixel_weight = nn.Parameter(torch.tensor(pixel_weight).clamp(0.1, 0.5))
        self.feature_weight = nn.Parameter(torch.tensor(feature_weight).clamp(0.1, 0.5))
        self.cross_level_weight = nn.Parameter(torch.tensor(cross_level_weight).clamp(0.1, 0.5))
        
        # Initialize fast virtual generators
        self.fast_virtual_generator = FastVirtualGenerator(
            radius=pixel_radius,
            device=device
        )
        
        # Initialize energy-based components
        self.register_buffer('energy_temp', torch.tensor(0.5))
        
        # Initialize loss scaling factors
        self.register_buffer('loss_scale', torch.tensor(1.0))
        self.warmup_epochs = 5
        
        # Progressive training state
        self.training_config = {'use_virtual_samples': False, 'pixel_only': True}
    
    def forward(self, x_clean: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Optimized forward pass with periodic virtual sample generation"""
        self.current_step += 1
        
        # Always compute basic classification loss
        logits_clean, features_clean = self.model(x_clean)
        classification_loss = F.cross_entropy(logits_clean, y)
        
        # Update progressive training config based on epoch
        self.training_config = self._get_progressive_config()
        
        # Fast path: just classification + simple regularization
        if not self.training_config['use_virtual_samples'] or self.current_step % self.virtual_sample_frequency != 0:
            return {
                'total_loss': classification_loss,
                'pixel_loss': classification_loss,
                'feature_loss': torch.zeros_like(classification_loss),
                'consistency_loss': torch.zeros_like(classification_loss),
                'alignment_loss': torch.zeros_like(classification_loss)
            }
        
        # Full hierarchical forward pass with optimizations
        return self._optimized_hierarchical_forward(x_clean, y, classification_loss)
    
    def _get_progressive_config(self):
        """Progressive training schedule"""
        if self.current_epoch < 5:
            return {'use_virtual_samples': False, 'pixel_only': True}
        elif self.current_epoch < 15:
            return {'use_virtual_samples': True, 'pixel_only': True}
        else:
            return {'use_virtual_samples': True, 'pixel_only': False}
    
    def _optimized_hierarchical_forward(self, x_clean: torch.Tensor, y: torch.Tensor, 
                                      classification_loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Optimized hierarchical forward pass with batched virtual sample generation"""
        batch_size = x_clean.size(0)
        
        # Generate virtual samples in batches
        virtual_samples = self._generate_virtual_batch(x_clean, y)
        
        # Forward pass on virtual samples
        logits_virtual, features_virtual = self.model(virtual_samples)
        
        # Compute energy-based separation
        energy_clean = -torch.logsumexp(logits_clean, dim=1).mean()
        energy_virtual = -torch.logsumexp(logits_virtual, dim=1).mean()
        energy_margin = F.relu(energy_clean - energy_virtual + 1.0)
        
        # Feature-level computations only if needed
        if not self.training_config['pixel_only']:
            feature_loss = self._compute_feature_loss(features_clean, features_virtual)
        else:
            feature_loss = torch.zeros_like(classification_loss)
        
        # Simplified loss combination
        total_loss = (
            classification_loss + 
            0.1 * energy_margin +
            0.1 * feature_loss
        ).clamp(-10, 10)
        
        return {
            'total_loss': total_loss,
            'pixel_loss': classification_loss + 0.1 * energy_margin,
            'feature_loss': feature_loss,
            'consistency_loss': torch.zeros_like(classification_loss),
            'alignment_loss': torch.zeros_like(classification_loss),
            'energy_id': energy_clean,
            'energy_ood': energy_virtual
        }
    
    def _generate_virtual_batch(self, x_clean: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate virtual samples in memory-efficient batches"""
        all_virtual = []
        
        for i in range(0, x_clean.size(0), self.batch_virtual_size):
            end_idx = min(i + self.batch_virtual_size, x_clean.size(0))
            x_chunk = x_clean[i:end_idx]
            y_chunk = y[i:end_idx] if y is not None else None
            
            # Generate virtual samples for chunk
            with torch.cuda.amp.autocast():
                virtual_chunk = self.fast_virtual_generator.generate_fast_virtual_samples(
                    x_chunk, y_chunk, self.model
                )
                all_virtual.append(virtual_chunk.detach())
            
            # Clear intermediate results
            del virtual_chunk
            if i % 10 == 0:  # Periodic cache clearing
                torch.cuda.empty_cache()
        
        return torch.cat(all_virtual, dim=0)
    
    def _compute_feature_loss(self, features_clean: torch.Tensor, 
                            features_virtual: torch.Tensor) -> torch.Tensor:
        """Simplified feature-level loss computation"""
        # Simple feature space separation
        dist = torch.norm(features_virtual - features_clean, p=2, dim=1)
        feature_loss = -torch.mean(dist)
        
        return feature_loss

class FastVirtualGenerator(nn.Module):
    def __init__(self, noise_scale=0.1, adv_scale=0.1, energy_temp=1.0):
        super().__init__()
        # Initialize learnable parameters
        self.noise_scale = nn.Parameter(torch.tensor(noise_scale))
        self.adv_scale = nn.Parameter(torch.tensor(adv_scale))
        self.register_buffer('energy_temp', torch.tensor(energy_temp))
        self.model = None  # Will be set externally
    
    def get_features(self, x, model):
        """Extract features from the model"""
        if hasattr(model, 'get_features'):
            # Use get_features method if available
            return model.get_features(x)
        elif hasattr(model, 'blocks'):
            # For ViT models, use the output before the head
            B = x.shape[0]
            
            # Get patch embeddings
            x = model.patch_embed(x)  # Shape: B, N, D
            
            # Add CLS token
            cls_tokens = model.cls_token.expand(B, -1, -1)  # Shape: B, 1, D
            x = torch.cat((cls_tokens, x), dim=1)  # Shape: B, N+1, D
            
            # Add position embedding
            if model.pos_embed is not None:
                # Get the current sequence length (including CLS token)
                curr_L = x.shape[1]
                pos_L = model.pos_embed.shape[1]
                
                if curr_L != pos_L:
                    # Need to interpolate position embeddings
                    # First, remove CLS token embedding and reshape to 2D grid
                    pos_embed = model.pos_embed
                    cls_pos_embed = pos_embed[:, 0:1]
                    pos_embed_grid = pos_embed[:, 1:].reshape(1, int((pos_L-1)**0.5), int((pos_L-1)**0.5), -1)
                    
                    # Interpolate grid to new size
                    new_size = int((curr_L-1)**0.5)
                    pos_embed_grid = torch.nn.functional.interpolate(
                        pos_embed_grid.permute(0, 3, 1, 2),
                        size=(new_size, new_size),
                        mode='bicubic',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                    
                    # Flatten grid and add CLS token embedding back
                    pos_embed_grid = pos_embed_grid.reshape(1, -1, pos_embed.shape[-1])
                    pos_embed = torch.cat([cls_pos_embed, pos_embed_grid], dim=1)
                    
                    x = x + pos_embed
                else:
                    x = x + model.pos_embed
            
            x = model.pos_drop(x)
            
            # Apply transformer blocks
            for block in model.blocks:
                x = block(x)
            
            x = model.norm(x)
            # Use CLS token features
            features = x[:, 0]
            return features
        else:
            # For other models, assume the forward pass returns (logits, features)
            _, features = model(x)
            return features
    
    def get_logits(self, features):
        """Get logits from features using model's head"""
        if self.model is None:
            raise ValueError("Model not set. Please set self.model before calling forward.")
        if hasattr(self.model, 'head'):
            return self.model.head(features)
        else:
            raise ValueError("Model does not have a head layer.")
    
    def forward(self, x_clean, y):
        """Forward pass for virtual sample generation"""
        if self.model is None:
            raise ValueError("Model not set. Please set self.model before calling forward.")
        
        # Get clean features and logits
        with torch.set_grad_enabled(self.training):
            features_clean = self.get_features(x_clean, self.model)
            logits_clean = self.get_logits(features_clean)
            
            # Different behavior for training and evaluation
            if self.training:
                # Initialize adversarial samples
                x_adv = x_clean.detach().clone()
                x_adv.requires_grad_(True)  # Enable gradients for adversarial optimization
                
                # Add noise to create initial perturbation
                noise = torch.randn_like(x_clean) * self.noise_scale
                x_adv = x_adv + noise
                
                # Compute initial adversarial features and logits
                features_adv = self.get_features(x_adv, self.model)
                logits_adv = self.get_logits(features_adv)
                
                # Compute losses
                classification_loss = F.cross_entropy(logits_adv, y)
                energy_score = -torch.logsumexp(logits_adv / self.energy_temp, dim=1).mean()
                feature_distance = torch.norm(features_adv - features_clean, p=2, dim=1).mean()
                
                # Combine losses for gradient computation
                combined_loss = classification_loss + 0.1 * energy_score + 0.1 * feature_distance
                
                # Compute gradient for adversarial optimization
                grad = torch.autograd.grad(combined_loss, x_adv)[0]
                
                # Update adversarial samples
                with torch.no_grad():
                    x_adv = x_adv.detach() + self.adv_scale * grad.sign()
                    x_adv = torch.clamp(x_adv, 0, 1)  # Ensure valid image range
                
                # Compute final features and logits
                features_adv = self.get_features(x_adv, self.model)
                logits_adv = self.get_logits(features_adv)
                
                # Compute final losses
                final_classification_loss = F.cross_entropy(logits_adv, y)
                final_energy_score = -torch.logsumexp(logits_adv / self.energy_temp, dim=1).mean()
                final_feature_distance = torch.norm(features_adv - features_clean, p=2, dim=1).mean()
                
                # Compute total loss
                total_loss = (
                    final_classification_loss + 
                    0.1 * final_energy_score + 
                    0.1 * final_feature_distance
                )
            else:
                # During evaluation, just add noise without gradients
                x_adv = x_clean + torch.randn_like(x_clean) * self.noise_scale
                x_adv = torch.clamp(x_adv, 0, 1)
                
                features_adv = self.get_features(x_adv, self.model)
                logits_adv = self.get_logits(features_adv)
                
                final_classification_loss = F.cross_entropy(logits_adv, y)
                final_energy_score = -torch.logsumexp(logits_adv / self.energy_temp, dim=1).mean()
                final_feature_distance = torch.norm(features_adv - features_clean, p=2, dim=1).mean()
                
                total_loss = (
                    final_classification_loss + 
                    0.1 * final_energy_score + 
                    0.1 * final_feature_distance
                )
        
        return {
            'x_adv': x_adv.detach(),
            'features_clean': features_clean.detach(),
            'features_adv': features_adv.detach(),
            'logits_clean': logits_clean.detach(),
            'logits_adv': logits_adv.detach(),
            'classification_loss': final_classification_loss.detach(),
            'energy_score': final_energy_score.detach(),
            'feature_distance': final_feature_distance.detach(),
            'total_loss': total_loss,
            'virtual_samples': x_adv.detach()  # For compatibility with generate_fast_virtual_samples
        }
    
    def generate_fast_virtual_samples(self, x_clean: torch.Tensor, y: torch.Tensor, 
                                    model: nn.Module) -> torch.Tensor:
        """Legacy method for compatibility"""
        self.model = model
        with torch.no_grad():
            results = self.forward(x_clean, y)
        return results['virtual_samples']

# ============================================================================
# PIXEL-LEVEL VIRTUAL SAMPLE GENERATOR
# ============================================================================

class PixelLevelVirtualGenerator(nn.Module):
    """Generate virtual OOD samples through pixel-level perturbations"""
    
    def __init__(self, radius=0.03, steps=7, lr=2e-3):
        super().__init__()
        self.radius = radius
        self.steps = steps
        self.lr = lr
    
    def generate_multiple_strategies(self, x_clean: torch.Tensor, y: torch.Tensor, model) -> torch.Tensor:
        """Generate virtual OOD using multiple pixel-level strategies"""
        strategies = [
            self._adversarial_perturbation,
            self._corruption_simulation,
            self._texture_mixing,
            self._frequency_domain_perturbation
        ]
        
        all_virtual = []
        samples_per_strategy = max(x_clean.size(0) // len(strategies), 1)
        
        # Process in smaller chunks to save memory
        chunk_size = getattr(model, 'strategy_batch_size', 4)  # Default to 4 if not set
        
        for i, strategy in enumerate(strategies):
            start_idx = i * samples_per_strategy
            end_idx = min((i + 1) * samples_per_strategy, x_clean.size(0))
            
            if start_idx >= end_idx:
                continue
                
            x_subset = x_clean[start_idx:end_idx]
            y_subset = y[start_idx:end_idx] if y is not None else None
            
            # Process in chunks
            chunk_results = []
            for chunk_start in range(0, x_subset.size(0), chunk_size):
                chunk_end = min(chunk_start + chunk_size, x_subset.size(0))
                x_chunk = x_subset[chunk_start:chunk_end]
                y_chunk = y_subset[chunk_start:chunk_end] if y_subset is not None else None
                
                try:
                    with torch.cuda.amp.autocast():  # Use mixed precision
                        virtual_chunk = strategy(x_chunk, y_chunk, model)
                        chunk_results.append(virtual_chunk.detach())  # Detach to free memory
                        
                    # Clear cache periodically
                    if getattr(model, 'empty_cache_freq', 10) > 0 and chunk_start % model.empty_cache_freq == 0:
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # If OOM occurs, try with even smaller chunk
                        if chunk_size > 1:
                            print(f"Warning: OOM with chunk size {chunk_size}, trying smaller chunks")
                            half_chunk = x_chunk.chunk(2)
                            for small_chunk in half_chunk:
                                if small_chunk.size(0) == 0:
                                    continue
                                try:
                                    with torch.cuda.amp.autocast():
                                        virtual_small = strategy(small_chunk, y_chunk[:small_chunk.size(0)] if y_chunk is not None else None, model)
                                        chunk_results.append(virtual_small.detach())
                                except Exception as e2:
                                    print(f"Warning: Strategy {strategy.__name__} failed even with smaller chunk: {str(e2)}")
                                    # Use corruption simulation as fallback
                                    virtual_small = self._corruption_simulation(small_chunk, y_chunk[:small_chunk.size(0)] if y_chunk is not None else None, model)
                                    chunk_results.append(virtual_small.detach())
                        else:
                            print(f"Warning: Strategy {strategy.__name__} failed: {str(e)}")
                            # Use corruption simulation as fallback for this chunk
                            virtual_chunk = self._corruption_simulation(x_chunk, y_chunk, model)
                            chunk_results.append(virtual_chunk.detach())
                    else:
                        raise e
                
                # Clear some intermediate variables
                del x_chunk, y_chunk
                if 'virtual_chunk' in locals():
                    del virtual_chunk
                torch.cuda.empty_cache()
            
            # Combine chunks
            if chunk_results:
                try:
                    virtual_subset = torch.cat(chunk_results, dim=0)
                    all_virtual.append(virtual_subset)
                except RuntimeError as e:
                    print(f"Warning: Failed to combine chunks: {str(e)}")
                    continue
            
            # Clear chunk results to free memory
            del chunk_results
            torch.cuda.empty_cache()
        
        if not all_virtual:
            # Fallback: if all strategies failed, use basic corruption
            print("Warning: All strategies failed, using basic corruption simulation")
            return self._corruption_simulation(x_clean, y, model)
        
        return torch.cat(all_virtual, dim=0)
    
    def _adversarial_perturbation(self, x: torch.Tensor, y: torch.Tensor, model) -> torch.Tensor:
        """Generate adversarial examples to push toward OOD region"""
        # Process in smaller chunks if input is large
        chunk_size = 4  # Small chunk size to avoid OOM
        if x.size(0) > chunk_size:
            chunks = []
            for i in range(0, x.size(0), chunk_size):
                end_idx = min(i + chunk_size, x.size(0))
                x_chunk = x[i:end_idx]
                y_chunk = y[i:end_idx] if y is not None else None
                try:
                    with torch.cuda.amp.autocast():
                        adv_chunk = self._adversarial_perturbation(x_chunk, y_chunk, model)
                    chunks.append(adv_chunk)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Try with even smaller chunk
                        for j in range(i, end_idx):
                            try:
                                with torch.cuda.amp.autocast():
                                    adv_single = self._adversarial_perturbation(x[j:j+1], 
                                                                              y[j:j+1] if y is not None else None, 
                                                                              model)
                                chunks.append(adv_single)
                            except RuntimeError:
                                # If still OOM, use input as is
                                chunks.append(x[j:j+1])
                    else:
                        raise e
            return torch.cat(chunks, dim=0)
        
        # Process single small chunk
        x_adv = x.clone().requires_grad_(True)
        
        for _ in range(3):  # 3 PGD steps
            try:
                with torch.cuda.amp.autocast():
                    logits, _ = model(x_adv)
                    id_logits = logits[:, :model.num_classes] if logits.size(1) > model.num_classes else logits
                    
                    # Maximize entropy to make predictions uncertain
                    probs = F.softmax(id_logits, dim=1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                    loss = -entropy.mean()  # Minimize entropy = maximize uncertainty
                    
                    # Compute gradient
                    grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
                    
                    # Update and project
                    with torch.no_grad():
                        x_adv = x_adv + self.lr * grad.sign()
                        delta = x_adv - x
                        delta = torch.clamp(delta, -self.radius, self.radius)
                        x_adv = torch.clamp(x + delta, 0, 1)
                    
                    # Clear unnecessary tensors
                    del logits, id_logits, probs, entropy, loss, grad
                    torch.cuda.empty_cache()
                    
                    x_adv = x_adv.detach().requires_grad_(True)
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # If OOM during perturbation, return input
                    print(f"Warning: OOM in adversarial perturbation, returning original input")
                    return x
                else:
                    raise e
        
        return x_adv.detach()
    
    def _corruption_simulation(self, x: torch.Tensor, y: torch.Tensor, model) -> torch.Tensor:
        """Simulate common image corruptions"""
        batch_size = x.size(0)
        corrupted = []
        
        for i in range(batch_size):
            corruption_type = torch.randint(0, 4, (1,)).item()
            
            if corruption_type == 0:  # Gaussian noise
                noise = torch.randn_like(x[i]) * 0.1
                corrupted_img = torch.clamp(x[i] + noise, 0, 1)
            elif corruption_type == 1:  # Motion blur (approximated)
                # Apply blur per channel
                blurred = x[i:i+1].clone()
                kernel_size = 5
                kernel = torch.ones(1, 1, kernel_size, 1, device=x.device) / kernel_size
                for c in range(x.size(1)):
                    blurred[:,c:c+1] = F.conv2d(
                        x[i:i+1,c:c+1], 
                        kernel, 
                        padding=(kernel_size//2, 0)
                    )
                corrupted_img = blurred.squeeze(0)
            elif corruption_type == 2:  # Brightness
                factor = 0.5 + torch.rand(x_clean.size(0), 1, 1, 1, device=x_clean.device) * 1.0
                corrupted_img = torch.clamp(x_clean * factor, 0, 1)
            else:  # Contrast
                mean_val = x_clean[i].mean()
                factor = 0.5 + torch.rand(1, device=x_clean.device) * 1.0
                corrupted_img = torch.clamp((x_clean[i] - mean_val) * factor + mean_val, 0, 1)
            
            corrupted.append(corrupted_img)
        
        return torch.stack(corrupted)
    
    def _texture_mixing(self, x: torch.Tensor, y: torch.Tensor, model) -> torch.Tensor:
        """Mix textures between different images"""
        batch_size = x.size(0)
        mixed = []
        
        for i in range(batch_size):
            # Select another random image for mixing
            j = torch.randint(0, batch_size, (1,)).item()
            while j == i:
                j = torch.randint(0, batch_size, (1,)).item()
            
            # Random mixing coefficient
            alpha = torch.rand(1, device=x.device) * 0.5 + 0.25  # 0.25 to 0.75
            
            # Mix in frequency domain for texture mixing
            mixed_img = alpha * x[i] + (1 - alpha) * x[j]
            mixed.append(mixed_img)
        
        return torch.stack(mixed)
    
    def _frequency_domain_perturbation(self, x: torch.Tensor, y: torch.Tensor, model) -> torch.Tensor:
        """Perturb in frequency domain"""
        batch_size = x.size(0)
        perturbed = []
        
        # Define Sobel filters
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                             dtype=torch.float32, device=x.device).unsqueeze(0)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                             dtype=torch.float32, device=x.device).unsqueeze(0)
        
        for i in range(batch_size):
            # High-frequency noise
            high_freq_noise = torch.randn_like(x[i]) * 0.05
            
            # Apply high-pass filter effect (edge enhancement)
            edges = torch.zeros_like(x[i])
            for c in range(x.size(1)):
                # Reshape input for valid convolution
                img = x[i:i+1, c:c+1]  # Shape: [1, 1, H, W]
                
                # Apply Sobel filters
                edge_x = F.conv2d(img, sobel_x, padding=1)
                edge_y = F.conv2d(img, sobel_y, padding=1)
                
                # Compute edge magnitude
                edges[c] = torch.sqrt(edge_x.squeeze() ** 2 + edge_y.squeeze() ** 2)
            
            # Add edge-enhanced noise
            perturbed_img = torch.clamp(x[i] + high_freq_noise + 0.1 * edges, 0, 1)
            perturbed.append(perturbed_img)
        
        return torch.stack(perturbed)

# ============================================================================
# FEATURE-LEVEL VIRTUAL SAMPLE GENERATOR  
# ============================================================================

class FeatureLevelVirtualGenerator(nn.Module):
    """Generate virtual OOD samples in feature space"""
    
    def __init__(self, radius=0.1, steps=5, lr=1e-2):
        super().__init__()
        self.radius = radius
        self.steps = steps
        self.lr = lr
    
    def generate_feature_perturbations(self, features_clean: torch.Tensor, 
                                     y: torch.Tensor, model) -> torch.Tensor:
        """Generate virtual OOD features using multiple strategies"""
        strategies = [
            self._adversarial_feature_perturbation,
            self._class_interpolation,
            self._manifold_extrapolation,
            self._gaussian_mixture_sampling
        ]
        
        all_virtual = []
        samples_per_strategy = features_clean.size(0) // len(strategies)
        
        for i, strategy in enumerate(strategies):
            start_idx = i * samples_per_strategy
            end_idx = (i + 1) * samples_per_strategy if i < len(strategies) - 1 else features_clean.size(0)
            
            features_subset = features_clean[start_idx:end_idx]
            y_subset = y[start_idx:end_idx] if y is not None else None
            
            virtual_subset = strategy(features_subset, y_subset, model)
            all_virtual.append(virtual_subset)
        
        return torch.cat(all_virtual, dim=0)
    
    def _adversarial_feature_perturbation(self, features: torch.Tensor, 
                                        y: torch.Tensor, model) -> torch.Tensor:
        """Generate adversarial perturbations in feature space"""
        features_adv = features.clone().requires_grad_(True)
        
        for _ in range(self.steps):
            logits = model.classifier(features_adv)
            
            # Maximize prediction uncertainty
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            loss = -entropy.mean()
            
            grad = torch.autograd.grad(loss, features_adv, retain_graph=True)[0]
            features_adv = features_adv + self.lr * grad.sign()
            
            # Project to epsilon ball
            delta = features_adv - features
            delta_norm = torch.norm(delta, p=2, dim=1, keepdim=True)
            delta = delta * torch.min(torch.ones_like(delta_norm), 
                                    self.radius / (delta_norm + 1e-8))
            features_adv = features + delta
            
            features_adv = features_adv.detach().requires_grad_(True)
        
        return features_adv.detach()
    
    def _class_interpolation(self, features: torch.Tensor, y: torch.Tensor, model) -> torch.Tensor:
        """Interpolate between different classes"""
        batch_size = features.size(0)
        interpolated = []
        
        unique_labels = torch.unique(y)
        
        for i in range(batch_size):
            if len(unique_labels) > 1:
                # Select different class
                other_labels = unique_labels[unique_labels != y[i]]
                if len(other_labels) > 0:
                    target_label = other_labels[torch.randint(0, len(other_labels), (1,))]
                    
                    # Find sample from target class
                    target_mask = (y == target_label)
                    if target_mask.sum() > 0:
                        target_idx = torch.where(target_mask)[0][0]
                        
                        # Interpolate
                        alpha = torch.rand(1, device=features.device) * 0.6 + 0.2
                        interp_feature = alpha * features[i] + (1 - alpha) * features[target_idx]
                        interpolated.append(interp_feature)
                    else:
                        interpolated.append(features[i] + torch.randn_like(features[i]) * 0.1)
                else:
                    interpolated.append(features[i] + torch.randn_like(features[i]) * 0.1)
            else:
                interpolated.append(features[i] + torch.randn_like(features[i]) * 0.1)
        
        return torch.stack(interpolated)
    
    def _manifold_extrapolation(self, features: torch.Tensor, y: torch.Tensor, model) -> torch.Tensor:
        """Extrapolate beyond the manifold"""
        centroid = features.mean(dim=0, keepdim=True)
        extrapolated = []
        
        for feature in features:
            direction = feature - centroid.squeeze()
            extrapolation_factor = 1.5 + torch.rand(1, device=features.device) * 0.5
            extrapolated_feature = centroid.squeeze() + extrapolation_factor * direction
            extrapolated.append(extrapolated_feature)
        
        return torch.stack(extrapolated)
    
    def _gaussian_mixture_sampling(self, features: torch.Tensor, y: torch.Tensor, model) -> torch.Tensor:
        """Sample from Gaussian mixture in feature space"""
        # Compute per-class statistics
        unique_labels = torch.unique(y)
        class_means = {}
        class_stds = {}
        
        for label in unique_labels:
            mask = (y == label)
            class_features = features[mask]
            class_means[label.item()] = class_features.mean(dim=0)
            class_stds[label.item()] = class_features.std(dim=0) + 1e-6
        
        # Sample from mixture
        mixed_samples = []
        for i in range(features.size(0)):
            # Select random class (potentially different from original)
            label_idx = torch.randint(0, len(unique_labels), (1,)).item()
            selected_label = unique_labels[label_idx].item()
            
            # Sample from Gaussian
            mean = class_means[selected_label]
            std = class_stds[selected_label]
            
            # Add some noise to make it OOD-like
            noise_factor = 2.0  # Increase variance for OOD
            sample = mean + noise_factor * std * torch.randn_like(mean)
            mixed_samples.append(sample)
        
        return torch.stack(mixed_samples)

# ============================================================================
# CROSS-LEVEL CONSISTENCY MODULE
# ============================================================================

class CrossLevelConsistency(nn.Module):
    """Ensure consistency between pixel and feature level perturbations"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pixel_virtual: torch.Tensor, feature_virtual: torch.Tensor, 
                model) -> torch.Tensor:
        """Compute consistency loss between pixel and feature virtual samples"""
        # Extract features from pixel virtual samples
        with torch.no_grad():
            _, features_from_pixel = model(pixel_virtual)
            if features_from_pixel.dim() > 2:
                features_from_pixel = features_from_pixel.view(features_from_pixel.size(0), -1)
        
        # Both should be OOD-like in feature space
        consistency_loss = F.mse_loss(
            F.normalize(features_from_pixel, dim=1),
            F.normalize(feature_virtual, dim=1)
        )
        
        return consistency_loss

# ============================================================================
# MULTI-LEVEL UNCERTAINTY ESTIMATOR
# ============================================================================

class MultiLevelUncertaintyEstimator(nn.Module):
    """Estimate uncertainty at both pixel and feature levels"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def pixel_level_uncertainty(self, x_clean: torch.Tensor, 
                               x_virtual: torch.Tensor, model) -> Dict[str, torch.Tensor]:
        """Estimate uncertainty at pixel level"""
        with torch.no_grad():
            # Prediction variance
            logits_clean, _ = model(x_clean)
            logits_virtual, _ = model(x_virtual)
            
            # Entropy-based uncertainty
            entropy_clean = self._compute_entropy(logits_clean)
            entropy_virtual = self._compute_entropy(logits_virtual)
            
            # Confidence-based uncertainty
            confidence_clean = torch.max(F.softmax(logits_clean, dim=1), dim=1)[0]
            confidence_virtual = torch.max(F.softmax(logits_virtual, dim=1), dim=1)[0]
        
        return {
            'entropy_clean': entropy_clean,
            'entropy_virtual': entropy_virtual,
            'confidence_clean': confidence_clean,
            'confidence_virtual': confidence_virtual
        }
    
    def feature_level_uncertainty(self, features_clean: torch.Tensor, 
                                features_virtual: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate uncertainty at feature level"""
        with torch.no_grad():
            # Feature space distances
            feature_distances = torch.norm(features_virtual - features_clean, p=2, dim=1)
            
            # Feature variance
            feature_var_clean = torch.var(features_clean, dim=1)
            feature_var_virtual = torch.var(features_virtual, dim=1)
        
        return {
            'feature_distances': feature_distances,
            'feature_var_clean': feature_var_clean,
            'feature_var_virtual': feature_var_virtual
        }
    
    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute prediction entropy"""
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        return entropy

# ============================================================================
# INTEGRATION WITH MULTI-SCORING FRAMEWORK
# ============================================================================

class HierarchicalDROWithMultiScoring(nn.Module):
    def __init__(self, model, pixel_radius=0.03, feature_radius=0.1, pixel_weight=0.3, feature_weight=0.5, cross_weight=0.2, device='cuda'):
        super().__init__()
        self.model = model
        self.device = device
        
        # Initialize loss weights as Parameters
        self.pixel_weight = nn.Parameter(torch.tensor(pixel_weight))
        self.feature_weight = nn.Parameter(torch.tensor(feature_weight))
        self.cross_weight = nn.Parameter(torch.tensor(cross_weight))
        
        # Initialize radii
        self.pixel_radius = pixel_radius
        self.feature_radius = feature_radius
        
        # Initialize virtual sample generator
        self.fast_virtual_generator = FastVirtualGenerator(
            noise_scale=pixel_radius,  # Use pixel_radius for noise scale
            adv_scale=pixel_radius,    # Use pixel_radius for adversarial scale
            energy_temp=0.5            # Fixed energy temperature
        ).to(device)
        self.fast_virtual_generator.model = self.model  # Set model reference
        
        # Initialize training hyperparameters
        self.warmup_epochs = 5
        self.virtual_sample_frequency = 5
        self.batch_virtual_size = 4
        self.current_step = 0
        self.enable_value_checks = False
        self.log_value_ranges = False
        
        # Initialize tensorboard writer
        self.writer = None
        self.global_step = 0
        
        # Print initialization parameters
        print("\nHierarchicalDRO Initialization:")
        print(f"Warmup epochs: {self.warmup_epochs}")
        print(f"Virtual sample frequency: {self.virtual_sample_frequency}")
        print(f"Batch virtual size: {self.batch_virtual_size}")
        print(f"Initial loss weights: pixel={pixel_weight:.3f}, feature={feature_weight:.3f}, cross={cross_weight:.3f}")
        print(f"Energy temperature: {0.5:.3f}")
        print(f"Initial loss scale: {1.000:.3f}")
    
    def forward(self, x, y):
        """Forward pass through the model and virtual generator"""
        # Set model reference for virtual generator
        self.fast_virtual_generator.model = self.model
        
        # Generate virtual samples and compute losses
        results = self.fast_virtual_generator(x, y)
        
        return results

    def train_hierarchical_dro(self, train_loader, num_epochs=30, lr=0.001, val_loader=None, config=None, checkpoint_dir='checkpoints'):
        """Train the model using hierarchical DRO"""
        try:
            # Set up mixed precision if enabled
            scaler = torch.cuda.amp.GradScaler() if getattr(self, 'mixed_precision', False) else None
            
            # Update attributes from config
            if config:
                for key, value in config.items():
                    setattr(self, key, value)
                print(f"\nSetting mixed_precision = {self.mixed_precision}")
                print(f"Setting gradient_accumulation_steps = {self.gradient_accumulation_steps}")
                print(f"Setting grad_checkpoint = {self.grad_checkpoint}")
                print(f"Setting empty_cache_freq = {self.empty_cache_freq}")
                print(f"Setting ood_loader = {self.ood_loader}")
            
            print("\nUsing ViT-specific learning rate schedule")
            
            # Set up optimizers with parameter groups
            model_params = []
            if hasattr(self.model, 'patch_embed'):
                patch_embed_params = list(self.model.patch_embed.parameters())
                model_params.append({
                    'params': patch_embed_params,
                    'lr': lr * 0.1  # Lower LR for patch embedding
                })
                print(f"\nParameter groups:")
                print(f"patch_embed: {len(patch_embed_params)} tensors, {sum(p.numel() for p in patch_embed_params):,} parameters, lr={lr * 0.1}")
            
            if hasattr(self.model, 'blocks'):
                blocks_params = list(self.model.blocks.parameters())
                model_params.append({
                    'params': blocks_params,
                    'lr': lr * 0.5  # Medium LR for transformer blocks
                })
                print(f"blocks: {len(blocks_params)} tensors, {sum(p.numel() for p in blocks_params):,} parameters, lr={lr * 0.5}")
            
            if hasattr(self.model, 'head'):
                head_params = list(self.model.head.parameters())
                model_params.append({
                    'params': head_params,
                    'lr': lr  # Full LR for head
                })
                print(f"head: {len(head_params)} tensors, {sum(p.numel() for p in head_params):,} parameters, lr={lr}")
            
            # DRO parameters (weights and virtual generators)
            dro_params = [
                {'params': [self.pixel_weight, self.feature_weight, self.cross_weight], 'lr': lr * 0.01},
                {'params': list(self.fast_virtual_generator.parameters()), 'lr': lr * 0.01}
            ]
            
            # Create optimizers
            model_optimizer = torch.optim.AdamW(model_params)
            dro_optimizer = torch.optim.AdamW(dro_params)
            
            # Initialize tensorboard writer
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            try:
                self.writer = SummaryWriter(f'runs/hierarchical_dro_{timestamp}')
            except Exception as e:
                print(f"Warning: Failed to initialize tensorboard writer: {str(e)}")
                self.writer = None
            
            # Warmup scheduler with longer warmup
            warmup_epochs = self.warmup_epochs
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                model_optimizer, 
                max_lr=[g['lr'] for g in model_params],  # Different max LRs for each group
                epochs=num_epochs, 
                steps_per_epoch=len(train_loader),
                pct_start=warmup_epochs/num_epochs,
                div_factor=25.0,
                final_div_factor=1e4,
                anneal_strategy='cos'
            )
            
            # Initialize mixed precision training
            scaler = torch.cuda.amp.GradScaler() if getattr(self, 'mixed_precision', False) else None
            
            best_loss = float('inf')
            best_val_loss = float('inf')
            consecutive_errors = 0
            max_consecutive_errors = 3
            
            # Create checkpoint directory if it doesn't exist
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            for epoch in trange(num_epochs, desc='Training epochs'):
                # Set current epoch for warmup calculations
                self.current_epoch = epoch
                
                # Training mode
                self.model.train()
                self.fast_virtual_generator.train() # Ensure virtual generators are in training mode
                
                train_metrics = self._train_epoch(
                    train_loader, 
                    model_optimizer, 
                    dro_optimizer, 
                    scheduler, 
                    scaler
                )
                
                # Validation
                if val_loader is not None:
                    val_metrics = self._validate(val_loader)
                    
                    # Log validation metrics
                    for key, value in val_metrics.items():
                        self.writer.add_scalar(f'validation/{key}', value, epoch)
                    
                    # Save best model
                    if val_metrics['total_loss'] < best_val_loss:
                        best_val_loss = val_metrics['total_loss']
                        checkpoint_path = os.path.join(checkpoint_dir, f'best_model_{timestamp}.pt')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': model_optimizer.state_dict(),
                            'val_loss': best_val_loss,
                        }, checkpoint_path)
                
                # Log training metrics
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f'train/{key}', value, epoch)
                
                # Print progress
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"Train Loss: {train_metrics['total_loss']:.4f}")
                if val_loader is not None:
                    print(f"Val Loss: {val_metrics['total_loss']:.4f}")
                
                # Clear cache periodically
                if epoch % 5 == 0:
                    torch.cuda.empty_cache()
        
        except Exception as e:
            print("\nTraining stopped due to error:", str(e))
            traceback.print_exc()
            if self.writer is not None:
                self.writer.close()
            raise e
        
        finally:
            if self.writer is not None:
                self.writer.close()

    def _train_epoch(self, train_loader, model_optimizer, dro_optimizer, scheduler, scaler):
        """Train for one epoch"""
        epoch_metrics = defaultdict(float)
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch_idx, (data, targets) in enumerate(pbar):
            try:
                # Move data to device
                data = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Clear gradients
                model_optimizer.zero_grad(set_to_none=True)
                dro_optimizer.zero_grad(set_to_none=True)
                
                # Set model reference for virtual generator
                self.fast_virtual_generator.model = self.model
                
                # Forward pass with mixed precision if enabled
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        results = self.fast_virtual_generator(data, targets)  # Use fast_virtual_generator directly
                        total_loss = results['total_loss']
                    
                    # Backward pass with gradient scaling
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(model_optimizer)
                    scaler.unscale_(dro_optimizer)
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.fast_virtual_generator.parameters(), max_norm=0.5)
                    
                    # Step optimizers with scaling
                    scaler.step(model_optimizer)
                    scaler.step(dro_optimizer)
                    scaler.update()
                else:
                    # Standard forward and backward pass
                    results = self.fast_virtual_generator(data, targets)  # Use fast_virtual_generator directly
                    total_loss = results['total_loss']
                    total_loss.backward()
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.fast_virtual_generator.parameters(), max_norm=0.5)
                    
                    # Step optimizers
                    model_optimizer.step()
                    dro_optimizer.step()
                
                # Step scheduler AFTER optimizer steps
                scheduler.step()
                
                # Update metrics
                for key, value in results.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        epoch_metrics[key] += value.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': epoch_metrics['total_loss'] / (batch_idx + 1),
                    'lr': scheduler.get_last_lr()[0]
                })
                
                self.global_step += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nOOM in batch {batch_idx}. Skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Average metrics
        return {k: v / num_batches for k, v in epoch_metrics.items()}

    def _validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        self.fast_virtual_generator.eval()  # Set virtual generator to eval mode
        val_metrics = defaultdict(float)
        num_batches = len(val_loader)
        
        with torch.no_grad():  # Disable gradients for validation
            for data, targets in tqdm(val_loader, desc='Validating', leave=False):
                data = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                results = self.fast_virtual_generator(data, targets)
                
                # Update metrics
                for key, value in results.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        val_metrics[key] += value.item()
        
        # Average metrics
        return {k: v / num_batches for k, v in val_metrics.items()}

    def _evaluate_epoch(self, epoch):
        """Evaluate model during training"""
        self.model.eval()
        metrics = defaultdict(float)
        
        with torch.no_grad():
            # Implement validation logic here
            # This is a placeholder for actual validation metrics
            metrics['total_loss'] = 0.0
            metrics['accuracy'] = 0.0
        
        self.model.train()
        return metrics

    def _log_example_images(self, data, results, epoch):
        """Log example images to tensorboard"""
        # Original images
        grid = torchvision.utils.make_grid(data[:8])
        self.writer.add_image('images/original', grid, epoch)
        
        # Virtual samples
        if 'pixel_virtual_samples' in results:
            grid = torchvision.utils.make_grid(results['pixel_virtual_samples'][:8])
            self.writer.add_image('images/pixel_virtual', grid, epoch)
        
        # Feature visualizations (if available)
        if 'feature_virtual_samples' in results:
            try:
                feature_images = self._visualize_features(results['feature_virtual_samples'][:8])
                grid = torchvision.utils.make_grid(feature_images)
                self.writer.add_image('images/feature_virtual', grid, epoch)
            except:
                pass

    def _visualize_features(self, features, size=(224, 224)):
        """Convert feature vectors to visualizable images using PCA"""
        from sklearn.decomposition import PCA
        
        # Reshape features to 2D
        features_2d = features.cpu().view(features.size(0), -1).numpy()
        
        # Apply PCA
        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(features_2d)
        
        # Normalize to [0, 1]
        features_norm = (features_pca - features_pca.min()) / (features_pca.max() - features_pca.min())
        
        # Convert to RGB images
        images = torch.from_numpy(features_norm).float()
        images = images.view(images.size(0), 3, 1, 1)
        images = F.interpolate(images, size=size, mode='bilinear', align_corners=False)
        
        return images

    def setup_multi_scoring(self, id_loader):
        """Setup multi-scoring framework after hierarchical DRO training"""
        # Import your multi-scoring class
        from your_multi_scoring_module import MultiScoreOODDetector
        
        self.multi_scorer = MultiScoreOODDetector(
            self.model, 
            num_classes=self.num_classes,
            device=self.device
        )
        
        # Fit statistics for methods like Mahalanobis
        self.multi_scorer.fit_feature_statistics(id_loader)
        print("Multi-scoring framework initialized with hierarchical DRO-trained model")
    
    def comprehensive_evaluation(self, id_loader, ood_loaders, methods=None):
        """Comprehensive evaluation combining both levels of robustness"""
        if self.multi_scorer is None:
            raise RuntimeError("Must call setup_multi_scoring() first")
        
        if methods is None:
            methods = ['energy', 'mahalanobis', 'msp', 'odin', 'gradnorm', 'react', 'ensemble']
        
        print("Evaluating hierarchical DRO model with multi-scoring...")
        
        results = {}
        
        # Evaluate on each OOD dataset
        for ood_name, ood_loader in ood_loaders.items():
            print(f"\nEvaluating on {ood_name}...")
            
            # Get limited samples for efficiency
            id_data = self._get_limited_data(id_loader, max_samples=2000)
            ood_data = self._get_limited_data(ood_loader, max_samples=2000)
            
            ood_results = {}
            
            for method in tqdm(methods, desc=f'Computing scores for {ood_name}'):
                try:
                    # Compute scores
                    if method == 'ensemble':
                        id_scores, _, _ = self.multi_scorer.compute_ensemble_score(id_data)
                        ood_scores, _, _ = self.multi_scorer.compute_ensemble_score(ood_data)
                    else:
                        score_fn = getattr(self.multi_scorer, f'compute_{method}_score')
                        id_scores = score_fn(id_data)
                        ood_scores = score_fn(ood_data)
                    
                    # Compute metrics
                    metrics = self._compute_detection_metrics(id_scores, ood_scores)
                    ood_results[method] = metrics
                    
                    print(f"    {method}: AUROC={metrics['auroc']:.4f}, FPR@95%={metrics['fpr95']:.4f}")
                    
                except Exception as e:
                    print(f"    Failed to compute {method}: {e}")
                    continue
            
            results[ood_name] = ood_results
        
        return results
    
    def _get_limited_data(self, loader, max_samples=2000):
        """Get limited data for efficient evaluation"""
        data_list = []
        samples_collected = 0
        
        for data, _ in loader:
            data_list.append(data)
            samples_collected += data.size(0)
            if samples_collected >= max_samples:
                break
        
        return torch.cat(data_list)[:max_samples].to(self.device)
    
    def _compute_detection_metrics(self, id_scores, ood_scores):
        """Compute standard OOD detection metrics"""
        from sklearn.metrics import roc_auc_score, roc_curve
        
        id_scores_np = id_scores.cpu().numpy()
        ood_scores_np = ood_scores.cpu().numpy()
        
        y_true = np.concatenate([np.zeros_like(id_scores_np), np.ones_like(ood_scores_np)])
        y_score = np.concatenate([id_scores_np, ood_scores_np])
        
        auroc = roc_auc_score(y_true, y_score)
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        idx_95 = np.argmin(np.abs(tpr - 0.95))
        fpr95 = fpr[idx_95]
        
        return {
            'auroc': auroc,
            'fpr95': fpr95,
            'separation': np.mean(ood_scores_np) - np.mean(id_scores_np)
        }

    def _log_value_ranges(self, results):
        """Log value ranges for monitoring"""
        for key, value in results.items():
            if torch.is_tensor(value) and value.numel() > 1:
                self.writer.add_histogram(f'value_ranges/{key}', value, self.global_step)
                self.writer.add_scalar(f'value_ranges/{key}_min', value.min().item(), self.global_step)
                self.writer.add_scalar(f'value_ranges/{key}_max', value.max().item(), self.global_step)


# Example usage of hierarchical pixel + feature level DRO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar10', choices=['cifar10','cifar100','imagenet'])
    p.add_argument('--ood',     default='svhn',    choices=['svhn','cifar100','imagenet_a','imagenet_r','textures'])
    p.add_argument('--backbone','-b', default='resnet18', 
                  choices=['resnet18', 'resnet50', 'vit_base_patch16_224', 'vit_large_patch16_224', 
                          'dino_vits16', 'dino_vitb16'])
    p.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    p.add_argument('--epochs',  type=int, default=30)
    p.add_argument('--batch',   type=int, default=16)  # Reduced default batch size
    p.add_argument('--lr',      type=float, default=1e-3)
    # Additional arguments for hierarchical DRO
    p.add_argument('--pixel-radius',    type=float, default=0.03)
    p.add_argument('--feature-radius',  type=float, default=0.1)
    p.add_argument('--pixel-weight',    type=float, default=0.3)
    p.add_argument('--feature-weight',  type=float, default=0.5)
    p.add_argument('--cross-weight',    type=float, default=0.2)
    p.add_argument('--grad-checkpoint', action='store_true', help='Use gradient checkpointing')
    # Memory management options
    p.add_argument('--num-workers', type=int, default=4, help='Number of dataloader workers')
    p.add_argument('--strategy-batch-size', type=int, default=4, help='Batch size for virtual sample generation')
    p.add_argument('--empty-cache-freq', type=int, default=10, help='Frequency of torch.cuda.empty_cache() calls')
    p.add_argument('--mixed-precision', action='store_true', help='Use mixed precision training')
    p.add_argument('--gradient-accumulation', type=int, default=1, help='Number of gradient accumulation steps')
    # Directory settings
    p.add_argument('--data-dir', type=str, default='/home/tanmoy/research/data', 
                  help='Path to data directory')
    p.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                  help='Directory for saving checkpoints')
    p.add_argument('--output-dir', type=str, default='results',
                  help='Directory for saving results')
    p.add_argument('--runs-dir', type=str, default='runs',
                  help='Directory for tensorboard runs')
    return p.parse_args()

def setup_logging_and_directories(args):
    """Set up logging and create necessary directories"""
    # Print all arguments for logging purposes
    print("\nTraining Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    # Create necessary directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.runs_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

def get_imagenet_transform(train=True):
    """Get standard ImageNet transforms"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            normalize
        ])
    return transform

class FlatImageDataset(torch.utils.data.Dataset):
    """Dataset for loading images from a flat directory (no class subfolders)"""
    def __init__(self, root_dir, transform=None, is_ood=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_ood = is_ood
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(root_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
        self.image_files.sort()  # Sort for reproducibility
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        
        # Read image directly as tensor
        image = torchvision.io.read_image(img_name)
        
        # Convert to float and scale to [0, 1]
        image = image.float() / 255.0
        
        # Apply transforms
        if self.transform is not None:
            if isinstance(self.transform, transforms.Compose):
                # Filter out ToTensor transform since we already have a tensor
                tensor_transforms = transforms.Compose([
                    t for t in self.transform.transforms 
                    if not isinstance(t, transforms.ToTensor)
                ])
                image = tensor_transforms(image)
            else:
                image = self.transform(image)
        
        # Use -1 for OOD data, 0 otherwise
        label = -1 if self.is_ood else 0
        
        return image, label

def get_dataset(name, data_dir, train=True, batch_size=128, is_ood=False):
    """Get dataset loader based on name.
    Args:
        name: Dataset name (imagenet, cifar100, textures, svhn)
        data_dir: Root directory for datasets
        train: Whether to load training set
        batch_size: Batch size for data loader
        is_ood: Whether this is OOD data
    Returns:
        If train=True, returns (train_loader, val_loader)
        If train=False, returns (test_loader,)
    """
    if name == 'imagenet':
        if train and not is_ood:
            # For training/validation, use the test directory and split it
            dataset = FlatImageDataset(
                root_dir=os.path.join(data_dir, 'Imagenet/test'),
                transform=get_imagenet_transform(train=True),
                is_ood=False
            )
            
            # Split into train/val with fixed seed
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            print(f"Using {train_size} images for training")
            print(f"Using {val_size} images for validation")
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
            
            return train_loader, val_loader
        else:
            # For test or OOD, use the test directory directly
            dataset = FlatImageDataset(
                root_dir=os.path.join(data_dir, 'Imagenet/test'),
                transform=get_imagenet_transform(train=False),
                is_ood=is_ood
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
            return (loader,)
            
    elif name == 'cifar100':
        transform = get_imagenet_transform(train=False)  # Use ImageNet normalization
        dataset = torchvision.datasets.CIFAR100(
            root=os.path.join(data_dir, 'CIFAR100'),
            train=train,
            transform=transform,
            download=True
        )
        print(f"Found {len(dataset)} CIFAR100 {'training' if train else 'test'} images")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        return (loader,)
        
    elif name == 'textures':
        transform = get_imagenet_transform(train=False)  # Use ImageNet normalization
        dataset = torchvision.datasets.DTD(
            root=os.path.join(data_dir, 'DTD'),
            split='train' if train else 'test',
            transform=transform,
            download=True
        )
        print(f"Found {len(dataset)} DTD {'training' if train else 'test'} images")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        return (loader,)
        
    elif name == 'svhn':
        transform = get_imagenet_transform(train=False)  # Use ImageNet normalization
        dataset = torchvision.datasets.SVHN(
            root=os.path.join(data_dir, 'SVHN'),
            split='train' if train else 'test',
            transform=transform,
            download=True
        )
        print(f"Found {len(dataset)} SVHN {'training' if train else 'test'} images")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        return (loader,)
    
    else:
        raise ValueError(f"Unknown dataset: {name}")

def get_model(args):
    """Get model based on backbone name."""
    num_classes = 1000 if args.dataset == 'imagenet' else 100 if args.dataset == 'cifar100' else 10
    
    if args.backbone.startswith('vit'):
        # ViT models from timm
        model = timm.create_model(
            args.backbone,
            pretrained=args.pretrained,
            num_classes=num_classes,
            drop_path_rate=0.1,  # Recommended for fine-tuning
            img_size=224
        )
        if args.grad_checkpoint:
            model.set_grad_checkpointing(True)
            
    elif args.backbone.startswith('dino'):
        # DINO models
        if args.backbone == 'dino_vits16':
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=args.pretrained)
            embed_dim = 384  # ViT-S/16 embedding dimension
        else:  # dino_vitb16
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=args.pretrained)
            embed_dim = 768  # ViT-B/16 embedding dimension
        
        # Replace head for classification
        model.head = nn.Linear(embed_dim, num_classes)
        
        # Enable gradient checkpointing with use_reentrant=False
        if args.grad_checkpoint:
            def checkpoint_fn(module, *args, **kwargs):
                return torch.utils.checkpoint.checkpoint(
                    module,
                    *args,
                    use_reentrant=False,
                    **kwargs
                )
            
            # Apply checkpointing to transformer blocks
            for block in model.blocks:
                block.forward = functools.partial(checkpoint_fn, block.forward)
            print("Gradient checkpointing enabled with use_reentrant=False")
    
    else:
        # Standard ResNet models
        model = TimmFeatureExtractor(
            model_name=args.backbone,
            num_classes=num_classes,
            pretrained=args.pretrained,
            use_checkpoint=args.grad_checkpoint
        )
    
    # Print model configuration
    print(f"\nModel Configuration:")
    print(f"Backbone: {args.backbone}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Gradient Checkpointing: {args.grad_checkpoint}")
    print(f"Number of Classes: {num_classes}")
    
    return model

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up logging and directories
    setup_logging_and_directories(args)
    
    # Get model
    model = get_model(args)
    model = model.to('cuda')
    
    # Print model summary
    print("\nModel Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Get data loaders
    train_loader, val_loader = get_dataset(args.dataset, data_dir=args.data_dir, train=True, batch_size=args.batch)
    ood_loader = get_dataset(args.ood, data_dir=args.data_dir, train=True, batch_size=args.batch, is_ood=True)[0]
    
    print(f"\nUsing {args.ood} as OOD data")
    
    # Initialize hierarchical DRO system
    hierarchical_system = HierarchicalDROWithMultiScoring(
        model=model,
        pixel_radius=args.pixel_radius,
        feature_radius=args.feature_radius,
        pixel_weight=args.pixel_weight,
        feature_weight=args.feature_weight,
        cross_weight=args.cross_weight,
        device='cuda'
    )
    
    # Create training config
    config = {
        'mixed_precision': args.mixed_precision,
        'gradient_accumulation_steps': args.gradient_accumulation,
        'grad_checkpoint': args.grad_checkpoint,
        'empty_cache_freq': args.empty_cache_freq,
        'ood_loader': ood_loader
    }
    
    # Train the model
    hierarchical_system.train_hierarchical_dro(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        config=config,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Step 2: Setup multi-scoring framework
    print("\nSetting up multi-scoring framework...")
    # Assuming id_test_loader is defined elsewhere or needs to be created
    # For now, we'll create a dummy one if it's not available
    if 'id_test_loader' not in locals():
        id_test_loader = get_dataset(args.dataset, train=False, batch_size=args.batch, data_dir=args.data_dir)[0]
    hierarchical_system.setup_multi_scoring(id_test_loader)
    
    # Step 3: Comprehensive evaluation
    print("\nPerforming comprehensive evaluation...")
    # Assuming ood_loaders is defined elsewhere or needs to be created
    # For now, we'll create a dummy one if it's not available
    if 'ood_loaders' not in locals():
        ood_loaders = {
            'cifar100': get_dataset('cifar100', train=False, batch_size=args.batch, data_dir=args.data_dir, is_ood=True)[0],
            'svhn': get_dataset('svhn', train=False, batch_size=args.batch, data_dir=args.data_dir, is_ood=True)[0],
            'imagenet_a': get_dataset('imagenet_a', train=False, batch_size=args.batch, data_dir=args.data_dir, is_ood=True)[0],
            'imagenet_r': get_dataset('imagenet_r', train=False, batch_size=args.batch, data_dir=args.data_dir, is_ood=True)[0],
            'textures': get_dataset('textures', train=False, batch_size=args.batch, data_dir=args.data_dir, is_ood=True)[0]
        }
    results = hierarchical_system.comprehensive_evaluation(
        id_test_loader, 
        ood_loaders,
        methods=['energy', 'mahalanobis', 'msp', 'odin', 'ensemble']
    )
    
    # Save results
    save_dir = Path(f"{args.output_dir}/{args.dataset}_{args.backbone}_{args.ood}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {save_dir}/results.json")

if __name__ == "__main__":
    main()