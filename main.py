import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import os
import time
from tqdm import tqdm
import timm  # Add timm import

# For PyTorch-OOD models
from pytorch_ood.detector import EnergyBased, KLMatching, Mahalanobis, ODIN
from pytorch_ood.model import WideResNet

class LatentDROAnalyzer(nn.Module):
    """
    End-to-end latent feature analysis using DRO and outlier exposure
    with multiple backbone models from PyTorch-OOD
    """
    def __init__(
        self,
        model_name='wideresnet',  # Options: 'wideresnet', 'densenet', 'vit'
        feature_dim=None,
        num_classes=10,
        epsilon=0.5,
        alpha=0.1,   # DRO weight
        beta=0.2,    # Latent regularization weight
        device=None,
        results_dir='results/latent_dro'
    ):
        super().__init__()  # Initialize the parent nn.Module class
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize model
        self.model, self.feature_dim = self._init_model(model_name, num_classes)
        print(f"Using {model_name} with feature dimension: {self.feature_dim}")
        
        # For different OOD detectors
        self.energy_detector = EnergyBased(self.model)
        
        # For K+1 formulation
        self.kplus1_classifier = nn.Linear(self.feature_dim, num_classes + 1).to(self.device)
        
        # DRO parameters
        self.register_buffer('theta', torch.zeros(self.feature_dim, device=self.device))
        self.register_buffer('bias', torch.zeros(1, device=self.device))
        
        # Statistics for feature normalization
        self.register_buffer('feature_mean', torch.zeros(self.feature_dim, device=self.device))
        self.register_buffer('feature_std', torch.ones(self.feature_dim, device=self.device))
        
        # Latent feature bank
        self.id_features_bank = None
        self.ood_features_bank = None
        
        # Flag for fitted state
        self._is_fitted = False
    
    def _init_model(self, model_name, num_classes):
        """Initialize model from PyTorch-OOD"""
        if model_name == 'wideresnet':
            # WideResNet is a good default choice for OOD detection
            model = WideResNet(num_classes=num_classes, pretrained=None).to(self.device)
            feature_dim = model.fc.in_features if hasattr(model, 'fc') else 640  # Default for WRN
        
        elif model_name == 'densenet':
            # DenseNet has strong feature representations
            model = DenseNet(num_classes=num_classes, pretrained=None).to(self.device)
            feature_dim = model.classifier.in_features if hasattr(model, 'classifier') else 1920
        
        elif model_name == 'vit':
            # Vision Transformer for modern architecture comparison
            model = VisionTransformer(num_classes=num_classes, pretrained=None).to(self.device)
            feature_dim = model.head.in_features if hasattr(model, 'head') else 768
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        return model, feature_dim
    
    def extract_features(self, x):
        """Extract features from the backbone model"""
        self.model.eval()  # Ensure eval mode for feature extraction
        
        with torch.no_grad():
            # Different extraction methods based on model type
            if self.model_name == 'wideresnet':
                # WideResNet specific method
                if hasattr(self.model, 'forward_feature'):
                    features = self.model.forward_feature(x)
                else:
                    # Fallback to hook method for feature extraction
                    features = self._extract_via_hook(x)
                
            elif self.model_name == 'densenet':
                # DenseNet features extraction
                if hasattr(self.model, 'features'):
                    x = self.model.features(x)
                    x = F.relu(x, inplace=True)
                    x = F.adaptive_avg_pool2d(x, (1, 1))
                    features = torch.flatten(x, 1)
                else:
                    features = self._extract_via_hook(x)
                    
            elif self.model_name == 'vit':
                # Vision Transformer feature extraction
                if hasattr(self.model, 'forward_features'):
                    features = self.model.forward_features(x)
                else:
                    features = self._extract_via_hook(x)
            
            else:
                features = self._extract_via_hook(x)
        
        return features
    
    def _extract_via_hook(self, x):
        """Extract features using forward hook as fallback method"""
        features = []
        
        def hook(module, input, output):
            features.append(output.detach())
        
        # Find the layer before the final classification layer
        if hasattr(self.model, 'fc'):
            # For ResNet-like models
            handle = self.model.fc.register_forward_hook(hook)
            _ = self.model(x)
            handle.remove()
            return features[0]
        
        elif hasattr(self.model, 'classifier'):
            # For DenseNet-like models
            handle = self.model.classifier.register_forward_hook(hook)
            _ = self.model(x)
            handle.remove()
            return features[0]
        
        elif hasattr(self.model, 'head'):
            # For ViT-like models
            handle = self.model.head.register_forward_hook(hook)
            _ = self.model(x)
            handle.remove()
            return features[0]
        
        else:
            # Last resort: use the second-to-last module
            penultimate_layer = list(self.model.modules())[-2]
            handle = penultimate_layer.register_forward_hook(hook)
            _ = self.model(x)
            handle.remove()
            return features[0]
    
    def normalize_features(self, features):
        """Normalize features with stored statistics"""
        return (features - self.feature_mean) / (self.feature_std + 1e-8)
    
    def fit_dro_boundary(self, id_loader, ood_loader=None, max_samples=10000):
        """
        Fit the DRO decision boundary for OOD detection
        
        This establishes the initial decision boundary for later refinement
        """
        print("Fitting DRO boundary...")
        self.model.eval()
        
        # 1. Collect ID features
        id_features = self._collect_features(id_loader, max_samples, "ID")
        
        # 2. Collect OOD features if available
        ood_features = None
        if ood_loader is not None:
            ood_features = self._collect_features(ood_loader, max_samples // 5, "OOD")
        
        # 3. Compute feature statistics from ID data
        self.feature_mean = id_features.mean(0)
        self.feature_std = id_features.std(0) + 1e-8
        
        # 4. Normalize features
        norm_id_features = self.normalize_features(id_features)
        
        # Store feature banks for later analysis
        self.id_features_bank = norm_id_features
        
        if ood_features is not None:
            norm_ood_features = self.normalize_features(ood_features)
            self.ood_features_bank = norm_ood_features
            
            # 5. Fit decision boundary using SVM-like approach
            print("Fitting boundary with ID and OOD data...")
            from sklearn.svm import LinearSVC
            
            # Prepare data for classifier
            X = torch.cat([norm_id_features, norm_ood_features], dim=0).cpu().numpy()
            y = np.concatenate([np.zeros(len(norm_id_features)), np.ones(len(norm_ood_features))])
            
            # Train linear classifier
            try:
                svm = LinearSVC(C=1.0, class_weight='balanced')
                svm.fit(X, y)
                
                # Extract parameters
                self.theta = torch.tensor(svm.coef_[0], dtype=torch.float32, device=self.device)
                self.bias = torch.tensor(svm.intercept_[0], dtype=torch.float32, device=self.device)
                print(f"SVM fit successful with intercept {self.bias.item():.4f}")
            except Exception as e:
                print(f"SVM fitting failed: {e}, falling back to PCA direction")
                # Use PCA as fallback
                self._fit_with_pca(norm_id_features, norm_ood_features)
        else:
            # Without OOD data, use PCA to find principal direction
            print("No OOD data available, using PCA for boundary...")
            self._fit_with_pca(norm_id_features)
        
        self._is_fitted = True
        
        # 6. Visualize the initial feature space
        self.visualize_feature_space(
            norm_id_features.cpu().numpy(), 
            None if ood_features is None else norm_ood_features.cpu().numpy(),
            save_path=os.path.join(self.results_dir, "initial_feature_space.png")
        )
        
        return self
    
    def _collect_features(self, loader, max_samples, dataset_type=""):
        """Helper method to collect features from a dataloader"""
        features_list = []
        samples_collected = 0
        
        with torch.no_grad():
            for x, _ in tqdm(loader, desc=f"Collecting {dataset_type} features"):
                x = x.to(self.device)
                batch_features = self.extract_features(x)
                features_list.append(batch_features.cpu())
                
                samples_collected += x.size(0)
                if samples_collected >= max_samples:
                    break
        
        return torch.cat(features_list, dim=0).to(self.device)
    
    def _fit_with_pca(self, id_features, ood_features=None):
        """Fit boundary using PCA direction"""
        # Start with ID features centered
        centered_features = id_features - id_features.mean(0)
        
        # Use torch's PCA implementation
        U, S, V = torch.pca_lowrank(centered_features, q=1)
        
        # First principal component as direction
        self.theta = V[:, 0]
        
        # Determine decision boundary location
        if ood_features is not None:
            # Project both ID and OOD to the direction
            id_proj = torch.matmul(id_features, self.theta)
            ood_proj = torch.matmul(ood_features, self.theta)
            
            # Check which side has more OOD samples
            id_mean = id_proj.mean().item()
            ood_mean = ood_proj.mean().item()
            
            if id_mean > ood_mean:
                # ID samples have higher projection, so we want θᵀx + b < 0 for ID
                # Set bias based on percentile of ID projections
                percentile_95 = torch.quantile(id_proj, 0.95)
                self.bias = -percentile_95 - 0.1  # Small margin
            else:
                # OOD samples have higher projection, so we want θᵀx + b > 0 for ID
                # Set bias based on percentile of ID projections
                percentile_05 = torch.quantile(id_proj, 0.05)
                self.bias = -percentile_05 + 0.1  # Small margin
        else:
            # Without OOD samples, choose a threshold that classifies 95% of ID samples as ID
            id_proj = torch.matmul(id_features, self.theta)
            percentile_95 = torch.quantile(id_proj, 0.95)
            self.bias = -percentile_95 - 0.1  # Small margin
        
        print(f"PCA direction found with bias {self.bias.item():.4f}")
    
    def visualize_feature_space(self, id_features, ood_features=None, method='tsne', 
                               perplexity=30, save_path=None):
        """
        Visualize the feature space using dimensionality reduction
        
        Args:
            id_features: Numpy array of ID features
            ood_features: Numpy array of OOD features (optional)
            method: 'tsne' or 'pca'
            perplexity: Perplexity parameter for t-SNE
            save_path: Path to save the visualization
        """
        print(f"Visualizing feature space using {method}...")
        
        # Convert to numpy if tensors
        if isinstance(id_features, torch.Tensor):
            id_features = id_features.cpu().numpy()
        
        if ood_features is not None and isinstance(ood_features, torch.Tensor):
            ood_features = ood_features.cpu().numpy()
        
        # Combine features for dimensionality reduction
        if ood_features is not None:
            combined_features = np.vstack([id_features, ood_features])
            combined_labels = np.concatenate([
                np.zeros(len(id_features)),
                np.ones(len(ood_features))
            ])
        else:
            combined_features = id_features
            combined_labels = np.zeros(len(id_features))
        
        # Apply dimensionality reduction
        if method == 'tsne':
            embedding = TSNE(n_components=2, perplexity=perplexity, 
                             random_state=42).fit_transform(combined_features)
        else:  # PCA
            embedding = PCA(n_components=2).fit_transform(combined_features)
        
        # Plot the embedding
        plt.figure(figsize=(10, 8))
        
        # ID samples
        id_mask = combined_labels == 0
        plt.scatter(embedding[id_mask, 0], embedding[id_mask, 1], 
                    c='blue', label='ID', alpha=0.5, s=10)
        
        # OOD samples if available
        if ood_features is not None:
            ood_mask = combined_labels == 1
            plt.scatter(embedding[ood_mask, 0], embedding[ood_mask, 1], 
                        c='red', label='OOD', alpha=0.5, s=10)
        
        # Add details
        plt.title(f'Latent Feature Space Visualization using {method.upper()}')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def apply_dro_perturbation(self, features):
        """
        Apply DRO perturbation to features
        
        This creates worst-case adversarial perturbations in feature space
        according to the DRO principle
        """
        if not self._is_fitted:
            raise RuntimeError("Must fit DRO boundary first")
        
        # Normalize feature direction for consistent perturbation magnitude
        direction = self.theta / (torch.norm(self.theta) + 1e-8)
        
        # Apply perturbation along the decision boundary direction
        perturbed_features = features + self.epsilon * direction.unsqueeze(0)
        
        return perturbed_features
    
    def forward(self, x, apply_dro=False, return_features=False):
        """
        Forward pass for both standard classification and OOD detection
        
        Args:
            x: Input images
            apply_dro: Whether to apply DRO perturbation
            return_features: Whether to return extracted features
        
        Returns:
            tuple of (class_logits, kplus1_logits, dro_scores, features)
        """
        # Extract features
        features = self.extract_features(x)
        
        # Normalize features
        norm_features = self.normalize_features(features)
        
        # Apply DRO perturbation if requested
        if apply_dro and self._is_fitted:
            perturbed_features = self.apply_dro_perturbation(norm_features)
        else:
            perturbed_features = norm_features
        
        # Standard classification (K classes)
        # Replace features_to_logits with direct use of model.fc
        if hasattr(self.model, 'fc'):
            class_logits = self.model.fc(perturbed_features)
        elif hasattr(self.model, 'classifier'):
            class_logits = self.model.classifier(perturbed_features)
        elif hasattr(self.model, 'head'):
            class_logits = self.model.head(perturbed_features)
        else:
            raise AttributeError("Model doesn't have a recognized classification layer")
        
        # K+1 classification for OOD detection
        kplus1_logits = self.kplus1_classifier(perturbed_features)
        
        # DRO score for OOD detection
        dro_scores = None
        if self._is_fitted:
            dro_scores = torch.matmul(perturbed_features, self.theta) + self.bias
        
        if return_features:
            return class_logits, kplus1_logits, dro_scores, perturbed_features
        else:
            return class_logits, kplus1_logits, dro_scores
    
    def compute_dro_loss(self, features, labels):
        """
        Compute DRO loss based on worst-case perturbations
        
        Args:
            features: Normalized feature tensors
            labels: Class labels where num_classes = OOD class
        """
        if not self._is_fitted:
            return torch.tensor(0.0, device=self.device)
        
        # Identify ID vs OOD samples in the batch
        id_mask = labels < self.num_classes
        ood_mask = ~id_mask
        
        # Skip if all samples are of the same type
        if not torch.any(id_mask) or not torch.any(ood_mask):
            return torch.tensor(0.0, device=self.device)
        
        # Get normalized direction
        direction = self.theta / (torch.norm(self.theta) + 1e-8)
        
        # Create worst-case perturbations for ID samples (push toward OOD)
        id_features = features[id_mask]
        id_perturbed = id_features + self.epsilon * direction.unsqueeze(0)
        
        # Create worst-case perturbations for OOD samples (push toward ID)
        ood_features = features[ood_mask]
        ood_perturbed = ood_features - self.epsilon * direction.unsqueeze(0)
        
        # Compute DRO scores
        id_scores = torch.matmul(id_perturbed, self.theta) + self.bias
        ood_scores = torch.matmul(ood_perturbed, self.theta) + self.bias
        
        # Robust margin loss: ensure ID scores < 0 and OOD scores > 0
        # with a margin in between
        margin = 1.0
        dro_loss = F.relu(margin + id_scores).mean() + F.relu(margin - ood_scores).mean()
        
        return dro_loss
    
    def compute_latent_contrastive_loss(self, features, labels):
        """
        Compute contrastive loss to improve latent space structure
        
        Args:
            features: Normalized feature tensors
            labels: Class labels where num_classes = OOD class
        """
        # Identify ID vs OOD samples
        id_mask = labels < self.num_classes
        ood_mask = ~id_mask
        
        # Skip if all samples are of the same type
        if not torch.any(id_mask) or not torch.any(ood_mask):
            return torch.tensor(0.0, device=self.device)
        
        # Separate features
        id_features = features[id_mask]
        ood_features = features[ood_mask]
        
        # Compute pairwise distances within and between groups
        # 1. Within-class similarity for ID samples
        id_norm = F.normalize(id_features, dim=1)
        id_similarity = torch.matmul(id_norm, id_norm.t())
        
        # 2. Between-class dissimilarity (ID vs OOD)
        ood_norm = F.normalize(ood_features, dim=1)
        between_similarity = torch.matmul(id_norm, ood_norm.t())
        
        # Temperature parameter for contrastive loss
        tau = 0.1
        
        # Contrastive loss: maximize similarity within class, minimize between classes
        # Similar to NT-Xent loss
        within_term = -torch.log(torch.exp(id_similarity / tau).mean())
        between_term = -torch.log(1 - torch.exp(between_similarity / tau).mean())
        
        contrastive_loss = within_term + between_term
        return contrastive_loss
    
    def train_model(self, id_loader, ood_loader, val_id_loader=None, val_ood_loader=None,
                  num_epochs=100, lr=0.001, weight_decay=5e-4, use_scheduler=True):
        """
        Train the model with DRO regularization and K+1 formulation
        
        Args:
            id_loader: DataLoader for ID training samples
            ood_loader: DataLoader for OOD training samples
            val_id_loader: DataLoader for ID validation (optional)
            val_ood_loader: DataLoader for OOD validation (optional)
            num_epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay coefficient
            use_scheduler: Whether to use learning rate scheduler
        """
        print(f"Training model for {num_epochs} epochs...")
        
        # First fit DRO boundary if not already done
        if not self._is_fitted:
            self.fit_dro_boundary(id_loader, ood_loader)
            
        # Setup model for training
        self.model.train()
        
        # Optimizers
        params = list(self.model.parameters()) + list(self.kplus1_classifier.parameters())
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        
        # Scheduler
        scheduler = None
        if use_scheduler:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Training loop
        best_val_auroc = 0.0
        train_metrics = {'epoch': [], 'id_acc': [], 'loss': [], 'ce_loss': [], 
                        'dro_loss': [], 'latent_loss': []}
        val_metrics = {'epoch': [], 'id_acc': [], 'auroc': [], 'aupr': []}
        
        # Create iterators for data loaders
        id_iterator = iter(id_loader)
        ood_iterator = iter(ood_loader)
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training metrics
            epoch_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_dro_loss = 0.0
            epoch_latent_loss = 0.0
            id_correct = 0
            id_total = 0
            batch_count = 0
            
            # Number of batches
            num_batches = min(len(id_loader), len(ood_loader) * 5)  # Sample OOD less frequently
            
            for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Get ID batch
                try:
                    id_x, id_y = next(id_iterator)
                except StopIteration:
                    id_iterator = iter(id_loader)
                    id_x, id_y = next(id_iterator)
                
                id_x, id_y = id_x.to(self.device), id_y.to(self.device)
                
                # Get OOD batch every 5th iteration
                if i % 5 == 0:
                    try:
                        ood_x, _ = next(ood_iterator)
                    except StopIteration:
                        ood_iterator = iter(ood_loader)
                        ood_x, _ = next(ood_iterator)
                    
                    ood_x = ood_x.to(self.device)
                    # Label as K for K+1 formulation
                    ood_y = torch.full((ood_x.size(0),), self.num_classes, 
                                     dtype=torch.long, device=self.device)
                    
                    # Combine ID and OOD data
                    x = torch.cat([id_x, ood_x], dim=0)
                    y = torch.cat([id_y, ood_y], dim=0)
                else:
                    # Use only ID data for most iterations
                    x, y = id_x, id_y
                
                # Forward pass with DRO perturbation
                _, kplus1_logits, _, features = self.forward(x, apply_dro=True, return_features=True)
                
                # Cross-entropy loss for K+1 classification
                ce_loss = F.cross_entropy(kplus1_logits, y)
                
                # DRO loss
                dro_loss = self.compute_dro_loss(features, y)
                
                # Latent contrastive loss
                latent_loss = self.compute_latent_contrastive_loss(features, y)
                
                # Total loss
                loss = ce_loss + self.alpha * dro_loss + self.beta * latent_loss
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_dro_loss += dro_loss.item()
                epoch_latent_loss += latent_loss.item()
                batch_count += 1
                
                # ID classification accuracy (only for ID samples)
                id_mask = y < self.num_classes
                if torch.any(id_mask):
                    _, predicted = kplus1_logits[id_mask, :self.num_classes].max(1)
                    id_correct += (predicted == y[id_mask]).sum().item()
                    id_total += id_mask.sum().item()
            
            # Average training metrics
            avg_loss = epoch_loss / batch_count
            avg_ce_loss = epoch_ce_loss / batch_count
            avg_dro_loss = epoch_dro_loss / batch_count
            avg_latent_loss = epoch_latent_loss / batch_count
            id_accuracy = 100 * id_correct / max(id_total, 1)
            
            # Update learning rate
            if scheduler:
                scheduler.step()
            
            # Validation if provided
            val_id_acc = 0.0
            val_auroc = 0.0
            val_aupr = 0.0
            
            if val_id_loader is not None and val_ood_loader is not None:
                val_metrics_dict = self.evaluate(val_id_loader, val_ood_loader)
                val_id_acc = val_metrics_dict['id_accuracy']
                val_auroc = val_metrics_dict['auroc']
                val_aupr = val_metrics_dict['aupr']
                
                # Save best model
                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'kplus1_state_dict': self.kplus1_classifier.state_dict(),
                        'theta': self.theta,
                        'bias': self.bias,
                        'feature_mean': self.feature_mean,
                        'feature_std': self.feature_std,
                        'best_auroc': best_val_auroc
                    }, os.path.join(self.results_dir, 'best_model.pth'))
                    print(f"✓ Best model saved with AUROC: {best_val_auroc:.4f}")
                
                # Visualize feature space during training
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    # Collect features for visualization
                    id_features = self._collect_features(
                        val_id_loader, max_samples=1000, dataset_type="val ID")
                    ood_features = self._collect_features(
                        val_ood_loader, max_samples=1000, dataset_type="val OOD")
                    
                    # Normalize
                    norm_id_features = self.normalize_features(id_features)
                    norm_ood_features = self.normalize_features(ood_features)
                    
                    # Visualize
                    self.visualize_feature_space(
                        norm_id_features.cpu().numpy(),
                        norm_ood_features.cpu().numpy(),
                        save_path=os.path.join(self.results_dir, f"feature_space_epoch_{epoch+1}.png")
                    )
            
            # Record metrics
            train_metrics['epoch'].append(epoch + 1)
            train_metrics['id_acc'].append(id_accuracy)
            train_metrics['loss'].append(avg_loss)
            train_metrics['ce_loss'].append(avg_ce_loss)
            train_metrics['dro_loss'].append(avg_dro_loss)
            train_metrics['latent_loss'].append(avg_latent_loss)
            
            if val_id_loader is not None:
                val_metrics['epoch'].append(epoch + 1)
                val_metrics['id_acc'].append(val_id_acc)
                val_metrics['auroc'].append(val_auroc)
                val_metrics['aupr'].append(val_aupr)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.1f}s")
            print(f"Train - Loss: {avg_loss:.4f} (CE: {avg_ce_loss:.4f}, "
                  f"DRO: {avg_dro_loss:.4f}, Latent: {avg_latent_loss:.4f})")
            print(f"Train - ID Accuracy: {id_accuracy:.2f}%")
            
            if val_id_loader is not None:
                print(f"Val - ID Accuracy: {val_id_acc:.2f}%")
                print(f"Val - AUROC: {val_auroc:.4f}, AUPR: {val_aupr:.4f}")
            
            print("-" * 80)
        
        # Plot training curves
        self._plot_training_curves(train_metrics, val_metrics)
        
        return self
    
    def _plot_training_curves(self, train_metrics, val_metrics):
        """Plot and save training curves"""
        # Loss curves
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(train_metrics['epoch'], train_metrics['loss'], 'b-', label='Total Loss')
        plt.plot(train_metrics['epoch'], train_metrics['ce_loss'], 'g-', label='CE Loss')
        plt.plot(train_metrics['epoch'], train_metrics['dro_loss'], 'r-', label='DRO Loss')
        plt.plot(train_metrics['epoch'], train_metrics['latent_loss'], 'c-', label='Latent Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy
        plt.subplot(2, 2, 2)
        plt.plot(train_metrics['epoch'], train_metrics['id_acc'], 'b-', label='Train ID Acc')
        if val_metrics and len(val_metrics['epoch']) > 0:
            plt.plot(val_metrics['epoch'], val_metrics['id_acc'], 'r-', label='Val ID Acc')
        plt.title('ID Classification Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # AUROC and AUPR
        if val_metrics and len(val_metrics['epoch']) > 0:
            plt.subplot(2, 2, 3)
            plt.plot(val_metrics['epoch'], val_metrics['auroc'], 'g-', label='AUROC')
            plt.plot(val_metrics['epoch'], val_metrics['aupr'], 'm-', label='AUPR')
            plt.title('OOD Detection Performance')
            plt.xlabel('Epoch')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_curves.png'))
        plt.close()
    
    def evaluate(self, id_loader, ood_loader, save_results=False):
        """
        Evaluate OOD detection performance
        
        Args:
            id_loader: DataLoader with ID test samples
            ood_loader: DataLoader with OOD test samples
            save_results: Whether to save detailed results
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("Evaluating OOD detection performance...")
        self.model.eval()
        
        # Collect scores
        id_dro_scores = []
        id_energy_scores = []
        id_kplus1_scores = []
        id_correct = 0
        id_total = 0
        
        ood_dro_scores = []
        ood_energy_scores = []
        ood_kplus1_scores = []
        
        # Process ID data
        with torch.no_grad():
            for x, y in tqdm(id_loader, desc="Evaluating ID data"):
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                class_logits, kplus1_logits, dro_scores = self.forward(x)
                
                # Classification accuracy
                _, predicted = class_logits.max(1)
                id_correct += (predicted == y).sum().item()
                id_total += y.size(0)
                
                # Energy score
                energy_score = self.energy_detector(class_logits)
                
                # K+1 OOD score (softmax probability of the K+1 class)
                kplus1_prob = F.softmax(kplus1_logits, dim=1)[:, -1]
                
                # Store scores
                if dro_scores is not None:
                    id_dro_scores.extend(dro_scores.cpu().numpy())
                id_energy_scores.extend(energy_score.cpu().numpy())
                id_kplus1_scores.extend(kplus1_prob.cpu().numpy())
        
        # Process OOD data
        with torch.no_grad():
            for x, _ in tqdm(ood_loader, desc="Evaluating OOD data"):
                x = x.to(self.device)
                
                # Forward pass
                class_logits, kplus1_logits, dro_scores = self.forward(x)
                
                # Energy score
                energy_score = self.energy_detector(class_logits)
                
                # K+1 OOD score
                kplus1_prob = F.softmax(kplus1_logits, dim=1)[:, -1]
                
                # Store scores
                if dro_scores is not None:
                    ood_dro_scores.extend(dro_scores.cpu().numpy())
                ood_energy_scores.extend(energy_score.cpu().numpy())
                ood_kplus1_scores.extend(kplus1_prob.cpu().numpy())
        
        # Convert to numpy arrays
        id_energy_scores = np.array(id_energy_scores)
        ood_energy_scores = np.array(ood_energy_scores)
        id_kplus1_scores = np.array(id_kplus1_scores)
        ood_kplus1_scores = np.array(ood_kplus1_scores)
        
        # Compute metrics for different methods
        results = {}
        
        # ID classification accuracy
        id_accuracy = 100 * id_correct / id_total
        results['id_accuracy'] = id_accuracy
        print(f"ID Classification Accuracy: {id_accuracy:.2f}%")
        
        # 1. Energy-based OOD detection
        auroc_energy = roc_auc_score(
            np.concatenate([np.zeros_like(id_energy_scores), np.ones_like(ood_energy_scores)]),
            np.concatenate([id_energy_scores, ood_energy_scores])
        )
        results['auroc_energy'] = auroc_energy
        print(f"Energy-based AUROC: {auroc_energy:.4f}")
        
        # 2. K+1 OOD detection
        auroc_kplus1 = roc_auc_score(
            np.concatenate([np.zeros_like(id_kplus1_scores), np.ones_like(ood_kplus1_scores)]),
            np.concatenate([id_kplus1_scores, ood_kplus1_scores])
        )
        results['auroc_kplus1'] = auroc_kplus1
        print(f"K+1 AUROC: {auroc_kplus1:.4f}")
        
        # 3. DRO OOD detection (if available)
        if len(id_dro_scores) > 0 and len(ood_dro_scores) > 0:
            id_dro_scores = np.array(id_dro_scores)
            ood_dro_scores = np.array(ood_dro_scores)
            
            auroc_dro = roc_auc_score(
                np.concatenate([np.zeros_like(id_dro_scores), np.ones_like(ood_dro_scores)]),
                np.concatenate([id_dro_scores, ood_dro_scores])
            )
            results['auroc_dro'] = auroc_dro
            print(f"DRO AUROC: {auroc_dro:.4f}")
            
            # Compute AUPR
            precision, recall, _ = precision_recall_curve(
                np.concatenate([np.zeros_like(id_dro_scores), np.ones_like(ood_dro_scores)]),
                np.concatenate([id_dro_scores, ood_dro_scores])
            )
            aupr_dro = auc(recall, precision)
            results['aupr'] = aupr_dro
            print(f"DRO AUPR: {aupr_dro:.4f}")
            
            # Ensemble score (combine DRO and K+1)
            # Normalize scores before combining
            id_dro_norm = (id_dro_scores - id_dro_scores.min()) / (id_dro_scores.max() - id_dro_scores.min())
            ood_dro_norm = (ood_dro_scores - id_dro_scores.min()) / (id_dro_scores.max() - id_dro_scores.min())
            
            id_ensemble = 0.5 * id_dro_norm + 0.5 * id_kplus1_scores
            ood_ensemble = 0.5 * ood_dro_norm + 0.5 * ood_kplus1_scores
            
            auroc_ensemble = roc_auc_score(
                np.concatenate([np.zeros_like(id_ensemble), np.ones_like(ood_ensemble)]),
                np.concatenate([id_ensemble, ood_ensemble])
            )
            results['auroc_ensemble'] = auroc_ensemble
            print(f"Ensemble AUROC: {auroc_ensemble:.4f}")
            
            # Use the best performing method
            results['auroc'] = max(auroc_dro, auroc_kplus1, auroc_ensemble)
        else:
            # If DRO not available, use K+1
            results['auroc'] = auroc_kplus1
            
            # Compute AUPR
            precision, recall, _ = precision_recall_curve(
                np.concatenate([np.zeros_like(id_kplus1_scores), np.ones_like(ood_kplus1_scores)]),
                np.concatenate([id_kplus1_scores, ood_kplus1_scores])
            )
            aupr_kplus1 = auc(recall, precision)
            results['aupr'] = aupr_kplus1
        
        # Save detailed results if requested
        if save_results:
            save_dict = {
                'id_energy_scores': id_energy_scores,
                'ood_energy_scores': ood_energy_scores,
                'id_kplus1_scores': id_kplus1_scores,
                'ood_kplus1_scores': ood_kplus1_scores
            }
            
            if len(id_dro_scores) > 0:
                save_dict['id_dro_scores'] = id_dro_scores
                save_dict['ood_dro_scores'] = ood_dro_scores
            
            np.savez(os.path.join(self.results_dir, 'evaluation_scores.npz'), **save_dict)
            
            # Plot score distributions
            self._plot_score_distributions(
                id_dro_scores if len(id_dro_scores) > 0 else id_kplus1_scores,
                ood_dro_scores if len(ood_dro_scores) > 0 else ood_kplus1_scores,
                score_type='DRO' if len(id_dro_scores) > 0 else 'K+1',
                save_path=os.path.join(self.results_dir, 'score_distributions.png')
            )
            
            # Plot ROC curves
            self._plot_roc_curves(results, 
                                 id_scores={
                                     'Energy': id_energy_scores,
                                     'K+1': id_kplus1_scores,
                                     'DRO': id_dro_scores if len(id_dro_scores) > 0 else None
                                 },
                                 ood_scores={
                                     'Energy': ood_energy_scores,
                                     'K+1': ood_kplus1_scores,
                                     'DRO': ood_dro_scores if len(ood_dro_scores) > 0 else None
                                 },
                                 save_path=os.path.join(self.results_dir, 'roc_curves.png'))
        
        return results
    
    def _plot_score_distributions(self, id_scores, ood_scores, score_type='DRO', save_path=None):
        """Plot and save score distributions"""
        plt.figure(figsize=(10, 6))
        
        # Plot histograms
        plt.hist(id_scores, bins=50, alpha=0.5, label='ID samples')
        plt.hist(ood_scores, bins=50, alpha=0.5, label='OOD samples')
        
        # Add means
        plt.axvline(x=np.mean(id_scores), color='blue', linestyle='--', 
                   label=f'ID mean: {np.mean(id_scores):.3f}')
        plt.axvline(x=np.mean(ood_scores), color='orange', linestyle='--', 
                   label=f'OOD mean: {np.mean(ood_scores):.3f}')
        
        # Add details
        plt.title(f'{score_type} Score Distributions')
        plt.xlabel('Score')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def _plot_roc_curves(self, results, id_scores, ood_scores, save_path=None):
        """Plot and save ROC curves for different methods"""
        plt.figure(figsize=(10, 8))
        
        methods = ['Energy', 'K+1', 'DRO']
        colors = ['green', 'blue', 'red']
        
        # Plot each method's ROC curve
        for method, color in zip(methods, colors):
            if method in id_scores and id_scores[method] is not None:
                id_method_scores = id_scores[method]
                ood_method_scores = ood_scores[method]
                
                # Create labels
                y_true = np.concatenate([
                    np.zeros_like(id_method_scores),
                    np.ones_like(ood_method_scores)
                ])
                y_score = np.concatenate([id_method_scores, ood_method_scores])
                
                # Compute ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = results.get(f'auroc_{method.lower()}', 
                                     auc(fpr, tpr))
                
                # Plot
                plt.plot(fpr, tpr, color=color, lw=2,
                        label=f'{method} (AUROC = {roc_auc:.3f})')
        
        # Add baseline
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
        
        # Add details
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Different OOD Detection Methods')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def analyze_latent_space(self, id_loader, ood_loader, max_samples=1000):
        """
        Perform comprehensive latent space analysis
        
        Args:
            id_loader: DataLoader with ID samples
            ood_loader: DataLoader with OOD samples
            max_samples: Maximum number of samples to analyze
        """
        print("Performing latent space analysis...")
        self.model.eval()
        
        # 1. Collect features
        id_features = self._collect_features(id_loader, max_samples, "ID")
        ood_features = self._collect_features(ood_loader, max_samples, "OOD")
        
        # 2. Normalize features
        norm_id_features = self.normalize_features(id_features)
        norm_ood_features = self.normalize_features(ood_features)
        
        # 3. Convert to numpy for analysis
        id_features_np = norm_id_features.cpu().numpy()
        ood_features_np = norm_ood_features.cpu().numpy()
        
        # 4. Visualize with different methods
        # t-SNE visualization
        self.visualize_feature_space(
            id_features_np, ood_features_np, method='tsne', perplexity=30,
            save_path=os.path.join(self.results_dir, 'tsne_visualization.png')
        )
        
        # PCA visualization
        self.visualize_feature_space(
            id_features_np, ood_features_np, method='pca',
            save_path=os.path.join(self.results_dir, 'pca_visualization.png')
        )
        
        # 5. Analyze feature statistics
        # Compute distance metrics
        from scipy.spatial.distance import cdist
        
        # Within-class distances
        id_id_dist = cdist(id_features_np[:100], id_features_np[100:200], 'euclidean')
        id_id_mean = np.mean(id_id_dist)
        id_id_std = np.std(id_id_dist)
        
        # Between-class distances
        id_ood_dist = cdist(id_features_np[:100], ood_features_np[:100], 'euclidean')
        id_ood_mean = np.mean(id_ood_dist)
        id_ood_std = np.std(id_ood_dist)
        
        # Print statistics
        print(f"Within-ID distance: {id_id_mean:.4f} ± {id_id_std:.4f}")
        print(f"ID-OOD distance: {id_ood_mean:.4f} ± {id_ood_std:.4f}")
        print(f"Distance ratio (ID-OOD/Within-ID): {id_ood_mean/id_id_mean:.4f}")
        
        # 6. Apply DRO perturbation and analyze
        # Get normalized direction
        direction = self.theta / (torch.norm(self.theta) + 1e-8)
        
        # Create perturbed features
        id_perturbed = norm_id_features + self.epsilon * direction.unsqueeze(0)
        ood_perturbed = norm_ood_features - self.epsilon * direction.unsqueeze(0)
        
        # Convert to numpy
        id_perturbed_np = id_perturbed.cpu().numpy()
        ood_perturbed_np = ood_perturbed.cpu().numpy()
        
        # Visualize original vs perturbed
        plt.figure(figsize=(12, 10))
        
        # Create 2D PCA for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        combined_features = np.vstack([
            id_features_np, ood_features_np,
            id_perturbed_np, ood_perturbed_np
        ])
        pca_result = pca.fit_transform(combined_features)
        
        # Separate the results
        n_id = len(id_features_np)
        n_ood = len(ood_features_np)
        
        id_pca = pca_result[:n_id]
        ood_pca = pca_result[n_id:n_id+n_ood]
        id_pert_pca = pca_result[n_id+n_ood:n_id+n_ood+n_id]
        ood_pert_pca = pca_result[n_id+n_ood+n_id:]
        
        # Plot original and perturbed samples
        plt.scatter(id_pca[:, 0], id_pca[:, 1], c='blue', label='ID', alpha=0.5, s=10)
        plt.scatter(ood_pca[:, 0], ood_pca[:, 1], c='red', label='OOD', alpha=0.5, s=10)
        plt.scatter(id_pert_pca[:, 0], id_pert_pca[:, 1], c='cyan', 
                   label='ID (perturbed)', alpha=0.5, s=10)
        plt.scatter(ood_pert_pca[:, 0], ood_pert_pca[:, 1], c='magenta', 
                   label='OOD (perturbed)', alpha=0.5, s=10)
        
        # Add DRO direction
        # Project the direction vector to 2D
        direction_2d = pca.transform(direction.cpu().numpy().reshape(1, -1))[0]
        
        # Scale for better visualization
        arrow_scale = 20
        center = np.mean(pca_result, axis=0)
        plt.arrow(center[0], center[1], 
                 arrow_scale * direction_2d[0], arrow_scale * direction_2d[1],
                 head_width=1.0, head_length=1.5, fc='black', ec='black',
                 label='DRO Direction')
        
        # Add details
        plt.title('PCA of Original and DRO-Perturbed Features')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save
        plt.savefig(os.path.join(self.results_dir, 'dro_perturbation_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Feature importance analysis
        # Get magnitude of each feature in the DRO direction
        feature_importance = self.theta.abs().cpu().numpy()
        
        # Sort and plot top features
        top_indices = np.argsort(feature_importance)[-20:]  # Top 20 features
        top_values = feature_importance[top_indices]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_indices)), top_values)
        plt.title('Top Features by DRO Direction Magnitude')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance (Absolute Coefficient Value)')
        plt.xticks(range(len(top_indices)), top_indices)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.results_dir, 'feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 8. Compute feature space metrics
        metrics = {
            'id_id_distance_mean': id_id_mean,
            'id_id_distance_std': id_id_std,
            'id_ood_distance_mean': id_ood_mean,
            'id_ood_distance_std': id_ood_std,
            'distance_ratio': id_ood_mean / id_id_mean,
            'dro_direction_norm': torch.norm(self.theta).item(),
            'top_feature_indices': top_indices.tolist(),
            'top_feature_values': top_values.tolist()
        }
        
        # Save metrics
        import json
        with open(os.path.join(self.results_dir, 'latent_space_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics


def setup_datasets(id_dataset_name='cifar10', ood_dataset_name='svhn', batch_size=128):
    """
    Setup datasets for ID and OOD data
    
    Args:
        id_dataset_name: Name of ID dataset ('cifar10' or 'cifar100')
        ood_dataset_name: Name of OOD dataset ('svhn', 'tiny-imagenet', etc.)
        batch_size: Batch size for data loaders
        
    Returns:
        Dictionary containing data loaders
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load ID dataset
    if id_dataset_name.lower() == 'cifar10':
        id_train_dataset = torchvision.datasets.CIFAR10(
            root='/home/tanmoy/research/data', train=True, download=True, transform=train_transform)
        id_test_dataset = torchvision.datasets.CIFAR10(
            root='/home/tanmoy/research/data', train=False, download=True, transform=test_transform)
        num_classes = 10
    elif id_dataset_name.lower() == 'cifar100':
        id_train_dataset = torchvision.datasets.CIFAR100(
            root='/home/tanmoy/research/data', train=True, download=True, transform=train_transform)
        id_test_dataset = torchvision.datasets.CIFAR100(
            root='/home/tanmoy/research/data', train=False, download=True, transform=test_transform)
        num_classes = 100
    else:
        raise ValueError(f"Unknown ID dataset name: {id_dataset_name}")
    
    # Split training data for validation
    from torch.utils.data import random_split
    val_size = int(0.1 * len(id_train_dataset))
    train_size = len(id_train_dataset) - val_size
    id_train_dataset, id_val_dataset = random_split(
        id_train_dataset, [train_size, val_size])
    
    # Load OOD dataset
    if ood_dataset_name.lower() == 'svhn':
        ood_train_dataset = torchvision.datasets.SVHN(
            root='/home/tanmoy/research/data', split='train', download=True, transform=test_transform)
        ood_test_dataset = torchvision.datasets.SVHN(
            root='/home/tanmoy/research/data', split='test', download=True, transform=test_transform)
    elif ood_dataset_name.lower() == 'tiny-imagenet':
        # This dataset needs to be downloaded manually
        # and requires different transformations for 64x64 images
        tiny_transform = transforms.Compose([
            transforms.Resize(32),  # Resize to match CIFAR
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        ood_train_dataset = torchvision.datasets.ImageFolder(
            root='/home/tanmoy/research/data/tiny-imagenet-200/train', transform=tiny_transform)
        ood_test_dataset = torchvision.datasets.ImageFolder(
            root='/home/tanmoy/research/data/tiny-imagenet-200/val', transform=tiny_transform)
    else:
        raise ValueError(f"Unknown OOD dataset name: {ood_dataset_name}")
    
    # Split OOD data for validation
    val_size = int(0.1 * len(ood_train_dataset))
    train_size = len(ood_train_dataset) - val_size
    ood_train_dataset, ood_val_dataset = random_split(
        ood_train_dataset, [train_size, val_size])
    
    # Create data loaders
    id_train_loader = DataLoader(
        id_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    id_val_loader = DataLoader(
        id_val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    id_test_loader = DataLoader(
        id_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    ood_train_loader = DataLoader(
        ood_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    ood_val_loader = DataLoader(
        ood_val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    ood_test_loader = DataLoader(
        ood_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return {
        'id_train_loader': id_train_loader,
        'id_val_loader': id_val_loader,
        'id_test_loader': id_test_loader,
        'ood_train_loader': ood_train_loader,
        'ood_val_loader': ood_val_loader,
        'ood_test_loader': ood_test_loader,
        'num_classes': num_classes
    }


def run_latent_dro_analysis():
    """Run the complete latent DRO analysis pipeline"""
    # Setup paths and device
    results_dir = 'results/latent_dro'
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup datasets
    data_loaders = setup_datasets(
        id_dataset_name='cifar10',
        ood_dataset_name='svhn',
        batch_size=128
    )
    
    # Create analyzer with WideResNet
    model_results = {}
    
    models_to_test = ['wideresnet', 'densenet']  # Add 'vit' if available
    
    for model_name in models_to_test:
        print(f"\n{'='*80}\nTesting model: {model_name}\n{'='*80}")
        
        # Create model-specific results directory
        model_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize analyzer
        analyzer = LatentDROAnalyzer(
            model_name=model_name,
            num_classes=data_loaders['num_classes'],
            epsilon=0.5,
            alpha=0.1,  # DRO weight
            beta=0.2,   # Latent regularization weight
            device=device,
            results_dir=model_dir
        )
        
        # Fit DRO boundary for initial analysis
        analyzer.fit_dro_boundary(
            data_loaders['id_train_loader'],
            data_loaders['ood_train_loader']
        )
        
        # Train with DRO regularization
        analyzer.train_model(
            id_loader=data_loaders['id_train_loader'],
            ood_loader=data_loaders['ood_train_loader'],
            val_id_loader=data_loaders['id_val_loader'],
            val_ood_loader=data_loaders['ood_val_loader'],
            num_epochs=50,  # Reduce for faster testing
            lr=0.001,
            weight_decay=5e-4
        )
        
        # Evaluate on test data
        test_results = analyzer.evaluate(
            data_loaders['id_test_loader'],
            data_loaders['ood_test_loader'],
            save_results=True
        )
        
        # Perform latent space analysis
        latent_metrics = analyzer.analyze_latent_space(
            data_loaders['id_test_loader'],
            data_loaders['ood_test_loader']
        )
        
        # Store results
        model_results[model_name] = {
            'test_results': test_results,
            'latent_metrics': latent_metrics
        }
    
    # Comparative analysis across models
    print("\n" + "="*80)
    print("Comparative Analysis of Models")
    print("="*80)
    
    # Create comparison table
    comparison = {
        'Model': [],
        'ID Acc': [],
        'AUROC': [],
        'AUPR': [],
        'ID-OOD Dist Ratio': []
    }
    
    for model_name, results in model_results.items():
        comparison['Model'].append(model_name)
        comparison['ID Acc'].append(f"{results['test_results']['id_accuracy']:.2f}%")
        comparison['AUROC'].append(f"{results['test_results']['auroc']:.4f}")
        comparison['AUPR'].append(f"{results['test_results']['aupr']:.4f}")
        comparison['ID-OOD Dist Ratio'].append(
            f"{results['latent_metrics']['distance_ratio']:.2f}"
        )
    
    # Print comparison
    from tabulate import tabulate
    print(tabulate(comparison, headers='keys', tablefmt='grid'))
    
    # Save comparison to file
    with open(os.path.join(results_dir, 'model_comparison.txt'), 'w') as f:
        f.write(tabulate(comparison, headers='keys', tablefmt='grid'))
    
    return model_results


if __name__ == "__main__":
    run_latent_dro_analysis()
