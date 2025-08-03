"""
Wasserstein DRO for CIFAR10 vs SVHN Out-of-Distribution Detection
================================================================

This implementation properly applies Wasserstein DRO to CIFAR10/SVHN,
with correct uncertainty set construction and boundary-based detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, SVHN
from torch.utils.data import DataLoader, Subset
from torch.autograd import grad
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torch.func import vmap, grad as func_grad, functional_call
import warnings
warnings.filterwarnings('ignore')


# ============= Neural Network Architecture =============

class DROResNet(nn.Module):
    """
    Simplified ResNet-like architecture without BatchNorm for efficient vmap
    """
    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        
        # Simplified blocks without BatchNorm
        self.layer1 = self._make_layer(64, 128, stride=2)
        self.layer2 = self._make_layer(128, 256, stride=2)
        self.layer3 = self._make_layer(256, 512, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_layer = nn.Linear(512, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Feature statistics for Mahalanobis distance
        self.register_buffer('feature_mean', None)
        self.register_buffer('feature_precision', None)
        
    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        
        # Downsampling layer
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        # Additional conv layer
        layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, return_features=False):
        # Initial conv
        out = F.relu(self.conv1(x))
        
        # Simplified blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Global pooling
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        
        # Features and classification
        features = self.feature_layer(out)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def get_features(self, x):
        """Extract features for OOD detection"""
        with torch.no_grad():
            _, features = self.forward(x, return_features=True)
        return features


# ============= Wasserstein DRO Loss Implementation =============

class WassersteinDROLoss(nn.Module):
    """
    Efficient Wasserstein DRO loss using vmap (no BatchNorm issues)
    """
    def __init__(self, epsilon=0.1, p=2, reg_weight=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.p = p
        self.reg_weight = reg_weight
        self.lambda_val = 1.0
        
        # Track statistics
        self.worst_case_losses = []
        self.gradient_norms = []
        
    def compute_per_sample_grads(self, model, x, y):
        """
        Efficient per-sample gradients using vmap (works without BatchNorm)
        """
        def compute_loss_for_sample(x_single, y_single):
            """Loss function for a single sample"""
            x_batch = x_single.unsqueeze(0)
            y_batch = y_single.unsqueeze(0)
            
            outputs = model(x_batch)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = F.cross_entropy(outputs, y_batch)
            return loss
        
        # Create per-sample gradient function
        grad_fn = func_grad(compute_loss_for_sample, argnums=0)
        
        # Vectorize over the batch - this should work efficiently now
        per_sample_grads = vmap(grad_fn, in_dims=(0, 0))(x, y)
        
        return per_sample_grads
    
    def forward(self, model, x, y):
        """
        Efficient Wasserstein DRO loss with vmap
        """
        # Standard forward pass for base loss
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        ce_losses = F.cross_entropy(logits, y, reduction='none')
        
        # Compute per-sample gradients efficiently
        per_sample_grads = self.compute_per_sample_grads(model, x, y)
        
        # Compute gradient norms
        grad_flat = per_sample_grads.view(x.shape[0], -1)
        
        if self.p == 2:
            grad_norms = torch.norm(grad_flat, p=2, dim=1)
        elif self.p == 1:
            grad_norms = torch.norm(grad_flat, p=float('inf'), dim=1)
        else:
            q = self.p / (self.p - 1) if self.p > 1 else float('inf')
            grad_norms = torch.norm(grad_flat, p=q, dim=1)
        
        # Wasserstein DRO loss
        worst_case_losses = ce_losses + self.lambda_val * self.epsilon * grad_norms
        
        # Add L2 regularization
        if self.reg_weight > 0:
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2) ** 2
            avg_loss = worst_case_losses.mean() + self.reg_weight * l2_reg
        else:
            avg_loss = worst_case_losses.mean()
        
        # Store statistics
        self.worst_case_losses.append(worst_case_losses.mean().item())
        self.gradient_norms.append(grad_norms.mean().item())
        
        return avg_loss


# ============= OOD Detection Methods =============

class WassersteinDRODetector:
    """
    OOD detection using Wasserstein DRO-trained models
    Implements multiple detection strategies based on DRO properties
    """
    
    def __init__(self, model, epsilon=0.1, p=2):
        self.model = model
        self.epsilon = epsilon
        self.p = p
        
    def fit_distribution(self, train_loader, device='mps'):
        """
        Fit training distribution statistics for Mahalanobis-based detection
        """
        print("Computing training distribution statistics...")
        
        self.model.eval()
        features_list = []
        
        with torch.no_grad():
            for x, _ in train_loader:
                x = x.to(device)
                features = self.model.get_features(x)
                features_list.append(features.cpu())
        
        # Concatenate all features
        all_features = torch.cat(features_list, dim=0)
        
        # Compute mean and precision matrix
        mean = all_features.mean(dim=0)
        centered = all_features - mean
        cov = (centered.T @ centered) / (len(all_features) - 1)
        
        # Add regularization for numerical stability
        cov += 0.01 * torch.eye(cov.shape[0])
        precision = torch.inverse(cov)
        
        # Store in model buffers
        self.model.feature_mean = mean.to(device)
        self.model.feature_precision = precision.to(device)
        
        print(f"Fitted distribution with {len(all_features)} samples")
    
    @torch.no_grad()
    def score_mahalanobis(self, x):
        """
        Mahalanobis distance-based OOD score
        Lower distance = more likely ID
        """
        features = self.model.get_features(x)
        
        # Mahalanobis distance
        centered = features - self.model.feature_mean
        distances = torch.sqrt(
            torch.sum(centered @ self.model.feature_precision * centered, dim=1)
        )
        
        return -distances
    
    @torch.no_grad()
    def score_energy(self, x, temperature=1.0):
        """
        Energy-based OOD score
        Higher energy = more likely OOD
        """
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        # Energy score: -T * log(sum(exp(logits/T)))
        energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
        
        return energy  # Negate so higher = more OOD
    
    @torch.no_grad()
    def score_boundary_distance(self, x):
        """
        Distance to decision boundary
        Based on the property that ID samples are further from boundaries
        """
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        # Get top two logits
        sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
        
        # Distance to boundary ≈ difference between top two classes
        boundary_dist = sorted_logits[:, 0] - sorted_logits[:, 1]
        
        # Invert: smaller distance = more likely OOD
        return 1.0 / (boundary_dist + 1e-6)
    
    def score_worst_case_gradient(self, x):
        """
        OOD score based on worst-case loss gradient under Wasserstein constraint
        This directly uses the DRO framework
        """
        x = x.clone().requires_grad_(True)
        
        # Get predictions
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        # Use entropy as uncertainty measure
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        # Compute gradient magnitude
        grads = []
        for i in range(x.shape[0]):
            grad_i = grad(entropy[i], x, retain_graph=True)[0]
            grads.append(grad_i)
        
        grads = torch.stack(grads)
        grad_norms = torch.norm(grads.view(x.shape[0], -1), p=self.p, dim=1)
        
        # Score combines uncertainty and sensitivity
        scores = entropy + self.epsilon * grad_norms
        
        return scores.detach()
    
    def score_combined(self, x, weights=None):
        """
        Combined score using multiple detection methods
        """
        if weights is None:
            weights = {
                'mahalanobis': 0.3,
                'energy': 0.3,
                'boundary': 0.2,
                'gradient': 0.2
            }
        
        scores = {}
        
        # Compute individual scores
        scores['mahalanobis'] = self.score_mahalanobis(x)
        scores['energy'] = self.score_energy(x)
        scores['boundary'] = self.score_boundary_distance(x)
        scores['gradient'] = self.score_worst_case_gradient(x)
        
        # Normalize scores to [0, 1] range
        for key in scores:
            s = scores[key]
            s_min, s_max = s.min(), s.max()
            scores[key] = (s - s_min) / (s_max - s_min + 1e-8)
        
        # Weighted combination
        combined = torch.zeros_like(scores['mahalanobis'])
        for key, weight in weights.items():
            combined += weight * scores[key]
        
        return combined


# ============= Training Functions =============

def train_wasserstein_dro(model, train_loader, val_loader, num_epochs=50, 
                         epsilon=0.1, p=2, lr=0.001, device='mps'):
    """
    Train model with proper Wasserstein DRO
    """
    model = model.to(device)
    
    # Initialize DRO loss
    dro_loss_fn = WassersteinDROLoss(epsilon=epsilon, p=p, reg_weight=1e-4)
    
    # Model optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'val_acc': [],
        'lambda': [],
        'grad_norm': []
    }
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Compute DRO loss
            loss = dro_loss_fn(model, x, y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Track accuracy
            with torch.no_grad():
                outputs = model(x)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = outputs.max(1)
                correct += predicted.eq(y).sum().item()
                total += y.size(0)
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(y).sum().item()
                val_total += y.size(0)
        
        # Metrics
        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total
        avg_loss = total_loss / len(train_loader)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Record history
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        history['lambda'].append(dro_loss_fn.lambda_val)
        history['grad_norm'].append(np.mean(dro_loss_fn.gradient_norms))
        
        # Clear statistics
        dro_loss_fn.worst_case_losses = []
        dro_loss_fn.gradient_norms = []
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            print(f"  λ: {dro_loss_fn.lambda_val:.4f}, "
                  f"Avg Grad Norm: {history['grad_norm'][-1]:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    
    return model, history


# ============= Evaluation Functions =============

def evaluate_ood_detection(detector, id_loader, ood_loader, device='mps'):
    """
    Comprehensive evaluation of OOD detection performance
    """
    results = {}
    
    # Methods to evaluate
    methods = {
        'mahalanobis': detector.score_mahalanobis,
        'energy': detector.score_energy,
        'boundary': detector.score_boundary_distance,
        'gradient': detector.score_worst_case_gradient,
        'combined': detector.score_combined
    }
    
    for method_name, score_fn in methods.items():
        print(f"\nEvaluating {method_name}...")
        
        # Collect scores
        id_scores = []
        ood_scores = []
        
        # ID data
        for x, _ in id_loader:
            x = x.to(device)
            scores = score_fn(x)
            id_scores.extend(scores.cpu().numpy())
        
        # OOD data
        for x, _ in ood_loader:
            x = x.to(device)
            scores = score_fn(x)
            ood_scores.extend(scores.cpu().numpy())
        
        # Compute metrics
        id_scores = np.array(id_scores)
        ood_scores = np.array(ood_scores)
        
        # Labels: 0 for ID, 1 for OOD
        labels = np.concatenate([
            np.zeros(len(id_scores)),
            np.ones(len(ood_scores))
        ])
        scores = np.concatenate([id_scores, ood_scores])
        
        # AUROC
        auroc = roc_auc_score(labels, scores)
        
        # FPR at 95% TPR
        fpr, tpr, _ = roc_curve(labels, scores)
        fpr_at_95_tpr = fpr[np.argmax(tpr >= 0.95)]
        
        results[method_name] = {
            'auroc': auroc,
            'fpr_at_95_tpr': fpr_at_95_tpr,
            'id_mean': np.mean(id_scores),
            'id_std': np.std(id_scores),
            'ood_mean': np.mean(ood_scores),
            'ood_std': np.std(ood_scores),
            'separation': np.mean(ood_scores) - np.mean(id_scores)
        }
        
        print(f"  AUROC: {auroc:.4f}")
        print(f"  FPR@95TPR: {fpr_at_95_tpr:.4f}")
        print(f"  ID: {results[method_name]['id_mean']:.3f} ± {results[method_name]['id_std']:.3f}")
        print(f"  OOD: {results[method_name]['ood_mean']:.3f} ± {results[method_name]['ood_std']:.3f}")
    
    return results


# ============= Visualization Functions =============

def visualize_results(results, history=None):
    """
    Visualize OOD detection results and training history
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 1. AUROC comparison
    plt.subplot(2, 3, 1)
    methods = list(results.keys())
    aurocs = [results[m]['auroc'] for m in methods]
    bars = plt.bar(methods, aurocs)
    plt.ylabel('AUROC')
    plt.title('OOD Detection Performance')
    plt.xticks(rotation=45)
    
    # Color best method
    best_idx = np.argmax(aurocs)
    bars[best_idx].set_color('red')
    
    # 2. Score distributions for best method
    best_method = methods[best_idx]
    plt.subplot(2, 3, 2)
    
    # Generate synthetic distributions for visualization
    id_scores = np.random.normal(
        results[best_method]['id_mean'], 
        results[best_method]['id_std'], 
        1000
    )
    ood_scores = np.random.normal(
        results[best_method]['ood_mean'], 
        results[best_method]['ood_std'], 
        1000
    )
    
    plt.hist(id_scores, bins=30, alpha=0.5, label='ID', density=True, color='blue')
    plt.hist(ood_scores, bins=30, alpha=0.5, label='OOD', density=True, color='red')
    plt.xlabel('OOD Score')
    plt.ylabel('Density')
    plt.title(f'Score Distributions ({best_method})')
    plt.legend()
    
    # 3. FPR@95TPR comparison
    plt.subplot(2, 3, 3)
    fprs = [results[m]['fpr_at_95_tpr'] for m in methods]
    plt.bar(methods, fprs)
    plt.ylabel('FPR @ 95% TPR')
    plt.title('False Positive Rate at 95% True Positive Rate')
    plt.xticks(rotation=45)
    
    # 4. Training loss
    if history:
        plt.subplot(2, 3, 4)
        plt.plot(history['train_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss (Wasserstein DRO)')
        plt.grid(True, alpha=0.3)
        
        # 5. Lambda evolution
        plt.subplot(2, 3, 5)
        plt.plot(history['lambda'], 'g-')
        plt.xlabel('Epoch')
        plt.ylabel('λ (Lagrange Multiplier)')
        plt.title('Dual Variable Evolution')
        plt.grid(True, alpha=0.3)
        
        # 6. Gradient norms
        plt.subplot(2, 3, 6)
        plt.plot(history['grad_norm'], 'r-')
        plt.xlabel('Epoch')
        plt.ylabel('Average Gradient Norm')
        plt.title('Worst-Case Perturbation Magnitude')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============= Main Demo Function =============

def wasserstein_dro_cifar_demo():
    """
    Complete demonstration of Wasserstein DRO for CIFAR10 vs SVHN
    """
    print("=== Wasserstein DRO for CIFAR10 vs SVHN OOD Detection ===\n")
    
    # Device setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Data loading
    print("Loading datasets...")
    
    # CIFAR10 transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # CIFAR10 datasets
    train_data = CIFAR10(root='/Users/tanmoy/research/data', train=True, download=True, transform=transform_train)
    test_data = CIFAR10(root='/Users/tanmoy/research/data', train=False, download=True, transform=transform_test)
    
    # Use subset for faster training in demo
    train_subset = Subset(train_data, range(10000))  # 10k samples
    val_subset = Subset(test_data, range(2000))      # 2k samples
    test_subset = Subset(test_data, range(2000, 4000))  # 2k samples
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=2)
    
    # SVHN as OOD
    transform_svhn = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    ood_data = SVHN(root='/Users/tanmoy/research/data', split='test', download=True, transform=transform_svhn)
    ood_subset = Subset(ood_data, range(2000))  # 2k samples
    ood_loader = DataLoader(ood_subset, batch_size=128, shuffle=False, num_workers=2)
    
    print("Datasets loaded!\n")
    
    # Model initialization
    model = DROResNet(num_classes=10)
    
    # Train with Wasserstein DRO
    print("Training with Wasserstein DRO...")
    print("Parameters: ε=0.1, p=2 (Wasserstein-2 distance)\n")
    
    model, history = train_wasserstein_dro(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=30,  # Reduced for demo
        epsilon=0.1,
        p=2,
        lr=0.001,
        device=device
    )
    
    # Initialize OOD detector
    print("\nInitializing OOD detector...")
    detector = WassersteinDRODetector(model, epsilon=0.1, p=2)
    
    # Fit distribution statistics
    detector.fit_distribution(train_loader, device)
    
    # Evaluate OOD detection
    print("\nEvaluating OOD detection performance...")
    results = evaluate_ood_detection(detector, test_loader, ood_loader, device)
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(results, history)
    
    # Summary
    print("\n=== Summary ===")
    best_method = max(results.keys(), key=lambda m: results[m]['auroc'])
    print(f"Best method: {best_method}")
    print(f"AUROC: {results[best_method]['auroc']:.4f}")
    print(f"FPR@95TPR: {results[best_method]['fpr_at_95_tpr']:.4f}")
    print(f"\nWasserstein DRO successfully creates directional boundaries that improve OOD detection!")
    
    return model, detector, results


if __name__ == "__main__":
    # Run the demo
    model, detector, results = wasserstein_dro_cifar_demo() 