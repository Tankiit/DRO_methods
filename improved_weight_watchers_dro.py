"""
Improved Distributionally Robust Optimization for Out-of-Distribution Detection
with Enhanced Weight Spectral Analysis

Key Improvements:
1. More realistic synthetic data generation
2. Better OOD scoring mechanisms  
3. Enhanced feature learning
4. Improved DRO loss formulations
5. More robust training procedures

Requirements:
pip install torch torchvision numpy matplotlib scipy weightwatcher scikit-learn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import weightwatcher as ww
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from scipy.stats import powerlaw
import warnings
warnings.filterwarnings('ignore')

# ============= Improved DRO Loss Functions =============

class ImprovedWassersteinDROLoss(nn.Module):
    """Enhanced Wasserstein DRO loss with better regularization"""
    
    def __init__(self, epsilon=0.3, base_margin=1.0, adapt_rate=0.5, temperature=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.base_margin = base_margin
        self.adapt_rate = adapt_rate
        self.temperature = temperature
        
    def forward(self, outputs, targets, features=None):
        batch_size = outputs.size(0)
        
        # Temperature scaling for better calibration
        scaled_outputs = outputs / self.temperature
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(scaled_outputs, targets, reduction='none')
        
        if features is not None:
            # Enhanced Wasserstein penalty
            feature_norms = torch.norm(features, p=2, dim=1)
            
            # Normalize feature norms by batch statistics
            feature_norms_normalized = (feature_norms - feature_norms.mean()) / (feature_norms.std() + 1e-8)
            
            # Confidence-based adaptive penalty
            probs = F.softmax(scaled_outputs, dim=1)
            confidence, _ = torch.max(probs, dim=1)
            uncertainty = 1 - confidence
            
            # Entropy regularization
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            
            # Adaptive margin with entropy consideration
            adaptive_margin = self.base_margin * (1 + self.adapt_rate * uncertainty + 0.1 * entropy)
            
            # Enhanced penalty combining feature magnitude and distribution
            wasserstein_penalty = adaptive_margin * torch.clamp(
                feature_norms_normalized - self.epsilon, min=0
            )
            
            # Combined loss with better weighting
            loss = ce_loss + wasserstein_penalty
        else:
            loss = ce_loss
            
        return loss.mean()
    
    def compute_ood_score(self, outputs, features):
        """Enhanced OOD scoring with multiple signals"""
        scaled_outputs = outputs / self.temperature
        probs = F.softmax(scaled_outputs, dim=1)
        
        # 1. Confidence score (lower for OOD)
        confidence, _ = torch.max(probs, dim=1)
        
        # 2. Entropy score (higher for OOD)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        # 3. Energy score (higher for OOD)
        energy = -self.temperature * torch.logsumexp(scaled_outputs, dim=1)
        
        if features is not None:
            # 4. Feature norm score (higher for OOD)
            feature_norms = torch.norm(features, p=2, dim=1)
            feature_norms_normalized = (feature_norms - feature_norms.mean()) / (feature_norms.std() + 1e-8)
            
            # Combine multiple scores (higher = more likely OOD)
            ood_score = (1 - confidence) + 0.2 * entropy - 0.1 * energy + 0.1 * feature_norms_normalized
        else:
            ood_score = (1 - confidence) + 0.3 * entropy - 0.2 * energy
            
        return ood_score


class ImprovedCVaRDROLoss(nn.Module):
    """Enhanced CVaR DRO with better worst-case modeling"""
    
    def __init__(self, alpha=0.2, base_margin=1.0, adapt_rate=0.3, temperature=1.0):
        super().__init__()
        self.alpha = alpha
        self.base_margin = base_margin
        self.adapt_rate = adapt_rate
        self.temperature = temperature
        
    def forward(self, outputs, targets, features=None):
        scaled_outputs = outputs / self.temperature
        
        # Compute per-sample losses
        losses = F.cross_entropy(scaled_outputs, targets, reduction='none')
        
        # Adaptive CVaR: adjust alpha based on batch difficulty
        batch_loss_std = losses.std()
        adaptive_alpha = torch.clamp(self.alpha * (1 + batch_loss_std), 0.05, 0.5)
        
        # Sort losses and take worst alpha fraction
        k = max(1, int(torch.ceil(losses.shape[0] * adaptive_alpha).item()))
        worst_losses, worst_indices = torch.topk(losses, k)
        
        if features is not None:
            # Enhanced margins for worst-case samples
            probs = F.softmax(scaled_outputs, dim=1)
            confidence, _ = torch.max(probs, dim=1)
            uncertainty = 1 - confidence
            
            # Focus on worst samples
            worst_uncertainty = uncertainty[worst_indices]
            worst_features = features[worst_indices]
            
            # Feature-based penalty for worst samples
            feature_penalty = torch.norm(worst_features, p=2, dim=1) * 0.1
            
            # Adaptive margin
            adaptive_margin = self.base_margin * (1 + self.adapt_rate * worst_uncertainty)
            
            # Enhanced worst-case loss
            worst_losses = worst_losses * adaptive_margin + feature_penalty
        
        return worst_losses.mean()
    
    def compute_ood_score(self, outputs, features):
        """CVaR-based OOD scoring"""
        scaled_outputs = outputs / self.temperature
        probs = F.softmax(scaled_outputs, dim=1)
        
        # Multiple uncertainty measures
        confidence, _ = torch.max(probs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        energy = -self.temperature * torch.logsumexp(scaled_outputs, dim=1)
        
        # Worst-case perspective: focus on high uncertainty
        uncertainty_score = (1 - confidence) + 0.3 * entropy
        
        if features is not None:
            feature_norms = torch.norm(features, p=2, dim=1)
            # Combine with feature information
            ood_score = uncertainty_score - 0.1 * energy + 0.1 * feature_norms
        else:
            ood_score = uncertainty_score - 0.2 * energy
            
        return ood_score


# ============= Enhanced Neural Network Model =============

class ImprovedDRONet(nn.Module):
    """Enhanced neural network with better feature learning"""
    
    def __init__(self, input_dim=2, hidden_dim=128, num_classes=2, dropout=0.2):
        super().__init__()
        
        # Enhanced feature extractor with residual connections
        self.feature_extractor = nn.Sequential(
            # First block
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second block with residual
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Third block
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout//2),
        )
        
        # Feature projection for better representation
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Classifier with better architecture
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout//2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        
    def forward(self, x, return_features=False):
        # Extract features
        raw_features = self.feature_extractor(x)
        features = self.feature_projection(raw_features)
        
        # Classification
        outputs = self.classifier(features)
        
        if return_features:
            return outputs, features
        return outputs
    
    def get_ood_scores(self, x, dro_loss):
        """Enhanced OOD scoring"""
        with torch.no_grad():
            outputs, features = self.forward(x, return_features=True)
            ood_scores = dro_loss.compute_ood_score(outputs, features)
        return ood_scores


# ============= Enhanced Data Generation =============

def generate_realistic_synthetic_data(n_samples=2000):
    """Generate more challenging and realistic synthetic data"""
    np.random.seed(42)
    
    # Create more complex ID distributions
    # Class 1: mixture of Gaussians
    n1 = n_samples // 2
    component1_1 = np.random.multivariate_normal([-3, -1], [[0.5, 0.2], [0.2, 0.8]], n1//2)
    component1_2 = np.random.multivariate_normal([-1, 1], [[0.8, -0.3], [-0.3, 0.6]], n1//2)
    X1 = np.vstack([component1_1, component1_2])
    y1 = np.zeros(n1)
    
    # Class 2: mixture of Gaussians with different shape
    n2 = n_samples // 2
    component2_1 = np.random.multivariate_normal([2, -1], [[0.6, -0.4], [-0.4, 1.0]], n2//2)
    component2_2 = np.random.multivariate_normal([1, 2], [[1.0, 0.1], [0.1, 0.5]], n2//2)
    X2 = np.vstack([component2_1, component2_2])
    y2 = np.ones(n2)
    
    X_train = np.vstack([X1, X2])
    y_train = np.hstack([y1, y2]).astype(int)
    
    # Create more realistic OOD data with multiple types
    ood_samples = []
    
    # Type 1: Uniform in distant regions
    uniform_ood = np.random.uniform(-6, 6, (n_samples//3, 2))
    # Keep only points far from training distribution
    distances = np.min([
        np.linalg.norm(uniform_ood - [-2, 0], axis=1),
        np.linalg.norm(uniform_ood - [1.5, 0.5], axis=1)
    ], axis=0)
    uniform_ood = uniform_ood[distances > 3.5]
    if len(uniform_ood) > 0:
        ood_samples.append(uniform_ood)
    
    # Type 2: Gaussian clusters in different regions
    gaussian_ood1 = np.random.multivariate_normal([4, 4], [[0.3, 0], [0, 0.3]], n_samples//4)
    gaussian_ood2 = np.random.multivariate_normal([-4, 4], [[0.4, 0.1], [0.1, 0.4]], n_samples//4)
    ood_samples.extend([gaussian_ood1, gaussian_ood2])
    
    # Type 3: Ring-shaped distribution
    angles = np.random.uniform(0, 2*np.pi, n_samples//4)
    radii = np.random.uniform(4.5, 5.5, n_samples//4)
    ring_ood = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    ood_samples.append(ring_ood)
    
    # Combine all OOD samples
    X_ood = np.vstack(ood_samples)
    
    return X_train, y_train, X_ood


# ============= Enhanced Training with Better Monitoring =============

def train_improved_dro_model(model, train_loader, dro_loss, optimizer, 
                           epochs=100, device='cpu', monitor_weights=True,
                           patience=15, min_delta=1e-4):
    """Enhanced training with early stopping and better monitoring"""
    model.to(device)
    history = {
        'loss': [], 'worst_loss': [], 'weight_analysis': [],
        'lr': [], 'grad_norm': []
    }
    
    # Initialize weight analyzer - simplified version for this demo
    analyzer = SimpleWeightAnalyzer(model) if monitor_weights else None
    
    # Early stopping setup
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, verbose=True
    )
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        worst_loss = 0
        total_grad_norm = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs, features = model(data, return_features=True)
            loss = dro_loss(outputs, target, features)
            
            loss.backward()
            
            # Gradient clipping for stability
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_grad_norm += grad_norm.item()
            
            optimizer.step()
            
            total_loss += loss.item()
            worst_loss = max(worst_loss, loss.item())
        
        # Update learning rate
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        # Record metrics
        history['loss'].append(avg_loss)
        history['worst_loss'].append(worst_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['grad_norm'].append(total_grad_norm / len(train_loader))
        
        # Weight monitoring
        if analyzer and epoch % 10 == 0:
            try:
                results = analyzer.analyze(plot=False, verbose=False)
                history['weight_analysis'].append({
                    'epoch': epoch,
                    'avg_alpha': results.get('avg_alpha', 3.0),
                    'generalization_score': results.get('generalization_score', 0.7),
                    'num_issues': len(results.get('correlation_traps', []))
                })
                
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, "
                      f"Params={results.get('total_params', 0):,}, "
                      f"Grad_Norm={total_grad_norm / len(train_loader):.4f}")
            except Exception as e:
                print(f"Epoch {epoch}: Loss={avg_loss:.4f} (Weight analysis failed: {e})")
        elif epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping check
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with loss: {best_loss:.4f}")
    
    return history, analyzer


# ============= Simplified Weight Analyzer =============

class SimpleWeightAnalyzer:
    """Simplified weight analyzer when WeightWatcher is not available"""
    
    def __init__(self, model):
        self.model = model
        # Try to initialize WeightWatcher, fall back to basic analysis if it fails
        try:
            self.watcher = ww.WeightWatcher(model=model)
            self.has_weightwatcher = True
        except Exception as e:
            print(f"WeightWatcher initialization failed: {e}")
            print("Using basic parameter analysis instead...")
            self.watcher = None
            self.has_weightwatcher = False
    
    def analyze(self, plot=False, verbose=False):
        """Perform weight analysis with fallback to basic stats"""
        if self.has_weightwatcher and self.watcher is not None:
            try:
                # Try full WeightWatcher analysis
                details = self.watcher.analyze(
                    plot=False,
                    alphas=True,
                    spectralnorms=True,
                    softrank=True,
                    stablerank=True
                )
                
                results = self._process_ww_results(details, verbose)
                
                if plot and verbose:
                    try:
                        self._plot_basic_analysis(details)
                    except:
                        print("Plotting failed, continuing without plots")
                        
                return results
                
            except Exception as e:
                print(f"WeightWatcher analysis failed: {e}, falling back to basic analysis")
                self.has_weightwatcher = False
        
        # Basic analysis fallback
        return self._basic_analysis(verbose)
    
    def _process_ww_results(self, details, verbose=True):
        """Process WeightWatcher results"""
        results = {
            'layer_analysis': [],
            'avg_alpha': 0,
            'correlation_traps': [],
            'recommendations': [],
            'quality_metrics': {},
            'generalization_score': 0
        }
        
        total_alpha = 0
        num_layers = 0
        
        for index, row in details.iterrows():
            alpha = row.get('alpha', np.nan)
            if not np.isnan(alpha):
                total_alpha += alpha
                num_layers += 1
                
                if alpha < 2.0:
                    results['correlation_traps'].append(f"Layer {index}: Low α={alpha:.2f}")
                elif alpha > 6.0:
                    results['correlation_traps'].append(f"Layer {index}: High α={alpha:.2f}")
        
        results['avg_alpha'] = total_alpha / num_layers if num_layers > 0 else 3.0
        results['generalization_score'] = max(0.2, min(1.0, 1.0 - abs(results['avg_alpha'] - 4.0) / 4.0))
        
        # Basic quality metrics
        results['quality_metrics'] = {
            'num_layers_analyzed': num_layers,
            'alpha_in_range': sum(1 for alpha in [row.get('alpha', np.nan) for _, row in details.iterrows()] 
                                if not np.isnan(alpha) and 2.0 <= alpha <= 6.0) / max(1, num_layers)
        }
        
        if verbose:
            print(f"Weight Analysis: α={results['avg_alpha']:.3f}, Score={results['generalization_score']:.3f}")
        
        return results
    
    def _basic_analysis(self, verbose=False):
        """Basic parameter analysis when WeightWatcher fails"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Compute basic statistics
        param_stats = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:
                # For 2D+ tensors, compute some basic spectral properties
                param_2d = param.view(param.shape[0], -1)
                try:
                    U, S, V = torch.svd(param_2d.cpu())
                    spectral_norm = S[0].item() if len(S) > 0 else 1.0
                    stable_rank = (S**2).sum() / (S[0]**2) if len(S) > 0 else 1.0
                    param_stats.append({
                        'name': name,
                        'spectral_norm': spectral_norm,
                        'stable_rank': stable_rank.item()
                    })
                except:
                    # Fallback for problematic tensors
                    param_stats.append({
                        'name': name,
                        'spectral_norm': 1.0,
                        'stable_rank': 1.0
                    })
        
        # Generate pseudo-alpha based on parameter statistics
        avg_spectral_norm = np.mean([s['spectral_norm'] for s in param_stats]) if param_stats else 1.0
        avg_stable_rank = np.mean([s['stable_rank'] for s in param_stats]) if param_stats else 1.0
        
        # Heuristic to estimate alpha-like behavior
        pseudo_alpha = 2.0 + 2.0 * min(1.0, avg_spectral_norm / 5.0) + min(1.0, avg_stable_rank / 10.0)
        
        results = {
            'layer_analysis': param_stats,
            'avg_alpha': pseudo_alpha,
            'correlation_traps': [],
            'recommendations': ['Basic analysis only - install weightwatcher for full analysis'],
            'quality_metrics': {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'avg_spectral_norm': avg_spectral_norm,
                'avg_stable_rank': avg_stable_rank
            },
            'generalization_score': max(0.3, min(0.9, 1.0 - abs(pseudo_alpha - 3.5) / 3.5))
        }
        
        if verbose:
            print(f"Basic Analysis - Total Parameters: {total_params:,}")
            print(f"Estimated α: {pseudo_alpha:.3f}, Generalization Score: {results['generalization_score']:.3f}")
        
        return results
    
    def _plot_basic_analysis(self, details):
        """Create basic plots from WeightWatcher results"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Alpha values
            alphas = details['alpha'].dropna()
            if len(alphas) > 0:
                axes[0].bar(range(len(alphas)), alphas, alpha=0.7)
                axes[0].axhline(y=2.0, color='red', linestyle='--', alpha=0.7)
                axes[0].axhline(y=6.0, color='red', linestyle='--', alpha=0.7)
                axes[0].set_xlabel('Layer')
                axes[0].set_ylabel('Alpha (α)')
                axes[0].set_title('Power Law Exponents')
                axes[0].grid(True, alpha=0.3)
            
            # Spectral norms
            spectral_norms = details['spectral_norm'].dropna()
            if len(spectral_norms) > 0:
                axes[1].bar(range(len(spectral_norms)), spectral_norms, alpha=0.7)
                axes[1].set_xlabel('Layer')
                axes[1].set_ylabel('Spectral Norm')
                axes[1].set_title('Spectral Norms')
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Plotting failed: {e}")


# ============= Enhanced Evaluation =============

def evaluate_improved_ood_detection(model, dro_loss, X_train, y_train, X_ood, device='cpu'):
    """Comprehensive OOD detection evaluation"""
    model.eval()
    model.to(device)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_ood_tensor = torch.FloatTensor(X_ood).to(device)
    
    # Get scores
    train_scores = model.get_ood_scores(X_train_tensor, dro_loss).cpu().numpy()
    ood_scores = model.get_ood_scores(X_ood_tensor, dro_loss).cpu().numpy()
    
    # Calculate comprehensive metrics
    y_true = np.concatenate([np.zeros(len(train_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([train_scores, ood_scores])
    
    # ROC metrics
    auroc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # FPR at 95% TPR
    idx_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95 = fpr[idx_95]
    
    # AUPR
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    aupr = auc(recall, precision)
    
    # Statistical tests
    from scipy.stats import ks_2samp
    ks_stat, ks_pvalue = ks_2samp(train_scores, ood_scores)
    
    results = {
        'auroc': auroc,
        'aupr': aupr,
        'fpr_at_95': fpr_at_95,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'id_score_mean': np.mean(train_scores),
        'id_score_std': np.std(train_scores),
        'ood_score_mean': np.mean(ood_scores),
        'ood_score_std': np.std(ood_scores),
        'separation': np.mean(ood_scores) - np.mean(train_scores)
    }
    
    return results, train_scores, ood_scores


# ============= Enhanced Visualization =============

def plot_comprehensive_results(model, dro_loss, X_train, y_train, X_ood, 
                             train_scores, ood_scores, results, title="Enhanced DRO Results"):
    """Create comprehensive visualization of results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Decision boundary with OOD scores
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 2, X_train[:, 0].max() + 2
    y_min, y_max = X_train[:, 1].min() - 2, X_train[:, 1].max() + 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    mesh_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        mesh_scores = model.get_ood_scores(mesh_points, dro_loss).numpy()
    mesh_scores = mesh_scores.reshape(xx.shape)
    
    contour = axes[0, 0].contourf(xx, yy, mesh_scores, levels=20, cmap='viridis', alpha=0.8)
    fig.colorbar(contour, ax=axes[0, 0], label='OOD Score')
    
    # Plot data points
    axes[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdBu', 
                      edgecolors='black', s=40, alpha=0.8, label='ID Data')
    axes[0, 0].scatter(X_ood[:, 0], X_ood[:, 1], c='green', marker='x', 
                      s=20, alpha=0.6, label='OOD Data')
    axes[0, 0].set_title(f'{title}\nAUROC: {results["auroc"]:.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Score distributions
    axes[0, 1].hist(train_scores, bins=30, alpha=0.7, label='ID Scores', density=True)
    axes[0, 1].hist(ood_scores, bins=30, alpha=0.7, label='OOD Scores', density=True)
    axes[0, 1].set_xlabel('OOD Score')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Score Distributions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(
        np.concatenate([np.zeros(len(train_scores)), np.ones(len(ood_scores))]),
        np.concatenate([train_scores, ood_scores])
    )
    axes[0, 2].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={results["auroc"]:.3f})')
    axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 2].set_xlabel('False Positive Rate')
    axes[0, 2].set_ylabel('True Positive Rate')
    axes[0, 2].set_title('ROC Curve')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(
        np.concatenate([np.zeros(len(train_scores)), np.ones(len(ood_scores))]),
        np.concatenate([train_scores, ood_scores])
    )
    axes[1, 0].plot(recall, precision, linewidth=2, label=f'PR (AUC={results["aupr"]:.3f})')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Score statistics box plot
    score_data = [train_scores, ood_scores]
    axes[1, 1].boxplot(score_data, labels=['ID', 'OOD'])
    axes[1, 1].set_ylabel('OOD Score')
    axes[1, 1].set_title('Score Statistics')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Metrics summary
    axes[1, 2].axis('off')
    metrics_text = f"""
Performance Metrics:

AUROC: {results['auroc']:.4f}
AUPR: {results['aupr']:.4f}
FPR@95%TPR: {results['fpr_at_95']:.4f}

Score Statistics:
ID Mean: {results['id_score_mean']:.3f} ± {results['id_score_std']:.3f}
OOD Mean: {results['ood_score_mean']:.3f} ± {results['ood_score_std']:.3f}
Separation: {results['separation']:.3f}

Statistical Test:
KS Statistic: {results['ks_statistic']:.4f}
KS p-value: {results['ks_pvalue']:.2e}
"""
    
    axes[1, 2].text(0.05, 0.95, metrics_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    return fig


# ============= Main Improved Demonstration =============

def main_improved():
    """Enhanced main demonstration with better performance"""
    
    print("🚀 Starting Improved DRO-OOD Detection with Weight Watchers")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate improved synthetic data
    print("\n📊 Generating realistic synthetic data...")
    X_train, y_train, X_ood = generate_realistic_synthetic_data(n_samples=2000)
    print(f"ID samples: {len(X_train)}, OOD samples: {len(X_ood)}")
    
    # Create data loader
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize improved models
    models = {
        'Standard CE': (ImprovedDRONet(input_dim=2, hidden_dim=128), nn.CrossEntropyLoss()),
        'Improved Wasserstein DRO': (ImprovedDRONet(input_dim=2, hidden_dim=128), 
                                   ImprovedWassersteinDROLoss(epsilon=0.3, temperature=2.0)),
        'Improved CVaR DRO': (ImprovedDRONet(input_dim=2, hidden_dim=128), 
                             ImprovedCVaRDROLoss(alpha=0.2, temperature=2.0))
    }
    
    results_summary = {}
    
    for name, (model, loss_fn) in models.items():
        print(f"\n{'='*30} Training {name} {'='*30}")
        
        # Setup optimizer with better learning rate
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        
        # For standard cross-entropy, wrap it
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            class StandardLoss:
                def __call__(self, outputs, targets, features):
                    return F.cross_entropy(outputs, targets)
                def compute_ood_score(self, outputs, features):
                    probs = F.softmax(outputs, dim=1)
                    confidence, _ = torch.max(probs, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    return (1 - confidence) + 0.3 * entropy
            loss_fn = StandardLoss()
        
        # Enhanced training
        history, analyzer = train_improved_dro_model(
            model, train_loader, loss_fn, optimizer, 
            epochs=150, device=device, monitor_weights=True,
            patience=20
        )
        
        # Comprehensive evaluation
        print(f"\n📈 Evaluating {name}...")
        results, train_scores, ood_scores = evaluate_improved_ood_detection(
            model, loss_fn, X_train, y_train, X_ood, device=device
        )
        
        results_summary[name] = results
        
        # Print results
        print(f"\n📊 {name} Results:")
        print(f"   AUROC: {results['auroc']:.4f}")
        print(f"   AUPR: {results['aupr']:.4f}")
        print(f"   FPR@95%TPR: {results['fpr_at_95']:.4f}")
        print(f"   Score Separation: {results['separation']:.3f}")
        
        # Create comprehensive visualization
        fig = plot_comprehensive_results(
            model, loss_fn, X_train, y_train, X_ood,
            train_scores, ood_scores, results, title=name
        )
        plt.show()
        
        # Weight analysis if available
        if analyzer:
            print(f"\n🔍 Final Weight Analysis for {name}:")
            try:
                final_results = analyzer.analyze(plot=True)
                print(f"   Generalization Score: {final_results['generalization_score']:.3f}")
            except Exception as e:
                print(f"   Weight analysis failed: {e}")
    
    # Compare all methods
    print(f"\n{'='*70}")
    print("📊 FINAL COMPARISON")
    print(f"{'='*70}")
    
    comparison_data = []
    for name, results in results_summary.items():
        comparison_data.append([
            name,
            f"{results['auroc']:.4f}",
            f"{results['aupr']:.4f}",
            f"{results['fpr_at_95']:.4f}",
            f"{results['separation']:.3f}"
        ])
    
    print(f"{'Method':<25} {'AUROC':<8} {'AUPR':<8} {'FPR@95':<8} {'Separation':<10}")
    print("-" * 70)
    for row in comparison_data:
        print(f"{row[0]:<25} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<10}")
    
    print("\n✅ Improved Analysis Complete!")
    
    return results_summary


if __name__ == "__main__":
    main_improved() 