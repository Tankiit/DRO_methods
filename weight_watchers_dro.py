"""
Distributionally Robust Optimization for Out-of-Distribution Detection
with Adaptive Margins and Weight Spectral Analysis

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
from sklearn.metrics import roc_auc_score
from scipy.stats import powerlaw
import warnings
warnings.filterwarnings('ignore')

# ============= DRO Loss Functions =============

class WassersteinDROLoss(nn.Module):
    """Wasserstein DRO loss with adaptive margins"""
    
    def __init__(self, epsilon=0.5, base_margin=1.0, adapt_rate=0.3):
        super().__init__()
        self.epsilon = epsilon
        self.base_margin = base_margin
        self.adapt_rate = adapt_rate
        
    def forward(self, outputs, targets, features=None):
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        
        # Wasserstein penalty based on feature norms
        if features is not None:
            feature_norms = torch.norm(features, p=2, dim=1)
            wasserstein_penalty = torch.clamp(feature_norms - self.epsilon, min=0)
            
            # Adaptive margin based on prediction confidence
            probs = F.softmax(outputs, dim=1)
            confidence, _ = torch.max(probs, dim=1)
            uncertainty = 1 - confidence
            
            adaptive_margin = self.base_margin * (1 + self.adapt_rate * uncertainty)
            
            # Combined loss
            loss = ce_loss + adaptive_margin * wasserstein_penalty
        else:
            loss = ce_loss
            
        return loss.mean()
    
    def compute_ood_score(self, outputs, features):
        """Compute OOD score with adaptive margins"""
        probs = F.softmax(outputs, dim=1)
        confidence, _ = torch.max(probs, dim=1)
        
        if features is not None:
            feature_norms = torch.norm(features, p=2, dim=1)
            uncertainty = 1 - confidence
            adaptive_margin = self.base_margin * (1 + self.adapt_rate * uncertainty)
            
            # OOD score: lower confidence + higher feature norm = higher OOD
            ood_score = (1 - confidence) + 0.1 * torch.clamp(feature_norms - self.epsilon, min=0)
            ood_score = ood_score / adaptive_margin
        else:
            ood_score = 1 - confidence
            
        return ood_score


class CVaRDROLoss(nn.Module):
    """CVaR DRO loss focusing on worst-case samples"""
    
    def __init__(self, alpha=0.1, base_margin=1.0, adapt_rate=0.3):
        super().__init__()
        self.alpha = alpha
        self.base_margin = base_margin
        self.adapt_rate = adapt_rate
        
    def forward(self, outputs, targets, features=None):
        # Compute per-sample losses
        losses = F.cross_entropy(outputs, targets, reduction='none')
        
        # Sort losses and take worst alpha fraction
        k = max(1, int(np.ceil(losses.shape[0] * self.alpha)))
        worst_losses, _ = torch.topk(losses, k)
        
        # Apply adaptive margins to worst-case samples
        if features is not None:
            probs = F.softmax(outputs, dim=1)
            confidence, _ = torch.max(probs, dim=1)
            uncertainty = 1 - confidence
            
            # Get indices of worst samples
            _, worst_indices = torch.topk(losses, k)
            worst_uncertainty = uncertainty[worst_indices]
            
            adaptive_margin = self.base_margin * (1 + self.adapt_rate * worst_uncertainty)
            worst_losses = worst_losses * adaptive_margin
        
        return worst_losses.mean()
    
    def compute_ood_score(self, outputs, features):
        """Compute OOD score based on CVaR principle"""
        probs = F.softmax(outputs, dim=1)
        confidence, _ = torch.max(probs, dim=1)
        
        # Use entropy as additional uncertainty measure
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        # Adaptive margin
        uncertainty = 1 - confidence
        adaptive_margin = self.base_margin * (1 + self.adapt_rate * uncertainty)
        
        # OOD score combines confidence and entropy
        ood_score = (1 - confidence + 0.1 * entropy) / adaptive_margin
        
        return ood_score


# ============= Neural Network Model =============

class DRONet(nn.Module):
    """Neural network with DRO training and feature extraction"""
    
    def __init__(self, input_dim=2, hidden_dim=64, num_classes=2, dropout=0.1):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        outputs = self.classifier(features)
        
        if return_features:
            return outputs, features
        return outputs
    
    def get_ood_scores(self, x, dro_loss):
        """Compute OOD scores using the specified DRO loss function"""
        with torch.no_grad():
            outputs, features = self.forward(x, return_features=True)
            ood_scores = dro_loss.compute_ood_score(outputs, features)
        return ood_scores


# ============= Enhanced Weight Spectral Analysis =============

class WeightSpectralAnalyzer:
    """Enhanced analyzer with improved Weight Watchers integration"""
    
    def __init__(self, model):
        self.model = model
        # Initialize WeightWatcher with better error handling
        try:
            self.watcher = ww.WeightWatcher(model=model)
        except Exception as e:
            print(f"Warning: WeightWatcher initialization failed: {e}")
            self.watcher = None
        
    def analyze(self, plot=True, verbose=True):
        """Perform comprehensive weight analysis with enhanced error handling"""
        if self.watcher is None:
            print("WeightWatcher not available, performing basic analysis...")
            return self._basic_analysis()
        
        try:
            # Get detailed weight statistics
            details = self.watcher.analyze(
                plot=False,
                alphas=True,
                spectralnorms=True,
                softrank=True,
                stablerank=True,
                mp_softrank=True,
                randomize=True,  # Add randomization test
                fix_fingers=True  # Fix potential numerical issues
            )
            
            results = self._process_ww_results(details, verbose)
            
            if plot:
                self._plot_enhanced_spectral_analysis(details)
                
        except Exception as e:
            print(f"WeightWatcher analysis failed: {e}")
            print("Falling back to basic analysis...")
            results = self._basic_analysis()
            
        return results
    
    def _process_ww_results(self, details, verbose=True):
        """Process WeightWatcher results with enhanced metrics"""
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
        spectral_norms = []
        alphas = []
        
        for index, row in details.iterrows():
            layer_info = {
                'layer_id': row.get('layer_id', index),
                'alpha': row.get('alpha', np.nan),
                'stable_rank': row.get('stable_rank', np.nan),
                'spectral_norm': row.get('spectral_norm', np.nan),
                'mp_softrank': row.get('mp_softrank', np.nan),
                'softrank': row.get('softrank', np.nan),
                'shape': f"{row.get('M', 'N/A')} × {row.get('N', 'N/A')}",
                'num_params': row.get('M', 0) * row.get('N', 0)
            }
            
            # Enhanced correlation trap detection
            alpha = layer_info['alpha']
            if not np.isnan(alpha):
                total_alpha += alpha
                num_layers += 1
                alphas.append(alpha)
                
                # More nuanced alpha analysis
                if alpha < 1.5:
                    results['correlation_traps'].append(
                        f"Layer {layer_info['layer_id']}: Critical α={alpha:.2f} (heavily overtrained)"
                    )
                elif alpha < 2.0:
                    results['correlation_traps'].append(
                        f"Layer {layer_info['layer_id']}: Low α={alpha:.2f} (overtrained)"
                    )
                elif alpha > 8.0:
                    results['correlation_traps'].append(
                        f"Layer {layer_info['layer_id']}: Very high α={alpha:.2f} (undertrained)"
                    )
                elif alpha > 6.0:
                    results['correlation_traps'].append(
                        f"Layer {layer_info['layer_id']}: High α={alpha:.2f} (possibly undertrained)"
                    )
            
            # Spectral norm analysis
            spec_norm = layer_info['spectral_norm']
            if not np.isnan(spec_norm):
                spectral_norms.append(spec_norm)
                if spec_norm > 10.0:
                    results['correlation_traps'].append(
                        f"Layer {layer_info['layer_id']}: High spectral norm {spec_norm:.2f}"
                    )
            
            results['layer_analysis'].append(layer_info)
        
        # Calculate quality metrics
        results['avg_alpha'] = total_alpha / num_layers if num_layers > 0 else 0
        results['quality_metrics'] = self._calculate_quality_metrics(alphas, spectral_norms)
        results['generalization_score'] = self._estimate_generalization_score(results)
        
        # Enhanced recommendations
        results['recommendations'] = self._generate_enhanced_recommendations(results)
        
        if verbose:
            self._print_analysis_summary(results)
        
        return results
    
    def _calculate_quality_metrics(self, alphas, spectral_norms):
        """Calculate enhanced quality metrics"""
        metrics = {}
        
        if alphas:
            alphas = np.array(alphas)
            metrics['alpha_mean'] = np.mean(alphas)
            metrics['alpha_std'] = np.std(alphas)
            metrics['alpha_min'] = np.min(alphas)
            metrics['alpha_max'] = np.max(alphas)
            metrics['alpha_in_range'] = np.mean((alphas >= 2.0) & (alphas <= 6.0))
            
        if spectral_norms:
            spectral_norms = np.array(spectral_norms)
            metrics['spectral_norm_mean'] = np.mean(spectral_norms)
            metrics['spectral_norm_max'] = np.max(spectral_norms)
            metrics['spectral_norm_stable'] = np.mean(spectral_norms < 5.0)
            
        return metrics
    
    def _estimate_generalization_score(self, results):
        """Estimate generalization capability based on spectral properties"""
        score = 0.5  # Base score
        
        # Alpha contribution (40% weight)
        avg_alpha = results['avg_alpha']
        if 2.0 <= avg_alpha <= 6.0:
            alpha_score = 1.0 - abs(avg_alpha - 4.0) / 2.0  # Peak at 4.0
        else:
            alpha_score = 0.2
        score += 0.4 * alpha_score
        
        # Correlation traps penalty (30% weight)
        trap_penalty = min(len(results['correlation_traps']) * 0.1, 0.3)
        score -= trap_penalty
        
        # Quality metrics (30% weight)
        metrics = results['quality_metrics']
        if 'alpha_in_range' in metrics:
            score += 0.3 * metrics['alpha_in_range']
        
        return max(0.0, min(1.0, score))
    
    def _generate_enhanced_recommendations(self, results):
        """Generate enhanced recommendations based on analysis"""
        recommendations = []
        
        avg_alpha = results['avg_alpha']
        generalization_score = results['generalization_score']
        
        # Alpha-based recommendations
        if avg_alpha < 1.5:
            recommendations.append("🚨 CRITICAL: Model severely overtrained - stop training immediately")
        elif avg_alpha < 2.0:
            recommendations.append("⚠️  Model overtrained - implement early stopping or reduce learning rate")
        elif avg_alpha > 8.0:
            recommendations.append("📈 Model severely undertrained - increase training epochs or learning rate")
        elif avg_alpha > 6.0:
            recommendations.append("📊 Model may be undertrained - consider more training")
        else:
            recommendations.append("✅ Alpha values in good range")
        
        # Generalization score recommendations
        if generalization_score < 0.3:
            recommendations.append("🎯 Poor generalization expected - consider regularization or architecture changes")
        elif generalization_score < 0.6:
            recommendations.append("🎯 Moderate generalization expected - fine-tune hyperparameters")
        else:
            recommendations.append("🎯 Good generalization expected")
        
        # Correlation trap specific recommendations
        if len(results['correlation_traps']) > len(results['layer_analysis']) * 0.5:
            recommendations.append("🔧 Many layers have issues - consider reducing model complexity")
        
        # Quality metrics recommendations
        metrics = results['quality_metrics']
        if 'spectral_norm_max' in metrics and metrics['spectral_norm_max'] > 10.0:
            recommendations.append("🔧 High spectral norms detected - add spectral normalization")
        
        return recommendations
    
    def _print_analysis_summary(self, results):
        """Print a comprehensive analysis summary"""
        print("\n" + "="*60)
        print("🔍 WEIGHT SPECTRAL ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"📊 Average Alpha: {results['avg_alpha']:.3f}")
        print(f"🎯 Generalization Score: {results['generalization_score']:.3f}")
        print(f"⚠️  Issues Detected: {len(results['correlation_traps'])}")
        
        if results['quality_metrics']:
            print(f"\n📈 Quality Metrics:")
            for key, value in results['quality_metrics'].items():
                print(f"   {key}: {value:.3f}")
        
        if results['correlation_traps']:
            print(f"\n⚠️  Issues Detected:")
            for trap in results['correlation_traps']:
                print(f"   • {trap}")
        
        print(f"\n💡 Recommendations:")
        for rec in results['recommendations']:
            print(f"   • {rec}")
        
        print("="*60)
    
    def _basic_analysis(self):
        """Fallback basic analysis when WeightWatcher is not available"""
        results = {
            'layer_analysis': [],
            'avg_alpha': 0,
            'correlation_traps': [],
            'recommendations': ['WeightWatcher not available - install with: pip install weightwatcher'],
            'quality_metrics': {},
            'generalization_score': 0.5
        }
        
        # Basic parameter count analysis
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        results['quality_metrics'] = {
            'total_params': total_params,
            'trainable_params': trainable_params
        }
        
        print(f"Basic Analysis - Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        
        return results
    
    def _plot_enhanced_spectral_analysis(self, details):
        """Enhanced plotting with more comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Alpha values by layer
        alphas = details['alpha'].dropna()
        if len(alphas) > 0:
            bars = axes[0, 0].bar(range(len(alphas)), alphas, alpha=0.7)
            axes[0, 0].axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Safe range')
            axes[0, 0].axhline(y=6.0, color='red', linestyle='--', alpha=0.7)
            axes[0, 0].axhline(y=4.0, color='green', linestyle='-', alpha=0.7, label='Optimal')
            
            # Color bars based on quality
            for i, (bar, alpha) in enumerate(zip(bars, alphas)):
                if alpha < 2.0 or alpha > 6.0:
                    bar.set_color('red')
                elif 3.0 <= alpha <= 5.0:
                    bar.set_color('green')
                else:
                    bar.set_color('orange')
            
            axes[0, 0].set_xlabel('Layer')
            axes[0, 0].set_ylabel('Alpha (α)')
            axes[0, 0].set_title('Power Law Exponents by Layer')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Stable rank
        stable_ranks = details['stable_rank'].dropna()
        if len(stable_ranks) > 0:
            axes[0, 1].bar(range(len(stable_ranks)), stable_ranks, alpha=0.7, color='blue')
            axes[0, 1].set_xlabel('Layer')
            axes[0, 1].set_ylabel('Stable Rank')
            axes[0, 1].set_title('Stable Rank by Layer')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Spectral norms
        spectral_norms = details['spectral_norm'].dropna()
        if len(spectral_norms) > 0:
            bars = axes[0, 2].bar(range(len(spectral_norms)), spectral_norms, alpha=0.7)
            axes[0, 2].axhline(y=5.0, color='orange', linestyle='--', alpha=0.7, label='Caution')
            axes[0, 2].axhline(y=10.0, color='red', linestyle='--', alpha=0.7, label='High')
            
            # Color bars based on spectral norm values
            for bar, norm in zip(bars, spectral_norms):
                if norm > 10.0:
                    bar.set_color('red')
                elif norm > 5.0:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
            
            axes[0, 2].set_xlabel('Layer')
            axes[0, 2].set_ylabel('Spectral Norm')
            axes[0, 2].set_title('Spectral Norms by Layer')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Alpha distribution histogram
        if len(alphas) > 0:
            axes[1, 0].hist(alphas, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(x=2.0, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].axvline(x=6.0, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].axvline(x=np.mean(alphas), color='blue', linestyle='-', alpha=0.7, label=f'Mean: {np.mean(alphas):.2f}')
            axes[1, 0].set_xlabel('Alpha Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Alpha Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # MP softrank vs softrank comparison
        mp_softrank = details['mp_softrank'].dropna()
        softrank = details['softrank'].dropna()
        if len(mp_softrank) > 0 and len(softrank) > 0:
            min_len = min(len(mp_softrank), len(softrank))
            axes[1, 1].scatter(softrank[:min_len], mp_softrank[:min_len], alpha=0.7)
            axes[1, 1].plot([0, max(softrank[:min_len])], [0, max(softrank[:min_len])], 'r--', alpha=0.5)
            axes[1, 1].set_xlabel('Softrank')
            axes[1, 1].set_ylabel('MP Softrank')
            axes[1, 1].set_title('Softrank vs MP Softrank')
            axes[1, 1].grid(True, alpha=0.3)
        
        # ESD plot for first analyzable layer
        try:
            # Find first layer with sufficient data
            for idx in range(min(3, len(details))):
                try:
                    self.watcher.plot_ESD(layer=idx, ax=axes[1, 2])
                    axes[1, 2].set_title(f'ESD of Layer {idx}')
                    break
                except:
                    continue
            else:
                axes[1, 2].text(0.5, 0.5, 'ESD plot not available', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('ESD Plot')
        except:
            axes[1, 2].text(0.5, 0.5, 'ESD plot not available', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('ESD Plot')
        
        plt.tight_layout()
        plt.show()
    
    def monitor_training(self, epoch, history):
        """Monitor training progress with weight analysis"""
        if epoch % 10 == 0 and self.watcher is not None:
            print(f"\n📊 Weight Analysis at Epoch {epoch}:")
            try:
                results = self.analyze(plot=False, verbose=False)
                history.setdefault('weight_analysis', []).append({
                    'epoch': epoch,
                    'avg_alpha': results['avg_alpha'],
                    'generalization_score': results['generalization_score'],
                    'num_issues': len(results['correlation_traps'])
                })
                
                print(f"   α: {results['avg_alpha']:.3f}, Score: {results['generalization_score']:.3f}")
                if results['correlation_traps']:
                    print(f"   Issues: {len(results['correlation_traps'])}")
                    
            except Exception as e:
                print(f"   Weight analysis failed: {e}")


# ============= Enhanced Visualization Functions =============

def plot_decision_boundary_with_analysis(model, X, y, dro_loss, analyzer, title="Decision Boundary"):
    """Enhanced decision boundary plot with weight analysis overlay"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get weight analysis
    results = analyzer.analyze(plot=False, verbose=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Decision boundary with OOD scores
    mesh_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    ood_scores = model.get_ood_scores(mesh_points, dro_loss).numpy()
    ood_scores = ood_scores.reshape(xx.shape)
    
    contour = ax1.contourf(xx, yy, ood_scores, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, ax=ax1, label='OOD Score')
    
    # Plot decision boundary
    with torch.no_grad():
        Z = model(mesh_points).argmax(dim=1).numpy()
    Z = Z.reshape(xx.shape)
    ax1.contour(xx, yy, Z, colors='k', linewidths=2, alpha=0.5)
    
    # Plot data points
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', 
               edgecolors='black', s=50, alpha=0.9)
    
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title(f'{title}\nGeneralization Score: {results["generalization_score"]:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Weight analysis summary
    ax2.axis('off')
    
    # Create text summary
    summary_text = f"""
Weight Spectral Analysis Summary

Average Alpha (α): {results['avg_alpha']:.3f}
Generalization Score: {results['generalization_score']:.3f}
Issues Detected: {len(results['correlation_traps'])}

Quality Metrics:
"""
    
    for key, value in results['quality_metrics'].items():
        if isinstance(value, float):
            summary_text += f"  {key}: {value:.3f}\n"
    
    if results['recommendations']:
        summary_text += "\nRecommendations:\n"
        for i, rec in enumerate(results['recommendations'][:3]):  # Show first 3
            summary_text += f"• {rec}\n"
    
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    return fig


# ============= Enhanced Training Function =============

def train_dro_model_with_monitoring(model, train_loader, dro_loss, optimizer, 
                                   epochs=50, device='cpu', monitor_weights=True):
    """Enhanced training with weight monitoring"""
    model.to(device)
    history = {'loss': [], 'worst_loss': [], 'weight_analysis': []}
    
    # Initialize weight analyzer
    analyzer = WeightSpectralAnalyzer(model) if monitor_weights else None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        worst_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs, features = model(data, return_features=True)
            loss = dro_loss(outputs, target, features)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            worst_loss = max(worst_loss, loss.item())
        
        avg_loss = total_loss / len(train_loader)
        history['loss'].append(avg_loss)
        history['worst_loss'].append(worst_loss)
        
        # Weight monitoring
        if analyzer and epoch % 10 == 0:
            analyzer.monitor_training(epoch, history)
        
        if epoch % 10 == 0:
            status = f'Epoch {epoch}: Loss={avg_loss:.4f}, Worst={worst_loss:.4f}'
            if analyzer and history['weight_analysis']:
                latest = history['weight_analysis'][-1]
                status += f', α={latest["avg_alpha"]:.2f}, Score={latest["generalization_score"]:.2f}'
            print(status)
    
    return history, analyzer


# ============= Enhanced Example Usage =============

def generate_synthetic_data(n_samples=1000):
    """Generate 2D synthetic data for demonstration"""
    # In-distribution data: two Gaussian blobs
    np.random.seed(42)
    
    # Class 1
    X1 = np.random.randn(n_samples//2, 2) + np.array([-2, 0])
    y1 = np.zeros(n_samples//2)
    
    # Class 2
    X2 = np.random.randn(n_samples//2, 2) + np.array([2, 0])
    y2 = np.ones(n_samples//2)
    
    X_train = np.vstack([X1, X2])
    y_train = np.hstack([y1, y2]).astype(int)
    
    # OOD data: uniform in far regions
    X_ood = np.random.uniform(-5, 5, (n_samples, 2))
    # Keep only points far from training distribution
    distances = np.min([np.linalg.norm(X_ood - [-2, 0], axis=1),
                       np.linalg.norm(X_ood - [2, 0], axis=1)], axis=0)
    X_ood = X_ood[distances > 3]
    
    return X_train, y_train, X_ood


def main():
    """Enhanced main demonstration with comprehensive weight analysis"""
    
    print("🚀 Starting Enhanced DRO-OOD Detection with Weight Watchers")
    print("="*60)
    
    # Generate data
    X_train, y_train, X_ood = generate_synthetic_data()
    
    # Create data loader
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize models with different DRO losses
    models = {
        'Standard': (DRONet(), nn.CrossEntropyLoss()),
        'Wasserstein DRO': (DRONet(), WassersteinDROLoss(epsilon=0.5)),
        'CVaR DRO': (DRONet(), CVaRDROLoss(alpha=0.1))
    }
    
    histories = {}
    analyzers = {}
    
    for name, (model, loss_fn) in models.items():
        print(f"\n{'='*20} Training {name} {'='*20}")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # For standard cross-entropy, wrap it to match DRO interface
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            class StandardLoss:
                def __call__(self, outputs, targets, features):
                    return F.cross_entropy(outputs, targets)
                def compute_ood_score(self, outputs, features):
                    probs = F.softmax(outputs, dim=1)
                    confidence, _ = torch.max(probs, dim=1)
                    return 1 - confidence
            loss_fn = StandardLoss()
        
        # Enhanced training with monitoring
        history, analyzer = train_dro_model_with_monitoring(
            model, train_loader, loss_fn, optimizer, epochs=100, monitor_weights=True
        )
        histories[name] = history
        analyzers[name] = analyzer
        
        # Evaluate OOD detection
        X_train_tensor = torch.FloatTensor(X_train)
        X_ood_tensor = torch.FloatTensor(X_ood)
        
        train_scores = model.get_ood_scores(X_train_tensor, loss_fn).numpy()
        ood_scores = model.get_ood_scores(X_ood_tensor, loss_fn).numpy()
        
        # Compute AUC
        y_true = np.concatenate([np.zeros(len(train_scores)), np.ones(len(ood_scores))])
        y_scores = np.concatenate([train_scores, ood_scores])
        auc = roc_auc_score(y_true, y_scores)
        
        print(f"\n📊 {name} Results:")
        print(f"   OOD Detection AUC: {auc:.4f}")
        
        # Enhanced decision boundary plot with analysis
        plot_decision_boundary_with_analysis(
            model, X_train, y_train, loss_fn, analyzer,
            title=f"{name} (AUC={auc:.3f})"
        )
        plt.show()
        
        # Comprehensive weight analysis
        if analyzer:
            print(f"\n🔍 Comprehensive Weight Analysis for {name}:")
            final_results = analyzer.analyze(plot=True)
    
    # Plot enhanced training histories
    plot_enhanced_training_history(histories)
    
    print("\n✅ Analysis Complete!")


def plot_enhanced_training_history(histories):
    """Plot enhanced training history with weight analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    for name, history in histories.items():
        axes[0, 0].plot(history['loss'], label=f"{name} - Avg Loss", linewidth=2)
        axes[0, 1].plot(history['worst_loss'], label=f"{name} - Worst Loss", linestyle='--', linewidth=2)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Average Loss During Training')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Worst-Case Loss During Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Weight analysis evolution
    for name, history in histories.items():
        if 'weight_analysis' in history and history['weight_analysis']:
            weight_data = history['weight_analysis']
            epochs = [w['epoch'] for w in weight_data]
            alphas = [w['avg_alpha'] for w in weight_data]
            scores = [w['generalization_score'] for w in weight_data]
            
            axes[1, 0].plot(epochs, alphas, 'o-', label=f"{name}", linewidth=2, markersize=6)
            axes[1, 1].plot(epochs, scores, 's-', label=f"{name}", linewidth=2, markersize=6)
    
    axes[1, 0].axhline(y=2.0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(y=6.0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Average Alpha')
    axes[1, 0].set_title('Alpha Evolution During Training')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Generalization Score')
    axes[1, 1].set_title('Generalization Score Evolution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main() 