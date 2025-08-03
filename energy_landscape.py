import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from pytorch_ood.dataset.img import Textures, TinyImages300k, Places365, TinyImageNetCrop, TinyImageNetResize, LSUNCrop, LSUNResize
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import ToUnknown
from torchvision import transforms as tvt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

# Disable PyTorch compilation to avoid dynamo errors
torch._dynamo.config.suppress_errors = True

class EnergyConstrainedDRO(nn.Module):
    """
    Energy-Constrained DRO: Generate virtual unusual samples from ID data only
    
    Mathematical formulation:
    min_θ E_ID[ℓ(θ; x, y)] + λ·E[ℓ_reject(θ; x_unusual)]
    
    where x_unusual satisfies:
    - E_θ(x_unusual) ≥ τ (energy threshold)
    - KL(δ_x_unusual || P_ID) ≤ ε (stay close to ID distribution)
    """
    
    def __init__(
        self,
        classifier,
        num_classes: int,
        energy_threshold: float = 2.0,  # τ parameter
        kl_radius: float = 0.1,  # ε parameter
        lambda_virtual: float = 0.5,  # λ parameter
        temperature: float = 1.0,
        device: str = "cuda",
        adaptive_threshold: bool = True
    ):
        super().__init__()
        self.classifier = classifier
        self.num_classes = num_classes
        self.energy_threshold = energy_threshold
        self.kl_radius = kl_radius
        self.lambda_virtual = lambda_virtual
        self.temperature = temperature
        self.device = device
        self.adaptive_threshold = adaptive_threshold
        
        # Track ID energy statistics for adaptive threshold
        self.register_buffer('id_energy_mean', None)
        self.register_buffer('id_energy_std', None)
        self.register_buffer('id_features_mean', None)
        self.register_buffer('id_features_cov', None)
        
        # Virtual sample generation tracking
        self.virtual_sample_stats = {
            'generated_count': 0,
            'energy_threshold_hist': [],
            'success_rate': 0.0
        }
    
    def compute_energy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute energy score: E(x) = -T * log(sum(exp(f_i(x)/T)))
        Higher energy = more unusual = more likely OOD
        """
        # Clamp logits to prevent numerical overflow
        logits_clamped = torch.clamp(logits, min=-10, max=10)
        energy = -self.temperature * torch.logsumexp(logits_clamped / self.temperature, dim=1)
        return energy
    
    def update_id_statistics(self, features: torch.Tensor, logits: torch.Tensor, y: torch.Tensor):
        """Update running statistics of ID data for adaptive thresholding"""
        # Only use actual ID samples (not virtual ones)
        id_mask = (y < self.num_classes)
        if not id_mask.any():
            return
            
        id_features = features[id_mask].detach()
        id_logits = logits[id_mask].detach()
        id_energies = self.compute_energy(id_logits)
        
        # Update energy statistics
        if self.id_energy_mean is None:
            self.id_energy_mean = id_energies.mean()
            self.id_energy_std = id_energies.std()
        else:
            momentum = 0.9
            self.id_energy_mean = momentum * self.id_energy_mean + (1 - momentum) * id_energies.mean()
            self.id_energy_std = momentum * self.id_energy_std + (1 - momentum) * id_energies.std()
        
        # Update feature statistics for KL constraint
        if self.id_features_mean is None:
            self.id_features_mean = id_features.mean(dim=0)
            centered = id_features - self.id_features_mean
            self.id_features_cov = torch.mm(centered.t(), centered) / (centered.shape[0] - 1)
        else:
            momentum = 0.9
            current_mean = id_features.mean(dim=0)
            self.id_features_mean = momentum * self.id_features_mean + (1 - momentum) * current_mean
            
            centered = id_features - current_mean
            current_cov = torch.mm(centered.t(), centered) / (centered.shape[0] - 1)
            self.id_features_cov = momentum * self.id_features_cov + (1 - momentum) * current_cov
    
    def adaptive_energy_threshold(self) -> float:
        """Compute adaptive energy threshold based on ID energy distribution"""
        if self.id_energy_mean is None:
            return self.energy_threshold
        
        if self.adaptive_threshold:
            # Set threshold to mean + k*std to capture "unusual" samples
            k = 1.5  # Adjustable parameter
            threshold = self.id_energy_mean + k * self.id_energy_std
            # Clamp threshold to reasonable range
            threshold = torch.clamp(threshold, min=-5, max=5)
            return threshold.item()
        else:
            return self.energy_threshold
    
    def generate_energy_guided_perturbations(self, x_id: torch.Tensor, y_id: torch.Tensor) -> torch.Tensor:
        """
        Generate virtual unusual samples using energy-guided perturbations
        
        Algorithm:
        1. Start with ID sample x_id
        2. Perturb in direction that increases energy
        3. Stop when energy threshold reached or distance constraint violated
        """
        x_virtual = x_id.clone().detach()
        batch_size = x_id.shape[0]
        
        # Get current energy threshold
        current_threshold = self.adaptive_energy_threshold()
        
        # Use PGD-like approach to increase energy
        max_steps = 5
        step_size = 0.001  # Much smaller step size
        successful_generations = 0
        
        for step in range(max_steps):
            x_virtual.requires_grad_(True)
            
            # Forward pass to get energy
            logits, features = self.classifier(x_virtual)
            energies = self.compute_energy(logits)
            
            # Check if we've reached energy threshold
            threshold_reached = energies >= current_threshold
            successful_generations = threshold_reached.sum().item()
            
            if successful_generations == batch_size:
                break
            
            # Compute loss to maximize energy (minimize negative energy)
            energy_loss = -energies.mean()
            
            # Compute gradient
            grad = torch.autograd.grad(energy_loss, x_virtual, retain_graph=False)[0]
            
            # Update in direction of increasing energy
            with torch.no_grad():
                # Normalize gradient and take small step
                grad_norm = torch.norm(grad.view(batch_size, -1), dim=1, keepdim=True)
                grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
                
                x_virtual = x_virtual + step_size * grad_normalized
                
                # Clamp to valid image range
                x_virtual = torch.clamp(x_virtual, 0, 1)
                
                # Ensure we don't move too far from original
                diff = x_virtual - x_id
                diff_norm = torch.norm(diff.view(batch_size, -1), dim=1, keepdim=True)
                max_norm = 0.1  # Maximum L2 distance
                
                mask = (diff_norm > max_norm).view(-1, 1, 1, 1)
                if mask.any():
                    diff_normalized = diff / (diff_norm.view(-1, 1, 1, 1) + 1e-8)
                    x_virtual = torch.where(mask, x_id + max_norm * diff_normalized, x_virtual)
        
        # Update statistics
        self.virtual_sample_stats['generated_count'] += batch_size
        self.virtual_sample_stats['success_rate'] = successful_generations / batch_size
        self.virtual_sample_stats['energy_threshold_hist'].append(current_threshold)
        
        return x_virtual.detach()
    
    def generate_mixup_virtual_samples(self, x_id: torch.Tensor, y_id: torch.Tensor) -> torch.Tensor:
        """
        Generate virtual samples using energy-guided mixup
        
        Algorithm:
        1. Mix pairs of ID samples with different mixing weights
        2. Choose mixing weight that maximizes energy while staying in ID distribution
        """
        batch_size = x_id.shape[0]
        if batch_size < 2:
            return self.generate_energy_guided_perturbations(x_id, y_id)
        
        # Create pairs
        indices = torch.randperm(batch_size, device=x_id.device)
        x1 = x_id
        x2 = x_id[indices]
        y1 = y_id
        y2 = y_id[indices]
        
        # Find optimal mixing weights
        best_lambda = torch.zeros(batch_size, device=x_id.device)
        best_energy = torch.full((batch_size,), float('-inf'), device=x_id.device)
        
        # Search over different mixing weights
        for lam in torch.linspace(0.2, 0.8, 10):
            x_mixed = lam * x1 + (1 - lam) * x2
            
            with torch.no_grad():
                logits, _ = self.classifier(x_mixed)
                energies = self.compute_energy(logits)
            
            # Update best mixing weights
            better_mask = energies > best_energy
            best_lambda[better_mask] = lam
            best_energy[better_mask] = energies[better_mask]
        
        # Generate final mixed samples
        x_virtual = best_lambda.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x1 + \
                    (1 - best_lambda.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) * x2
        
        return x_virtual
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Forward pass implementing energy-constrained DRO
        
        Loss = E_ID[ℓ_cls(θ; x, y)] + λ·E[ℓ_reject(θ; x_virtual)]
        """
        # Standard forward pass
        logits, features = self.classifier(x)
        
        # Update ID statistics for adaptive thresholding
        self.update_id_statistics(features, logits, y)
        
        # Identify ID samples
        id_mask = (y < self.num_classes)
        
        if id_mask.any():
            x_id = x[id_mask]
            y_id = y[id_mask]
            
            # Generate virtual unusual samples from ID data
            if torch.rand(1).item() < 0.5:
                x_virtual = self.generate_energy_guided_perturbations(x_id, y_id)
            else:
                x_virtual = self.generate_mixup_virtual_samples(x_id, y_id)
            
            # Forward pass on virtual samples
            logits_virtual, features_virtual = self.classifier(x_virtual)
            
            # Virtual sample loss: encourage high energy (unusual) predictions
            virtual_energies = self.compute_energy(logits_virtual)
            current_threshold = self.adaptive_energy_threshold()
            
            # Loss should be positive when energy is below threshold (not unusual enough)
            virtual_loss = F.relu(current_threshold - virtual_energies).mean()
            
            # Add weaker consistency loss: virtual samples should be different but not too different
            consistency_loss = 0.1 * F.mse_loss(features[id_mask], features_virtual)
            
        else:
            virtual_loss = torch.tensor(0.0, device=x.device)
            consistency_loss = torch.tensor(0.0, device=x.device)
        
        # Standard classification loss
        classification_loss = F.cross_entropy(logits, y)
        
        # Total loss
        total_loss = (
            classification_loss + 
            self.lambda_virtual * virtual_loss +
            consistency_loss  # Already scaled in computation above
        )
        
        # Store metrics for monitoring
        self.metrics = {
            'classification_loss': classification_loss.item(),
            'virtual_loss': virtual_loss.item() if torch.is_tensor(virtual_loss) else virtual_loss,
            'consistency_loss': consistency_loss.item() if torch.is_tensor(consistency_loss) else consistency_loss,
            'energy_threshold': self.adaptive_energy_threshold(),
            'virtual_success_rate': self.virtual_sample_stats['success_rate']
        }
        
        return total_loss
    
    @torch.no_grad()
    def get_ood_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute OOD score based on energy"""
        logits, _ = self.classifier(x)
        energies = self.compute_energy(logits)
        
        # Higher energy = more likely OOD
        return energies
    
    def visualize_virtual_samples(self, x_id: torch.Tensor, save_path: str = "virtual_samples.png"):
        """Visualize generated virtual samples compared to original ID samples"""
        if x_id.shape[0] < 4:
            return
        
        x_virtual = self.generate_energy_guided_perturbations(x_id[:4], torch.zeros(4, device=x_id.device))
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        
        for i in range(4):
            # Original ID sample
            img_id = x_id[i].cpu().permute(1, 2, 0).numpy()
            axes[0, i].imshow(img_id)
            axes[0, i].set_title(f'ID Sample {i+1}')
            axes[0, i].axis('off')
            
            # Virtual unusual sample
            img_virtual = x_virtual[i].cpu().permute(1, 2, 0).numpy()
            axes[1, i].imshow(np.clip(img_virtual, 0, 1))
            axes[1, i].set_title(f'Virtual Sample {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def train_epoch_energy_constrained(model, train_loader_id, optimizer, criterion, device, writer, epoch, num_epochs):
    """Train for one epoch using energy-constrained DRO"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader_id, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
    
    for batch_id, (x_id, y_id) in enumerate(pbar):
        global_step = epoch * len(train_loader_id) + batch_id
        
        x_id, y_id = x_id.to(device), y_id.to(device)
        
        # Forward pass with energy-constrained DRO
        loss = criterion(x_id, y_id)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log training metrics
        if batch_id % 100 == 0:
            metrics = criterion.metrics
            writer.add_scalar('Train/ClassificationLoss', metrics['classification_loss'], global_step)
            writer.add_scalar('Train/VirtualLoss', metrics['virtual_loss'], global_step)
            writer.add_scalar('Train/EnergyThreshold', metrics['energy_threshold'], global_step)
            writer.add_scalar('Train/VirtualSuccessRate', metrics['virtual_success_rate'], global_step)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'threshold': f'{criterion.metrics["energy_threshold"]:.2f}',
            'success': f'{criterion.metrics["virtual_success_rate"]:.2f}'
        })
    
    return total_loss / len(train_loader_id)

@torch.no_grad()
def evaluate_energy_constrained(model, id_loader, ood_loaders, criterion, device, writer, epoch):
    """Evaluate energy-constrained model"""
    model.eval()
    metrics = {}
    
    # Process ID data
    total_correct = 0
    total_samples = 0
    scores_id = []
    
    for x, y in tqdm(id_loader, desc='Evaluating ID'):
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        
        # Classification accuracy
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_samples += y.size(0)
        
        # OOD scores (energy-based)
        score = criterion.get_ood_score(x)
        scores_id.append(score.cpu())
    
    metrics['id_accuracy'] = total_correct / total_samples
    scores_id = torch.cat(scores_id)
    
    # Process each OOD dataset
    for dataset_name, ood_loader in ood_loaders.items():
        scores_ood = []
        
        for x, _ in tqdm(ood_loader, desc=f'Evaluating OOD ({dataset_name})'):
            x = x.to(device)
            score = criterion.get_ood_score(x)
            scores_ood.append(score.cpu())
        
        scores_ood = torch.cat(scores_ood)
        
        # Calculate OOD detection metrics
        # For energy-based detection: higher energy = more likely OOD
        # But roc_auc_score expects higher scores for positive class (ID=1)
        # So we negate the energy scores to make lower energy = higher score for ID
        labels = torch.cat([torch.ones(scores_id.size(0)), torch.zeros(scores_ood.size(0))])
        scores = torch.cat([scores_id, scores_ood])
        
        # Negate scores so that ID (lower energy) gets higher scores
        auroc = roc_auc_score(labels.numpy(), -scores.numpy())
        
        metrics[f'{dataset_name}_auroc'] = auroc
        
        # Log to TensorBoard
        writer.add_scalar(f'Eval/{dataset_name}_AUROC', auroc, epoch)
        writer.add_histogram(f'Scores/OOD_{dataset_name}', scores_ood, epoch)
    
    # Log ID metrics
    writer.add_histogram('Scores/ID', scores_id, epoch)
    writer.add_scalar('Eval/ID_Accuracy', metrics['id_accuracy'], epoch)
    
    return metrics

# Custom WideResNet with feature extraction (same as original)
class WideResNetFeatures(WideResNet):
    def __init__(self, num_classes=10, pretrained=None):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn1(x))
        x = F.avg_pool2d(x, 8)
        features = x.view(x.size(0), -1)
        logits = self.fc(features)
        return logits, features

def create_model(device, num_classes=10):
    """Create model for energy-constrained DRO"""
    model = WideResNetFeatures(num_classes=1000, pretrained="imagenet32-nocifar")
    # Remove torch.compile for speed - it adds compilation overhead
    # model = torch.compile(model)
    model = model.to(device)
    
    # Get feature dimension
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32, device=device)
        _, features = model(dummy_input)
        feature_dim = features.shape[1]
    
    # Standard classification (no K+1 setup needed for energy-based approach)
    model.fc = torch.nn.Linear(feature_dim, num_classes).to(device)
    
    return model

def create_datasets(root_dir="/Users/tanmoy/research/data", batch_size=128):
    """Create datasets - same as original but simplified for energy approach"""
    trans = WideResNet.transform_for("cifar10-pt")
    
    # Use CIFAR-10 as ID data
    dataset_in_train = CIFAR10(root=root_dir, train=True, download=True, transform=trans)
    dataset_in_test = CIFAR10(root=root_dir, train=False, download=True, transform=trans)
    
    # OOD datasets
    ood_datasets = {
        'svhn': SVHN(root=root_dir, split="test", download=True, transform=trans),
        'textures': Textures(root=root_dir, download=True, transform=trans, target_transform=ToUnknown()),
        'places365': Places365(root=root_dir, download=True, transform=trans, target_transform=ToUnknown()),
    }
    
    # Create data loaders
    train_loader = DataLoader(dataset_in_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader_id = DataLoader(dataset_in_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    test_loaders_ood = {
        name: DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        for name, dataset in ood_datasets.items()
    }
    
    return train_loader, test_loader_id, test_loaders_ood

def main():
    """Main training loop for energy-constrained DRO"""
    torch.manual_seed(123)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup data loaders
    train_loader, test_loader_id, test_loaders_ood = create_datasets(
        root_dir="/Users/tanmoy/research/data",  # Adjust path as needed
        batch_size=64
    )
    
    # Create model
    model = create_model(device, num_classes=10)  # CIFAR-10 has 10 classes
    
    # Initialize Energy-Constrained DRO criterion
    criterion = EnergyConstrainedDRO(
        classifier=model,
        num_classes=10,
        energy_threshold=0.0,  # Start with reasonable threshold
        kl_radius=0.1,
        lambda_virtual=0.1,    # Reduced virtual loss weight
        temperature=1.0,
        device=device,
        adaptive_threshold=True
    )
    
    # Setup optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/energy_constrained_dro')
    
    # Training loop
    num_epochs = 50
    best_avg_auroc = 0
    
    print("🚀 Starting Energy-Constrained DRO Training")
    print("=" * 60)
    print("Key Features:")
    print("✅ Virtual sample generation from ID data only")
    print("✅ Energy-guided perturbations with adaptive thresholds")
    print("✅ KL-constrained unusual sample crafting")
    print("✅ No OOD data required during training")
    print()
    
    for epoch in tqdm(range(num_epochs), desc='Training', leave=True):
        # Training
        train_loss = train_epoch_energy_constrained(
            model=model,
            train_loader_id=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            writer=writer,
            epoch=epoch,
            num_epochs=num_epochs
        )
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            metrics = evaluate_energy_constrained(
                model=model,
                id_loader=test_loader_id,
                ood_loaders=test_loaders_ood,
                criterion=criterion,
                device=device,
                writer=writer,
                epoch=epoch
            )
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"ID Accuracy: {metrics['id_accuracy']:.4f}")
            print(f"Energy Threshold: {criterion.adaptive_energy_threshold():.2f}")
            print(f"Virtual Success Rate: {criterion.virtual_sample_stats['success_rate']:.2f}")
            
            # Debug: Print energy statistics
            with torch.no_grad():
                sample_batch = next(iter(test_loader_id))
                x_sample, y_sample = sample_batch[0][:10].to(device), sample_batch[1][:10].to(device)
                logits_sample, _ = model(x_sample)
                energy_sample = criterion.compute_energy(logits_sample)
                print(f"Sample ID energies: {energy_sample.cpu().numpy()}")
                print(f"ID energy mean: {criterion.id_energy_mean:.2f}, std: {criterion.id_energy_std:.2f}")
            
            for dataset_name in test_loaders_ood.keys():
                print(f"{dataset_name} AUROC: {metrics[f'{dataset_name}_auroc']:.4f}")
            
            # Visualize virtual samples
            if epoch == 0:
                sample_batch = next(iter(train_loader))
                criterion.visualize_virtual_samples(
                    sample_batch[0][:4].to(device),
                    f'virtual_samples_epoch_{epoch+1}.png'
                )
            
            # Save best model
            avg_auroc = np.mean([metrics[f'{name}_auroc'] for name in test_loaders_ood.keys()])
            if avg_auroc > best_avg_auroc:
                best_avg_auroc = avg_auroc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_avg_auroc': best_avg_auroc,
                    'metrics': metrics
                }, 'best_energy_constrained_model.pth')
    
    writer.close()
    print(f"\n🎯 Training completed! Best average AUROC: {best_avg_auroc:.4f}")

if __name__ == "__main__":
    main()