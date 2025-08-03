import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, SVHN
from torch.utils.data import DataLoader, Subset

def get_device():
    """Get the appropriate device (MPS or CPU) for Mac"""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"MPS (Metal Performance Shaders) is available and will be used")
    else:
        device = torch.device('cpu')
        print("MPS not available, using CPU")
    return device

def get_cifar10_data(batch_size=64, num_workers=2):
    """Load CIFAR10 dataset with standard transforms"""
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = CIFAR10(root='/Users/tanmoy/research/data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    testset = CIFAR10(root='/Users/tanmoy/research/data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader

def get_svhn_data(batch_size=64, num_workers=2):
    """Load SVHN dataset with standard transforms"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])
    
    # Use test set of SVHN as OOD data
    oodset = SVHN(root='/Users/tanmoy/research/data', split='test', download=True, transform=transform)
    oodloader = DataLoader(oodset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return oodloader

class CIFAR10Classifier(nn.Module):
    """CNN classifier for CIFAR10 with K+1 outputs"""
    def __init__(self, num_classes=10):
        super().__init__()
        # Separate conv and pool layers for better control
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, return_indices=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes + 1)  # K+1 classes
        )
        
        # Store indices for backward pass
        self.pool_indices = []
    
    def forward(self, x):
        # Clear previous indices
        self.pool_indices = []
        
        # First conv block
        x = F.relu(self.conv1(x))
        x, indices1 = self.pool1(x)
        self.pool_indices.append(indices1)
        
        # Second conv block
        x = F.relu(self.conv2(x))
        x, indices2 = self.pool2(x)
        self.pool_indices.append(indices2)
        
        # Third conv block
        x = F.relu(self.conv3(x))
        x, indices3 = self.pool3(x)
        self.pool_indices.append(indices3)
        
        # Flatten and classify
        features = x.view(x.size(0), -1)
        logits = self.classifier(features)
        return logits, features
    
    def get_features(self, x):
        """Get features without storing indices (for inference)"""
        x = F.relu(self.conv1(x))
        x, _ = self.pool1(x)
        x = F.relu(self.conv2(x))
        x, _ = self.pool2(x)
        x = F.relu(self.conv3(x))
        x, _ = self.pool3(x)
        return x.view(x.size(0), -1)

class VirtualKPlusOneDRO(nn.Module):
    """
    Virtual K+1 DRO: Generate virtual OOD samples from uncertainty sets
    and train K+1 classification with DRO regularization
    """
    def __init__(self, classifier, num_classes, epsilon=0.1, virtual_ratio=0.3, alpha=0.5):
        super().__init__()
        self.classifier = classifier
        self.num_classes = num_classes  # Original K classes
        self.epsilon = epsilon
        self.virtual_ratio = virtual_ratio
        self.alpha = alpha  # Weight for DRO regularization
        
        # Ensure classifier has K+1 outputs
        assert classifier.classifier[-1].out_features == num_classes + 1
    
    def generate_virtual_ood_from_uncertainty_set(self, x, y):
        """
        Generate virtual OOD samples from DRO's uncertainty set
        using worst-case perturbations within Wasserstein balls
        """
        x.requires_grad_(True)
        
        # Compute loss gradients for worst-case direction
        logits, _ = self.classifier(x)  # Extract logits from tuple
        logits = logits[:, :-1]  # Only original K classes
        ce_loss = F.cross_entropy(logits, y, reduction='none')
        
        # Gradient gives worst-case perturbation direction
        gradients = torch.autograd.grad(
            outputs=ce_loss.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Normalize gradients
        grad_norms = torch.norm(gradients.view(gradients.size(0), -1), p=2, dim=1, keepdim=True)
        grad_directions = gradients / (grad_norms.view(-1, *([1] * (gradients.dim() - 1))) + 1e-8)
        
        # Generate virtual OOD at boundary of uncertainty set
        virtual_ood = x + self.epsilon * grad_directions
        
        return virtual_ood.detach()
    
    def forward(self, x, y):
        """
        DRO-based K+1 classification with virtual samples from uncertainty sets
        """
        batch_size = x.size(0)
        virtual_size = int(batch_size * self.virtual_ratio)
        
        if virtual_size > 0:
            # Generate virtual OOD samples from uncertainty set
            virtual_indices = torch.randperm(batch_size)[:virtual_size]
            x_virtual_base = x[virtual_indices]
            y_virtual_base = y[virtual_indices]
            
            virtual_ood = self.generate_virtual_ood_from_uncertainty_set(
                x_virtual_base, y_virtual_base
            )
            
            # Combine original and virtual samples
            x_combined = torch.cat([x, virtual_ood])
            y_combined = torch.cat([
                y,
                torch.full((virtual_size,), self.num_classes, device=x.device)  # K+1 class
            ])
        else:
            x_combined = x
            y_combined = y
        
        # K+1 classification loss
        logits, _ = self.classifier(x_combined)  # Extract logits from tuple
        k_plus_one_loss = F.cross_entropy(logits, y_combined)
        
        # DRO regularization on original samples
        x.requires_grad_(True)
        orig_logits, _ = self.classifier(x)  # Extract logits from tuple
        orig_logits = orig_logits[:, :-1]  # Only K classes for DRO
        orig_loss = F.cross_entropy(orig_logits, y, reduction='none')
        
        gradients = torch.autograd.grad(
            outputs=orig_loss.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        
        grad_penalty = torch.norm(gradients.view(gradients.size(0), -1), p=2, dim=1)
        dro_regularization = (orig_loss + self.epsilon * grad_penalty).mean()
        
        # Combined loss: K+1 classification + DRO regularization
        total_loss = k_plus_one_loss + self.alpha * dro_regularization
        
        return total_loss
    
    @torch.no_grad()
    def get_ood_score(self, x):
        """
        Compute OOD score based on K+1 classification
        """
        logits, _ = self.classifier(x)  # Extract logits from tuple
        
        # Get maximum ID class logit
        id_logits = logits[:, :-1]  # All but last class
        max_id_logit = id_logits.max(1)[0]
        
        # Get OOD class logit
        ood_logit = logits[:, -1]
        
        # Score: higher means more likely to be ID
        score = max_id_logit - ood_logit
        
        return score

class DROInspiredKPlusOne(nn.Module):
    """
    Alternative: DRO-inspired virtual generation with multiple methods
    """
    def __init__(self, classifier, num_classes, epsilon=0.1, alpha=0.3):
        super().__init__()
        self.classifier = classifier
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
    
    def generate_virtual_ood_variants(self, x, y, method='gradient'):
        """
        Generate virtual OOD using different DRO-inspired methods
        """
        if method == 'gradient':
            # Gradient-based (worst-case direction)
            x.requires_grad_(True)
            logits, _ = self.classifier(x)  # Extract logits from tuple
            logits = logits[:, :-1]  # Only original K classes
            loss = F.cross_entropy(logits, y, reduction='none')
            gradients = torch.autograd.grad(loss.sum(), x, create_graph=True)[0]
            
            grad_norms = torch.norm(gradients.view(gradients.size(0), -1), p=2, dim=1, keepdim=True)
            normalized_grads = gradients / (grad_norms.view(-1, *([1] * (gradients.dim() - 1))) + 1e-8)
            
            return x + self.epsilon * normalized_grads
            
        elif method == 'random':
            # Random perturbations within L2 ball
            random_directions = torch.randn_like(x)
            random_norms = torch.norm(random_directions.view(random_directions.size(0), -1), p=2, dim=1, keepdim=True)
            normalized_random = random_directions / (random_norms.view(-1, *([1] * (random_directions.dim() - 1))) + 1e-8)
            
            radii = torch.rand(x.size(0), device=x.device).pow(1.0/x.view(x.size(0), -1).size(1))
            radii = radii.view(-1, *([1] * (x.dim() - 1)))
            
            return x + self.epsilon * radii * normalized_random
            
        elif method == 'interpolation':
            # Interpolation between samples
            if x.size(0) > 1:
                indices = torch.randperm(x.size(0))
                alpha = torch.rand(x.size(0), device=x.device).view(-1, *([1] * (x.dim() - 1)))
                return alpha * x + (1 - alpha) * x[indices]
            else:
                return x
    
    def forward(self, x, y, virtual_method='gradient'):
        """
        Train K+1 classification with DRO-inspired virtual OOD generation
        """
        # Generate virtual OOD samples
        virtual_ood = self.generate_virtual_ood_variants(x, y, method=virtual_method)
        
        # Take subset for computational efficiency
        subset_size = min(x.size(0) // 2, virtual_ood.size(0))
        virtual_ood = virtual_ood[:subset_size]
        
        # Combine with original data
        x_combined = torch.cat([x, virtual_ood])
        y_combined = torch.cat([
            y,
            torch.full((subset_size,), self.num_classes, device=x.device)
        ])
        
        # K+1 classification loss
        logits, _ = self.classifier(x_combined)  # Extract logits from tuple
        k_plus_one_loss = F.cross_entropy(logits, y_combined)
        
        # DRO regularization on original samples
        x.requires_grad_(True)
        orig_logits, _ = self.classifier(x)  # Extract logits from tuple
        orig_logits = orig_logits[:, :-1]  # Only K classes for DRO
        orig_loss = F.cross_entropy(orig_logits, y, reduction='none')
        
        gradients = torch.autograd.grad(orig_loss.sum(), x, create_graph=True)[0]
        grad_penalty = torch.norm(gradients.view(gradients.size(0), -1), p=2, dim=1)
        dro_reg = (orig_loss + self.epsilon * grad_penalty).mean()
        
        return k_plus_one_loss + self.alpha * dro_reg
    
    @torch.no_grad()
    def get_ood_score(self, x):
        """
        Compute OOD score based on K+1 classification
        """
        logits, _ = self.classifier(x)  # Extract logits from tuple
        
        id_logits = logits[:, :-1]
        max_id_logit = id_logits.max(1)[0]
        ood_logit = logits[:, -1]
        
        return max_id_logit - ood_logit

def train_virtual_k_plus_one_dro(model, train_loader, num_epochs=20, 
                                epsilon=0.1, virtual_ratio=0.3, device=None):
    """
    Training function for Virtual K+1 DRO
    """
    if device is None:
        device = get_device()
    
    print(f"\nTraining on device: {device}")
    model = model.to(device)
    criterion = VirtualKPlusOneDRO(
        classifier=model,
        num_classes=10,  # CIFAR10 has 10 classes
        epsilon=epsilon,
        virtual_ratio=virtual_ratio
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Track best model
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            loss = criterion(batch_x, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model with loss: {best_loss:.4f}")
    
    return criterion

@torch.no_grad()
def evaluate_virtual_k_plus_one(criterion, id_loader, ood_loader, device=None):
    """
    Evaluate Virtual K+1 DRO on ID and OOD data
    """
    if device is None:
        device = get_device()
    
    print(f"\nEvaluating on device: {device}")
    criterion.classifier.eval()
    
    # Collect scores
    id_scores = []
    ood_scores = []
    
    # ID data
    print("Processing ID data...")
    for batch_x, _ in id_loader:
        batch_x = batch_x.to(device)
        scores = criterion.get_ood_score(batch_x)
        id_scores.extend(scores.cpu().numpy())
    
    # OOD data
    print("Processing OOD data...")
    for batch_x, _ in ood_loader:
        batch_x = batch_x.to(device)
        scores = criterion.get_ood_score(batch_x)
        ood_scores.extend(scores.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score
    
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    
    auroc = roc_auc_score(labels, scores)
    
    print("\nEvaluation Results:")
    print(f"AUROC: {auroc:.4f}")
    print(f"ID Score Mean: {np.mean(id_scores):.4f}")
    print(f"OOD Score Mean: {np.mean(ood_scores):.4f}")
    print(f"Separation: {np.mean(id_scores) - np.mean(ood_scores):.4f}")
    
    return {
        'auroc': auroc,
        'id_score_mean': np.mean(id_scores),
        'ood_score_mean': np.mean(ood_scores),
        'separation': np.mean(id_scores) - np.mean(ood_scores)
    }

def cifar_svhn_demonstration():
    """
    Complete demonstration of Virtual K+1 DRO on CIFAR10 vs SVHN
    """
    torch.manual_seed(42)
    device = get_device()
    
    # Set number of workers based on device
    num_workers = 0 if device.type == 'mps' else 2  # MPS doesn't support multiple workers
    
    # Load datasets
    print("\nLoading datasets...")
    train_loader, test_loader = get_cifar10_data(num_workers=num_workers)
    ood_loader = get_svhn_data(num_workers=num_workers)
    
    # Create model
    model = CIFAR10Classifier(num_classes=10)
    print(f"\nModel device: {device}")
    model = model.to(device)
    
    print("\nTraining Virtual K+1 DRO...")
    criterion = train_virtual_k_plus_one_dro(
        model, 
        train_loader,
        num_epochs=30,
        epsilon=0.1,
        virtual_ratio=0.3,
        device=device
    )
    
    print("\nEvaluating on CIFAR10 (ID) vs SVHN (OOD)...")
    # Use subsets for evaluation
    test_subset = Subset(test_loader.dataset, range(2000))
    test_subset_loader = DataLoader(test_subset, batch_size=2000, shuffle=False, num_workers=num_workers)
    
    ood_subset = Subset(ood_loader.dataset, range(2000))
    ood_subset_loader = DataLoader(ood_subset, batch_size=2000, shuffle=False, num_workers=num_workers)
    
    results = evaluate_virtual_k_plus_one(
        criterion,
        test_subset_loader,
        ood_subset_loader,
        device=device
    )
    
    return criterion, results

if __name__ == "__main__":
    # Run with a specific model
    dro_detector, results = cifar_svhn_demonstration()

    # Or run all models (as in the main block)
    models = [
        'resnet18',
        'efficientnet_b0',
        'mobilenetv3_small',
        'vit_tiny_patch16_224'
    ]
    for model_name in models:
        dro_detector, results = cifar_svhn_demonstration()