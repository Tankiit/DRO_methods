#!/usr/bin/env python3
"""
ID-only robust OOD detection with *multiple scoring functions* and *DRO*
--score {energy|mahalanobis|msp}
--divergence {wasserstein|kl|chi2}
"""
import argparse, torch, torch.nn as nn, torch.nn.functional as F, numpy as np, json
import timm, tqdm, sklearn.covariance
from sklearn.metrics import roc_auc_score
from geomloss import SamplesLoss
from skwdro.torch import robustify
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path
import torchvision.utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- CLI ----------
def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', default='resnet18')
    p.add_argument('--score', choices=['energy','mahalanobis','msp'], default='energy')
    p.add_argument('--divergence', choices=['wasserstein','kl','chi2'], default='wasserstein')
    p.add_argument('--eps', default='auto')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=128)
    return p.parse_args()

# ---------- DATA ----------
def get_loaders(batch):
    import torchvision.transforms as T, torchvision.datasets as D
    tf = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize((0.5,)*3,(0.5,)*3)])
    tr = D.CIFAR10(root='/home/tanmoy/research/data', train=True, download=True, transform=tf)
    te = D.CIFAR10(root='/home/tanmoy/research/data', train=False, transform=tf)
    svhn = D.SVHN(root='/home/tanmoy/research/data', split='test', transform=tf)
    return (torch.utils.data.DataLoader(tr,  batch_size=batch, shuffle=True),
            torch.utils.data.DataLoader(te,  batch_size=512, shuffle=False),
            torch.utils.data.DataLoader(svhn, batch_size=512, shuffle=False))

# ---------- MODEL ----------
class Scorer(nn.Module):
    def __init__(self, backbone, score_type):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.score_type = score_type
        d = self.backbone.num_features
        if score_type == 'energy':
            self.head = nn.Linear(d, 10)
        elif score_type == 'mahalanobis':
            self.register_buffer('mean', torch.zeros(10, d))
            self.register_buffer('inv_cov', torch.eye(d))
        elif score_type == 'msp':
            self.head = nn.Linear(d, 10)  # Added classifier for MSP
        
        # Add classifier property for WDROLoss compatibility
        self.classifier = self.head if score_type in ['energy', 'msp'] else None

    def forward(self, x, return_feat=False):
        z = self.backbone(x)
        if return_feat:
            return z
        if self.score_type == 'energy':
            return -torch.logsumexp(self.head(z), dim=1)
        elif self.score_type == 'mahalanobis':
            diff = z.unsqueeze(1) - self.mean.unsqueeze(0)
            maha = torch.sum(diff * (diff @ self.inv_cov.T), dim=2)  # [B, 10]
            return -maha.min(1)[0]  # high = ID
        elif self.score_type == 'msp':
            logits = self.head(z)
            return logits.softmax(1).max(1)[0]  # high = ID

    def fit_mahalanobis(self, loader):
        feats, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                f = self.forward(x.cuda(), return_feat=True)
                feats.append(f); labels.append(y)
        feats = torch.cat(feats); labels = torch.cat(labels)
        for c in range(10):
            mask = labels == c
            self.mean[c] = feats[mask].mean(0)
            cov = sklearn.covariance.EmpiricalCovariance().fit(feats[mask].cpu().numpy())
            self.inv_cov = torch.tensor(cov.precision_, dtype=torch.float32, device=feats.device)

# ---------- DRO ----------
class WDROLoss(nn.Module):
    """
    Exact Wasserstein-DRO loss (dual form) with PGD inner maximization.
    Adapted for OOD scoring instead of classification.
    """
    def __init__(self, radius=0.1, cost_norm=2, inner_lr=2e-3, inner_steps=10):
        super().__init__()
        self.radius = radius
        self.cost_norm = cost_norm
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

    def forward(self, model, x, y):
        batch_size = x.size(0)
        device = x.device
        x_adv = x.clone().detach().requires_grad_(True)

        # --- inner maximisation: find worst-case x' within W-ball ---
        for _ in range(self.inner_steps):
            # Get OOD scores (higher means more ID-like)
            scores = model(x_adv)
            # Loss: we want to minimize ID scores for adversarial samples
            loss = -scores  # per-sample losses

            # gradient wrt input
            grad, = torch.autograd.grad(loss.sum(), x_adv, create_graph=True)
            with torch.no_grad():
                # ascent step
                x_adv += self.inner_lr * grad.sign()
                # l2 projection back onto ε-ball
                delta = x_adv - x
                delta_norm = delta.view(batch_size, -1).norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
                delta = delta * (self.radius / delta_norm).view(-1, 1, 1, 1)
                x_adv = x + delta
            x_adv.requires_grad_(True)

        # --- outer minimisation: loss on the adversarial distribution ---
        scores_clean = model(x)
        scores_adv = model(x_adv)
        
        # Maximize clean scores, minimize adversarial scores
        dro_loss = F.relu(1 + scores_adv.mean() - scores_clean.mean())

        return dro_loss

# ---------- TRAIN ----------
def train(args):
    # Create unique run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.backbone}_{args.score}_wdro_{timestamp}"  # Updated name
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(run_dir / "tensorboard")
    
    # Save configuration
    config = vars(args)
    config.update({
        "wdro_radius": 0.1,
        "wdro_inner_steps": 10,
        "wdro_inner_lr": 2e-3
    })
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    train_loader, id_test, ood_test = get_loaders(args.batch)
    scorer = Scorer(args.backbone, args.score).to(device)
    if args.score == 'mahalanobis':
        scorer.fit_mahalanobis(train_loader)

    # Initialize WDRO loss
    dro = WDROLoss(
        radius=0.1,  # Can be made configurable via args
        inner_steps=10,
        inner_lr=2e-3
    ).to(device)
    
    # Create optimizer for scorer parameters
    opt = torch.optim.AdamW(scorer.parameters(), lr=3e-4)

    # Training loop with enhanced progress tracking
    global_step = 0
    best_auc = 0.0
    
    for epoch in tqdm.trange(1, args.epochs+1, desc="Training epochs"):
        scorer.train()
        epoch_loss = 0.0
        batch_iterator = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        
        for x, y in batch_iterator:
            x, y = x.cuda(), y.cuda()
            loss = dro(scorer, x, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # Update metrics
            epoch_loss += loss.item()
            batch_loss = loss.item()
            
            # Update progress bar
            batch_iterator.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{epoch_loss/len(batch_iterator):.4f}'
            })
            
            # Log to tensorboard
            writer.add_scalar('train/batch_loss', batch_loss, global_step)
            global_step += 1
        
        # Log epoch metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
        
        # Evaluation at each epoch
        if epoch % 1 == 0:
            scorer.eval()
            id_scores, ood_scores = [], []
            with torch.no_grad():
                # ID test set
                for x,_ in tqdm.tqdm(id_test, desc="Evaluating ID", leave=False):
                    batch_scores = scorer(x.cuda()).cpu().numpy()
                    id_scores.extend(batch_scores)
                
                # OOD test set
                for x,_ in tqdm.tqdm(ood_test, desc="Evaluating OOD", leave=False):
                    batch_scores = scorer(x.cuda()).cpu().numpy()
                    ood_scores.extend(batch_scores)
            
            # Convert to numpy arrays
            id_scores = np.array(id_scores)
            ood_scores = np.array(ood_scores)
            
            # Calculate metrics
            labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
            scores = np.concatenate([id_scores, ood_scores])
            auc = roc_auc_score(labels, scores)
            fpr95 = np.percentile(id_scores, 5)
            
            # Log evaluation metrics
            writer.add_scalar('eval/auroc', auc, epoch)
            writer.add_scalar('eval/fpr95', fpr95, epoch)
            
            # Log score distributions
            writer.add_histogram('eval/ID_score_dist', id_scores, epoch)
            writer.add_histogram('eval/OOD_score_dist', ood_scores, epoch)
            
            # Create checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': scorer.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': avg_epoch_loss,
                'auc': auc,
                'fpr95': fpr95,
                'radius': dro.radius
            }
            
            # Save best model
            if auc > best_auc:
                best_auc = auc
                torch.save(checkpoint, run_dir / "best_model.pt")
            
            # Save regular checkpoint
            torch.save(checkpoint, run_dir / f"checkpoint_epoch_{epoch}.pt")
            
            # Print current results
            print(f"Epoch {epoch}: {args.score}+WDRO  AUROC={auc:.4f}  FPR95={fpr95:.4f}  radius={dro.radius:.4f}")
    
    # Save final results
    final_results = {
        "auc": auc,
        "fpr95": fpr95,
        "radius": dro.radius,
        "final_loss": avg_epoch_loss,
        "best_auc": best_auc
    }
    json.dump(final_results, open(run_dir / "final_results.json", "w"), indent=4)
    writer.close()

if __name__ == "__main__":
    train(parse())