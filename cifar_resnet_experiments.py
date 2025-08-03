import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm, trange
import timm
import argparse
import json
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import os
import pandas as pd
import yaml
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Import from existing modules
from feature_train import (
    HierarchicalDROWithMultiScoring, 
    TimmFeatureExtractor,
    MultiDatasetEvaluator,
    get_imagenet_transform
)
from multi_scoring import MultiScoreOODDetector

class CIFARResNetExperiment:
    """Comprehensive CIFAR experiments with ResNet models"""
    
    def __init__(self, 
                 dataset_name: str = 'cifar10',
                 model_name: str = 'resnet18',
                 data_dir: str = './data',
                 output_dir: str = './cifar_results',
                 device: str = 'cuda'):
        
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Dataset specific settings
        self.num_classes = 10 if dataset_name == 'cifar10' else 100
        self.input_size = 32
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.results_dir = self.output_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        self.scores_dir = self.output_dir / 'intermediate_scores'
        self.scores_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.model = None
        self.hierarchical_system = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.ood_loaders = {}
        
        # Results tracking
        self.training_history = []
        self.evaluation_results = []
        self.intermediate_scores = {}
        
    def get_cifar_transforms(self, train=True, dataset='cifar10'):
        """Get CIFAR-specific transforms"""
        if dataset == 'cifar10':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
        else:  # cifar100
            mean = [0.5071, 0.4867, 0.4408]
            std = [0.2675, 0.2565, 0.2761]
        
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        
        return transform
    
    def setup_data_loaders(self, batch_size=128, val_split=0.1):
        """Setup CIFAR data loaders"""
        print(f"Setting up {self.dataset_name.upper()} data loaders...")
        
        # Training data with validation split
        train_transform = self.get_cifar_transforms(train=True, dataset=self.dataset_name)
        test_transform = self.get_cifar_transforms(train=False, dataset=self.dataset_name)
        
        if self.dataset_name == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, transform=train_transform, download=True
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, transform=test_transform, download=True
            )
        else:  # cifar100
            train_dataset = torchvision.datasets.CIFAR100(
                root=self.data_dir, train=True, transform=train_transform, download=True
            )
            test_dataset = torchvision.datasets.CIFAR100(
                root=self.data_dir, train=False, transform=test_transform, download=True
            )
        
        # Split training data into train/val
        train_size = int((1 - val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Train samples: {len(train_subset)}")
        print(f"Validation samples: {len(val_subset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        
        self.val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        
        # Setup OOD loaders
        self.setup_ood_loaders(batch_size)
    
    def setup_ood_loaders(self, batch_size=128):
        """Setup OOD dataset loaders"""
        print("Setting up OOD dataset loaders...")
        
        test_transform = self.get_cifar_transforms(train=False, dataset=self.dataset_name)
        
        # CIFAR-100 as OOD for CIFAR-10 and vice versa
        if self.dataset_name == 'cifar10':
            cifar100_dataset = torchvision.datasets.CIFAR100(
                root=self.data_dir, train=False, transform=test_transform, download=True
            )
            self.ood_loaders['cifar100'] = DataLoader(
                cifar100_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )
        else:
            cifar10_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, transform=test_transform, download=True
            )
            self.ood_loaders['cifar10'] = DataLoader(
                cifar10_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )
        
        # SVHN
        try:
            svhn_dataset = torchvision.datasets.SVHN(
                root=self.data_dir, split='test', transform=test_transform, download=True
            )
            self.ood_loaders['svhn'] = DataLoader(
                svhn_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )
        except Exception as e:
            print(f"Failed to load SVHN: {e}")
        
        # Textures (DTD)
        try:
            # Resize DTD images to CIFAR size
            dtd_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
            ])
            dtd_dataset = torchvision.datasets.DTD(
                root=self.data_dir, split='test', transform=dtd_transform, download=True
            )
            self.ood_loaders['textures'] = DataLoader(
                dtd_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )
        except Exception as e:
            print(f"Failed to load DTD: {e}")
        
        print(f"Available OOD datasets: {list(self.ood_loaders.keys())}")
    
    def setup_model(self, pretrained=False):
        """Setup ResNet model"""
        print(f"Setting up {self.model_name} model...")
        
        self.model = TimmFeatureExtractor(
            model_name=self.model_name,
            num_classes=self.num_classes,
            pretrained=pretrained,
            use_checkpoint=False  # Usually not needed for smaller models
        )
        
        self.model = self.model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model: {self.model_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Classes: {self.num_classes}")
    
    def train_model(self, 
                   num_epochs=100,
                   lr=0.1,
                   use_hierarchical_dro=True,
                   save_intermediate=True):
        """Train the model with optional hierarchical DRO"""
        print(f"\nStarting training for {num_epochs} epochs...")
        
        if use_hierarchical_dro:
            # Initialize hierarchical DRO system
            self.hierarchical_system = HierarchicalDROWithMultiScoring(
                model=self.model,
                num_classes=self.num_classes,
                device=self.device,
                pixel_radius=0.03,
                feature_radius=0.1,
                pixel_weight=0.3,
                feature_weight=0.5,
                cross_level_weight=0.2
            )
            
            # Train with hierarchical DRO
            self.hierarchical_system.train_hierarchical_dro(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                num_epochs=num_epochs,
                lr=lr,
                config={
                    'mixed_precision': True,
                    'gradient_accumulation_steps': 1,
                    'grad_checkpoint': False,
                    'empty_cache_freq': 10
                },
                checkpoint_dir=str(self.checkpoint_dir)
            )
        else:
            # Standard training
            self._train_standard(num_epochs, lr, save_intermediate)
        
        # Save final model
        self.save_model('final_model')
        print("Training completed!")
    
    def _train_standard(self, num_epochs, lr, save_intermediate):
        """Standard training without hierarchical DRO"""
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=lr, 
            momentum=0.9, 
            weight_decay=5e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in trange(num_epochs, desc="Training"):
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, targets) in enumerate(tqdm(self.train_loader, leave=False)):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                logits, _ = self.model(data)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            scheduler.step()
            
            # Validation
            val_loss, val_acc = self._validate()
            
            # Record training history
            epoch_stats = {
                'epoch': epoch,
                'train_loss': epoch_loss / len(self.train_loader),
                'train_acc': 100. * correct / total,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': scheduler.get_last_lr()[0]
            }
            self.training_history.append(epoch_stats)
            
            # Save intermediate results
            if save_intermediate and (epoch + 1) % 10 == 0:
                self.save_intermediate_results(epoch)
            
            print(f"Epoch {epoch+1}: Train Acc: {100.*correct/total:.2f}%, Val Acc: {val_acc:.2f}%")
    
    def _validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                logits, _ = self.model(data)
                loss = criterion(logits, targets)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return val_loss / len(self.val_loader), 100. * correct / total
    
    def evaluate_ood_detection(self, methods=None, save_scores=True):
        """Comprehensive OOD detection evaluation"""
        if methods is None:
            methods = ['energy', 'mahalanobis', 'msp', 'odin', 'ensemble']
        
        print(f"\nEvaluating OOD detection with methods: {methods}")
        
        # Setup multi-scoring if using hierarchical DRO
        if self.hierarchical_system is not None:
            self.hierarchical_system.setup_multi_scoring(self.test_loader)
            multi_scorer = self.hierarchical_system.multi_scorer
        else:
            # Create standalone multi-scorer
            multi_scorer = MultiScoreOODDetector(
                self.model, 
                num_classes=self.num_classes, 
                device=self.device
            )
            multi_scorer.fit_feature_statistics(self.test_loader)
        
        # Get ID test data
        id_data = self._get_dataset_samples(self.test_loader, max_samples=2000)
        
        all_results = []
        
        for ood_name, ood_loader in self.ood_loaders.items():
            print(f"Evaluating against {ood_name}...")
            
            # Get OOD data
            ood_data = self._get_dataset_samples(ood_loader, max_samples=2000)
            
            for method in tqdm(methods, desc=f'Methods for {ood_name}'):
                try:
                    # Compute scores
                    if method == 'ensemble':
                        id_scores, _, _ = multi_scorer.compute_ensemble_score(id_data)
                        ood_scores, _, _ = multi_scorer.compute_ensemble_score(ood_data)
                    else:
                        score_fn = getattr(multi_scorer, f'compute_{method}_score')
                        id_scores = score_fn(id_data)
                        ood_scores = score_fn(ood_data)
                    
                    # Compute metrics
                    metrics = self._compute_ood_metrics(id_scores, ood_scores)
                    
                    # Save individual scores if requested
                    if save_scores:
                        self._save_scores(id_scores, ood_scores, method, ood_name)
                    
                    # Store results
                    result = {
                        'id_dataset': self.dataset_name,
                        'ood_dataset': ood_name,
                        'method': method,
                        'model': self.model_name,
                        **metrics,
                        'timestamp': datetime.now().isoformat()
                    }
                    all_results.append(result)
                    
                    print(f"  {method}: AUROC={metrics['auroc']:.4f}, FPR@95={metrics['fpr95']:.4f}")
                    
                except Exception as e:
                    print(f"  Failed {method}: {e}")
                    continue
        
        # Save results
        self.evaluation_results = all_results
        self._save_evaluation_results()
        
        return all_results
    
    def _get_dataset_samples(self, loader, max_samples=2000):
        """Get limited samples from dataset"""
        data_list = []
        samples_collected = 0
        
        for data, _ in loader:
            data_list.append(data)
            samples_collected += data.size(0)
            if samples_collected >= max_samples:
                break
        
        return torch.cat(data_list)[:max_samples].to(self.device)
    
    def _compute_ood_metrics(self, id_scores, ood_scores):
        """Compute OOD detection metrics"""
        id_scores_np = id_scores.cpu().numpy()
        ood_scores_np = ood_scores.cpu().numpy()
        
        # Labels: 0 for ID, 1 for OOD
        y_true = np.concatenate([np.zeros_like(id_scores_np), np.ones_like(ood_scores_np)])
        y_score = np.concatenate([id_scores_np, ood_scores_np])
        
        # AUROC
        auroc = roc_auc_score(y_true, y_score)
        
        # FPR@95% TPR
        fpr, tpr, _ = roc_curve(y_true, y_score)
        idx_95 = np.argmin(np.abs(tpr - 0.95))
        fpr95 = fpr[idx_95]
        
        # Detection error (minimum over threshold)
        detection_error = 0.5 * (1 - tpr + fpr).min()
        
        # Average precision
        from sklearn.metrics import average_precision_score
        aupr = average_precision_score(y_true, y_score)
        
        return {
            'auroc': float(auroc),
            'fpr95': float(fpr95),
            'detection_error': float(detection_error),
            'aupr': float(aupr),
            'id_mean': float(id_scores_np.mean()),
            'id_std': float(id_scores_np.std()),
            'ood_mean': float(ood_scores_np.mean()),
            'ood_std': float(ood_scores_np.std()),
            'separation': float(ood_scores_np.mean() - id_scores_np.mean())
        }
    
    def _save_scores(self, id_scores, ood_scores, method, ood_name):
        """Save individual scores for analysis"""
        scores_data = {
            'id_scores': id_scores.cpu().numpy(),
            'ood_scores': ood_scores.cpu().numpy(),
            'method': method,
            'ood_dataset': ood_name,
            'model': self.model_name,
            'id_dataset': self.dataset_name
        }
        
        filename = f'scores_{self.dataset_name}_{self.model_name}_{method}_{ood_name}.pkl'
        filepath = self.scores_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(scores_data, f)
    
    def save_intermediate_results(self, epoch):
        """Save intermediate training results"""
        intermediate_data = {
            'epoch': epoch,
            'training_history': self.training_history,
            'model_state': self.model.state_dict(),
            'dataset': self.dataset_name,
            'model_name': self.model_name
        }
        
        filename = f'intermediate_{self.dataset_name}_{self.model_name}_epoch_{epoch}.pkl'
        filepath = self.results_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(intermediate_data, f)
    
    def _save_evaluation_results(self):
        """Save evaluation results in multiple formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as DataFrame
        df = pd.DataFrame(self.evaluation_results)
        
        # CSV
        csv_path = self.results_dir / f'ood_results_{self.dataset_name}_{self.model_name}_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        
        # Excel with summary
        excel_path = self.results_dir / f'ood_results_{self.dataset_name}_{self.model_name}_{timestamp}.xlsx'
        with pd.ExcelWriter(excel_path) as writer:
            df.to_excel(writer, sheet_name='detailed_results', index=False)
            
            # Summary by method
            summary = df.groupby('method')[['auroc', 'fpr95', 'detection_error', 'aupr']].agg(['mean', 'std']).round(4)
            summary.to_excel(writer, sheet_name='method_summary')
            
            # Summary by OOD dataset
            ood_summary = df.groupby('ood_dataset')[['auroc', 'fpr95', 'detection_error', 'aupr']].agg(['mean', 'std']).round(4)
            ood_summary.to_excel(writer, sheet_name='ood_summary')
        
        # JSON
        json_path = self.results_dir / f'ood_results_{self.dataset_name}_{self.model_name}_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=4, default=str)
        
        print(f"Results saved to {self.results_dir}")
    
    def save_model(self, name='model'):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'dataset': self.dataset_name,
            'num_classes': self.num_classes,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.hierarchical_system is not None:
            checkpoint['hierarchical_system_state'] = {
                'pixel_weight': self.hierarchical_system.pixel_weight.item(),
                'feature_weight': self.hierarchical_system.feature_weight.item(),
                'cross_level_weight': self.hierarchical_system.cross_level_weight.item(),
            }
        
        filepath = self.checkpoint_dir / f'{name}_{self.dataset_name}_{self.model_name}.pt'
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        print(f"Model loaded from {filepath}")
    
    def create_visualizations(self):
        """Create visualization plots"""
        if not self.evaluation_results:
            print("No evaluation results to visualize")
            return
        
        # Create plots directory
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        df = pd.DataFrame(self.evaluation_results)
        
        # AUROC comparison
        plt.figure(figsize=(12, 8))
        auroc_pivot = df.pivot(index='ood_dataset', columns='method', values='auroc')
        sns.heatmap(auroc_pivot, annot=True, cmap='viridis', fmt='.3f', cbar_kws={'label': 'AUROC'})
        plt.title(f'AUROC Scores - {self.dataset_name.upper()} with {self.model_name}')
        plt.tight_layout()
        plt.savefig(plots_dir / f'auroc_heatmap_{self.dataset_name}_{self.model_name}.png', dpi=300)
        plt.close()
        
        # FPR@95 comparison
        plt.figure(figsize=(12, 8))
        fpr_pivot = df.pivot(index='ood_dataset', columns='method', values='fpr95')
        sns.heatmap(fpr_pivot, annot=True, cmap='viridis_r', fmt='.3f', cbar_kws={'label': 'FPR@95'})
        plt.title(f'FPR@95 Scores - {self.dataset_name.upper()} with {self.model_name}')
        plt.tight_layout()
        plt.savefig(plots_dir / f'fpr95_heatmap_{self.dataset_name}_{self.model_name}.png', dpi=300)
        plt.close()
        
        # Bar plot comparison
        plt.figure(figsize=(15, 10))
        
        # AUROC subplot
        plt.subplot(2, 2, 1)
        method_auroc = df.groupby('method')['auroc'].mean().sort_values(ascending=False)
        method_auroc.plot(kind='bar', color='skyblue')
        plt.title('Average AUROC by Method')
        plt.ylabel('AUROC')
        plt.xticks(rotation=45)
        
        # FPR@95 subplot
        plt.subplot(2, 2, 2)
        method_fpr = df.groupby('method')['fpr95'].mean().sort_values()
        method_fpr.plot(kind='bar', color='lightcoral')
        plt.title('Average FPR@95 by Method')
        plt.ylabel('FPR@95')
        plt.xticks(rotation=45)
        
        # AUPR subplot
        plt.subplot(2, 2, 3)
        method_aupr = df.groupby('method')['aupr'].mean().sort_values(ascending=False)
        method_aupr.plot(kind='bar', color='lightgreen')
        plt.title('Average AUPR by Method')
        plt.ylabel('AUPR')
        plt.xticks(rotation=45)
        
        # Detection Error subplot
        plt.subplot(2, 2, 4)
        method_de = df.groupby('method')['detection_error'].mean().sort_values()
        method_de.plot(kind='bar', color='gold')
        plt.title('Average Detection Error by Method')
        plt.ylabel('Detection Error')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'method_comparison_{self.dataset_name}_{self.model_name}.png', dpi=300)
        plt.close()
        
        # Training history plot if available
        if self.training_history:
            plt.figure(figsize=(15, 5))
            
            history_df = pd.DataFrame(self.training_history)
            
            plt.subplot(1, 3, 1)
            plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
            plt.plot(history_df['epoch'], history_df['val_loss'], label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            
            plt.subplot(1, 3, 2)
            plt.plot(history_df['epoch'], history_df['train_acc'], label='Train Acc')
            plt.plot(history_df['epoch'], history_df['val_acc'], label='Val Acc')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.title('Training and Validation Accuracy')
            
            plt.subplot(1, 3, 3)
            plt.plot(history_df['epoch'], history_df['lr'])
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'training_history_{self.dataset_name}_{self.model_name}.png', dpi=300)
            plt.close()
        
        print(f"Visualizations saved to {plots_dir}")
    
    def run_full_experiment(self, 
                           num_epochs=100,
                           lr=0.1,
                           batch_size=128,
                           use_hierarchical_dro=True,
                           pretrained=False,
                           methods=None):
        """Run complete experiment pipeline"""
        print(f"Starting full experiment: {self.dataset_name} with {self.model_name}")
        print(f"Hierarchical DRO: {use_hierarchical_dro}")
        print(f"Epochs: {num_epochs}, LR: {lr}, Batch size: {batch_size}")
        
        # Setup
        self.setup_data_loaders(batch_size=batch_size)
        self.setup_model(pretrained=pretrained)
        
        # Training
        self.train_model(
            num_epochs=num_epochs,
            lr=lr,
            use_hierarchical_dro=use_hierarchical_dro,
            save_intermediate=True
        )
        
        # Evaluation
        self.evaluate_ood_detection(methods=methods, save_scores=True)
        
        # Visualizations
        self.create_visualizations()
        
        # Summary
        self.print_summary()
        
        print(f"Experiment completed! Results saved to {self.output_dir}")
    
    def print_summary(self):
        """Print experiment summary"""
        if not self.evaluation_results:
            print("No results to summarize")
            return
        
        df = pd.DataFrame(self.evaluation_results)
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT SUMMARY: {self.dataset_name.upper()} with {self.model_name}")
        print(f"{'='*60}")
        
        # Overall statistics
        print(f"Total OOD datasets tested: {df['ood_dataset'].nunique()}")
        print(f"Methods evaluated: {list(df['method'].unique())}")
        
        # Best performing method
        avg_auroc = df.groupby('method')['auroc'].mean()
        best_method = avg_auroc.idxmax()
        print(f"Best method (AUROC): {best_method} ({avg_auroc[best_method]:.4f})")
        
        # Method comparison table
        print("\nMethod Performance Summary:")
        summary = df.groupby('method')[['auroc', 'fpr95', 'aupr', 'detection_error']].agg(['mean', 'std'])
        print(summary.round(4))
        
        # OOD dataset difficulty ranking
        print("\nOOD Dataset Difficulty (by average AUROC):")
        ood_difficulty = df.groupby('ood_dataset')['auroc'].mean().sort_values(ascending=False)
        for ood_name, auroc in ood_difficulty.items():
            print(f"  {ood_name}: {auroc:.4f}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CIFAR ResNet Experiments')
    
    # Dataset and model
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10',
                       help='Dataset to use for training')
    parser.add_argument('--model', default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                       help='ResNet model to use')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--no-hierarchical-dro', action='store_true',
                       help='Disable hierarchical DRO (use standard training)')
    
    # Evaluation
    parser.add_argument('--methods', nargs='+', 
                       choices=['energy', 'mahalanobis', 'msp', 'odin', 'ensemble'],
                       default=['energy', 'mahalanobis', 'msp', 'odin', 'ensemble'],
                       help='OOD detection methods to evaluate')
    
    # Directories
    parser.add_argument('--data-dir', default='./data',
                       help='Directory containing datasets')
    parser.add_argument('--output-dir', default='./cifar_results',
                       help='Directory for saving results')
    
    # Device
    parser.add_argument('--device', default='cuda',
                       help='Device to use for training')
    
    # Modes
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation (requires pre-trained model)')
    parser.add_argument('--model-path', default=None,
                       help='Path to pre-trained model for evaluation')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Create experiment
    experiment = CIFARResNetExperiment(
        dataset_name=args.dataset,
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    if args.eval_only:
        if args.model_path is None:
            raise ValueError("Must provide --model-path for evaluation-only mode")
        
        # Setup and load model
        experiment.setup_data_loaders(batch_size=args.batch_size)
        experiment.setup_model(pretrained=args.pretrained)
        experiment.load_model(args.model_path)
        
        # Run evaluation
        experiment.evaluate_ood_detection(methods=args.methods, save_scores=True)
        experiment.create_visualizations()
        experiment.print_summary()
    else:
        # Run full experiment
        experiment.run_full_experiment(
            num_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            use_hierarchical_dro=not args.no_hierarchical_dro,
            pretrained=args.pretrained,
            methods=args.methods
        )


if __name__ == "__main__":
    main() 