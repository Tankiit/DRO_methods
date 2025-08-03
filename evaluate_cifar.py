#!/usr/bin/env python3
"""
CIFAR-10/100 OOD Detection Evaluation Script
This script focuses on evaluating pre-trained models on CIFAR datasets for OOD detection.
"""

import torch
import torch.nn as nn
import argparse
import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Import the main classes from feature_train.py
from feature_train import (
    get_model, get_dataset, MultiDatasetEvaluator, 
    setup_logging_and_directories, get_dataset_config
)

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate CIFAR models for OOD detection')
    p.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    p.add_argument('--ood', default='cifar100,svhn,textures', 
                   help='Comma-separated list of OOD datasets')
    p.add_argument('--backbone', '-b', default='resnet18', 
                   choices=['resnet18', 'resnet50', 'vit_base_patch16_224', 'vit_large_patch16_224'])
    p.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--data-dir', type=str, default='/home/tanmoy/research/data')
    p.add_argument('--output-dir', type=str, default='cifar_results')
    p.add_argument('--checkpoint', type=str, help='Path to model checkpoint (optional)')
    p.add_argument('--methods', default='energy,mahalanobis,msp,odin,ensemble',
                   help='Comma-separated list of OOD detection methods')
    # Add dummy arguments to make functions work
    p.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    p.add_argument('--runs-dir', type=str, default='runs')
    p.add_argument('--grad-checkpoint', action='store_true', help='Use gradient checkpointing')
    return p.parse_args()

def load_checkpoint(model, checkpoint_path):
    """Load model weights from checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found")
        return False
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

def main():
    args = parse_args()
    
    print("="*60)
    print(f"CIFAR OOD Detection Evaluation")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Backbone: {args.backbone}")
    print(f"OOD Datasets: {args.ood}")
    print("="*60)
    
    # Set up directories
    setup_logging_and_directories(args)
    
    # Get model
    print("\nInitializing model...")
    model = get_model(args)
    model = model.to('cuda', non_blocking=True)
    
    # Load checkpoint if provided
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    elif args.pretrained:
        print("Using pretrained weights from timm/torchvision")
    else:
        print("Warning: Using randomly initialized weights!")
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = MultiDatasetEvaluator(
        model=model,
        base_data_dir=args.data_dir,
        device='cuda',
        batch_size=args.batch
    )
    
    # Try to load cached feature statistics
    cache_path = os.path.join(args.output_dir, f'feature_stats_cache_{args.dataset}_{args.backbone}.pt')
    cache_loaded = evaluator.load_feature_stats_cache(cache_path)
    if cache_loaded:
        print("✓ Using cached feature statistics")
    else:
        print("ℹ Computing feature statistics from scratch...")
    
    # Parse datasets
    ood_datasets = [d.strip() for d in args.ood.split(',')]
    id_datasets = [args.dataset]
    methods = [m.strip() for m in args.methods.split(',')]
    
    print(f"\nID Dataset: {id_datasets[0]}")
    print(f"OOD Datasets: {ood_datasets}")
    print(f"Detection Methods: {methods}")
    
    # Run evaluation
    print("\nStarting evaluation...")
    results_df, summaries = evaluator.evaluate_all_combinations(
        id_datasets=id_datasets,
        ood_datasets=ood_datasets,
        methods=methods
    )
    
    # Save results
    save_dir = evaluator.save_results(
        results_df=results_df,
        summaries=summaries,
        output_dir=args.output_dir,
        model_name=f"{args.dataset}_{args.backbone}_eval"
    )
    
    # Cache feature statistics for future runs
    if not cache_loaded:
        evaluator.save_feature_stats_cache(cache_path)
        print("✓ Feature statistics cached for future runs")
    
    # Print summary results
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    # Create a pivot table for easy reading
    if not results_df.empty:
        print("\nAUROC Scores:")
        auroc_pivot = results_df.pivot_table(
            values='auroc', 
            index='ood_dataset', 
            columns='method', 
            aggfunc='mean'
        )
        print(auroc_pivot.round(4))
        
        print("\nFPR@95% Scores:")
        fpr_pivot = results_df.pivot_table(
            values='fpr95', 
            index='ood_dataset', 
            columns='method', 
            aggfunc='mean'
        )
        print(fpr_pivot.round(4))
        
        # Best performing method per OOD dataset
        print("\nBest Method per OOD Dataset (by AUROC):")
        best_methods = results_df.loc[results_df.groupby('ood_dataset')['auroc'].idxmax()]
        for _, row in best_methods.iterrows():
            print(f"  {row['ood_dataset']}: {row['method']} (AUROC: {row['auroc']:.4f})")
        
        # Overall best method
        avg_performance = results_df.groupby('method')[['auroc', 'fpr95']].mean()
        best_overall = avg_performance['auroc'].idxmax()
        print(f"\nBest Overall Method: {best_overall}")
        print(f"  Average AUROC: {avg_performance.loc[best_overall, 'auroc']:.4f}")
        print(f"  Average FPR@95: {avg_performance.loc[best_overall, 'fpr95']:.4f}")
    
    print(f"\nDetailed results saved to: {save_dir}")
    print("="*60)

if __name__ == "__main__":
    main() 