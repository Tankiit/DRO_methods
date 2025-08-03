#!/usr/bin/env python3
"""
Evaluation-only script for pre-trained models.
This skips training and only runs evaluation, which is much faster.
"""

import argparse
import os
import torch
from pathlib import Path

# Import from the main training script
from feature_train import (
    get_model, MultiDatasetEvaluator, setup_logging_and_directories
)

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate pre-trained model without training")
    p.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    p.add_argument('--backbone', default='vit_base_patch16_224', 
                  choices=['resnet18', 'resnet50', 'vit_base_patch16_224', 'vit_large_patch16_224', 
                          'dino_vits16', 'dino_vitb16'])
    p.add_argument('--ood', default='cifar100,svhn,textures',
                  help='Comma-separated list of OOD datasets')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--data-dir', type=str, default='/home/tanmoy/research/data', 
                  help='Path to data directory')
    p.add_argument('--output-dir', type=str, default='eval_results',
                  help='Directory for saving evaluation results')
    p.add_argument('--use-cache', action='store_true', 
                  help='Use cached feature statistics if available')
    return p.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model from checkpoint: {args.checkpoint}")
    
    # Load model
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Initialize model architecture
    model_args = argparse.Namespace(
        backbone=args.backbone,
        pretrained=False,  # We're loading weights from checkpoint
        grad_checkpoint=False,  # Not needed for evaluation
        dataset='imagenet'
    )
    model = get_model(model_args)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model state dict from checkpoint")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model directly from checkpoint")
    
    model = model.to('cuda', non_blocking=True)
    model.eval()
    
    print("\nStarting evaluation-only mode...")
    evaluator = MultiDatasetEvaluator(
        model=model,
        base_data_dir=args.data_dir,
        device='cuda',
        batch_size=args.batch
    )
    
    # Try to load cached feature statistics if requested
    if args.use_cache:
        cache_path = os.path.join(args.output_dir, f'feature_stats_cache_{args.backbone}.pt')
        cache_loaded = evaluator.load_feature_stats_cache(cache_path)
        if cache_loaded:
            print("Using cached feature statistics!")
    
    # Parse datasets
    ood_datasets = args.ood.split(',') if args.ood else ['cifar100', 'svhn', 'textures']
    id_datasets = ['imagenet']
    
    print(f"\nEvaluating on ID datasets: {id_datasets}")
    print(f"Testing against OOD datasets: {ood_datasets}")
    
    # Run evaluation
    results_df, summaries = evaluator.evaluate_all_combinations(
        id_datasets=id_datasets,
        ood_datasets=ood_datasets,
        methods=['energy', 'mahalanobis', 'msp', 'odin', 'ensemble']
    )
    
    # Save feature statistics cache for future runs
    if args.use_cache:
        cache_path = os.path.join(args.output_dir, f'feature_stats_cache_{args.backbone}.pt')
        evaluator.save_feature_stats_cache(cache_path)
        print("Feature statistics cached for future runs!")
    
    # Save results
    save_dir = evaluator.save_results(
        results_df=results_df,
        summaries=summaries,
        output_dir=args.output_dir,
        model_name=f"{args.backbone}_eval_only"
    )
    
    print(f"\nEvaluation completed! Results saved to: {save_dir}")
    
    # Print quick summary
    print("\nQuick Summary:")
    for method in ['energy', 'mahalanobis', 'ensemble']:
        method_results = results_df[results_df['method'] == method]
        if not method_results.empty:
            avg_auroc = method_results['auroc'].mean()
            avg_fpr95 = method_results['fpr95'].mean()
            print(f"{method.upper()}: AUROC={avg_auroc:.4f}, FPR95={avg_fpr95:.4f}")

if __name__ == "__main__":
    main() 