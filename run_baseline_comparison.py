#!/usr/bin/env python3
"""
Baseline Comparison Runner for CIFAR OOD Detection
This script runs comprehensive baseline comparison using trained H-DRO models.
"""

import argparse
import torch
import os
import sys
from pathlib import Path
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ablation_analysis import run_baseline_comparison
from feature_train import HierarchicalDROWithMultiScoring, TimmFeatureExtractor

def parse_args():
    parser = argparse.ArgumentParser(description='Run comprehensive baseline comparison for CIFAR OOD detection')
    
    # Model and checkpoint configuration
    parser.add_argument('--checkpoint', type=str, help='Path to specific checkpoint file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', 
                       help='Directory containing checkpoints')
    parser.add_argument('--model-name', type=str, default='best_model', 
                       help='Model name pattern to search for')
    
    # Dataset and model configuration
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--backbone', default='resnet18', 
                       choices=['resnet18', 'resnet50', 'vit_base_patch16_224', 'efficientnet_b0'])
    parser.add_argument('--data-dir', default='/home/tanmoy/research/data', help='Data directory')
    
    # Output configuration
    parser.add_argument('--output-dir', default='baseline_comparison_results', 
                       help='Output directory for results')
    parser.add_argument('--run-name', default='', help='Custom run name for output directory')
    
    # Analysis configuration
    parser.add_argument('--quick-mode', action='store_true', 
                       help='Run in quick mode with fewer methods')
    parser.add_argument('--methods', nargs='+', 
                       help='Specific methods to test (default: all)')
    
    return parser.parse_args()

def find_latest_checkpoint(checkpoint_dir, model_name="best_model"):
    """Find the latest checkpoint file."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"Checkpoint directory {checkpoint_dir} does not exist!")
        return None
    
    # Look for checkpoint files
    patterns = [
        f"{model_name}*.pt",
        f"{model_name}*.pth", 
        "*.pt",
        "*.pth"
    ]
    
    checkpoint_files = []
    for pattern in patterns:
        checkpoint_files.extend(list(checkpoint_path.glob(pattern)))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return None
    
    # Sort by modification time and get the latest
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    print(f"Found latest checkpoint: {latest_checkpoint}")
    
    return latest_checkpoint

def detect_architecture_from_checkpoint(checkpoint):
    """Detect model architecture from checkpoint keys."""
    if 'model_state_dict' in checkpoint:
        keys = list(checkpoint['model_state_dict'].keys())
    elif 'state_dict' in checkpoint:
        keys = list(checkpoint['state_dict'].keys())
    else:
        keys = list(checkpoint.keys())
    
    # Check for ViT architecture
    if any('cls_token' in key or 'pos_embed' in key or 'patch_embed' in key for key in keys):
        return 'vit_base_patch16_224'
    
    # Check for ResNet architecture
    elif any('conv1.weight' in key or 'layer1' in key for key in keys):
        if any('layer4' in key for key in keys):
            # Check if it's ResNet50 (has bottleneck structure)
            if any('conv3.weight' in key for key in keys):
                return 'resnet50'
            else:
                return 'resnet18'
        else:
            return 'resnet18'
    
    # Check for EfficientNet
    elif any('stem' in key or 'blocks' in key for key in keys):
        return 'efficientnet_b0'
    
    # Default fallback
    return 'resnet18'

def load_trained_model(checkpoint_path, config, device='cuda'):
    """Load a trained model from checkpoint with automatic architecture detection."""
    try:
        print(f"Loading model from {checkpoint_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Detect architecture from checkpoint if not specified correctly
        detected_arch = detect_architecture_from_checkpoint(checkpoint)
        if detected_arch != config['backbone']:
            print(f"⚠️ Architecture mismatch detected!")
            print(f"   Config specifies: {config['backbone']}")
            print(f"   Checkpoint contains: {detected_arch}")
            print(f"   Using detected architecture: {detected_arch}")
            config['backbone'] = detected_arch
        
        # Get the state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Check if this is a wrapped HierarchicalDRO model or a raw model
        is_hierarchical_dro = any('model.' in key or 'fast_virtual_generator' in key for key in state_dict.keys())
        
        if is_hierarchical_dro:
            print("   Detected: HierarchicalDRO wrapped model")
            # Create base model first
            base_model = TimmFeatureExtractor(
                model_name=config['backbone'],
                num_classes=config['num_classes'],
                pretrained=False
            )
            
            # Create HierarchicalDRO wrapper
            model = HierarchicalDROWithMultiScoring(
                model=base_model,
                num_classes=config['num_classes'],
                device=device
            )
            
            # Load the full state dict
            model.load_state_dict(state_dict, strict=False)
            
        else:
            print("   Detected: Raw model (creating simple wrapper)")
            # This is a raw model, create a simple wrapper for evaluation
            model = TimmFeatureExtractor(
                model_name=config['backbone'],
                num_classes=config['num_classes'],
                pretrained=False
            )
            
            # Load the raw model state dict
            model.load_state_dict(state_dict, strict=False)
        
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Architecture: {config['backbone']}")
        print(f"   Classes: {config['num_classes']}")
        print(f"   Device: {device}")
        
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model from {checkpoint_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_config(args):
    """Create configuration dictionary from arguments."""
    num_classes = 100 if args.dataset == 'cifar100' else 10
    
    config = {
        'dataset': args.dataset,
        'backbone': args.backbone,
        'data_dir': args.data_dir,
        'num_classes': num_classes,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Add quick mode settings
    if args.quick_mode:
        config['quick_mode'] = True
    
    # Add specific methods if provided
    if args.methods:
        config['methods'] = args.methods
    
    return config

def main():
    args = parse_args()
    
    print("🔬 CIFAR OOD Detection - Baseline Comparison")
    print("=" * 60)
    
    # Create configuration
    config = create_config(args)
    device = config['device']
    
    print(f"📋 CONFIGURATION:")
    print(f"  Dataset: {config['dataset'].upper()}")
    print(f"  Backbone: {config['backbone']}")
    print(f"  Device: {device}")
    print(f"  Quick Mode: {args.quick_mode}")
    print()
    
    # Find and load checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"❌ Specified checkpoint {args.checkpoint} does not exist!")
            return
    else:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir, args.model_name)
        if checkpoint_path is None:
            print("❌ No checkpoint found! Please train a model first or specify a checkpoint path.")
            return
    
    # Load the trained model
    trained_model = load_trained_model(checkpoint_path, config, device)
    if trained_model is None:
        print("❌ Failed to load trained model!")
        return
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name if args.run_name else f"{config['dataset']}_{config['backbone']}"
    output_dir = f"{args.output_dir}/{run_name}_{timestamp}"
    
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Run baseline comparison
    try:
        print("🚀 Starting Baseline Comparison...")
        print("-" * 60)
        
        results = run_baseline_comparison(
            trained_model=trained_model,
            config=config,
            output_dir=output_dir,
            device=device
        )
        
        print("\n✅ Baseline comparison completed successfully!")
        print(f"📊 Results saved to: {Path(output_dir).absolute()}")
        
        # Quick summary
        if 'analysis' in results and 'performance_ranking' in results['analysis']:
            print("\n🏆 TOP 3 METHODS:")
            for i, method in enumerate(results['analysis']['performance_ranking'][:3]):
                print(f"  {i+1}. {method['method']} - AUROC: {method['auroc']:.4f}")
        
    except Exception as e:
        print(f"❌ Baseline comparison failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 