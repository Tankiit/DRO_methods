#!/usr/bin/env python3
"""
Comprehensive Analysis Runner for CIFAR OOD Detection
This script runs both ablation studies and efficiency analysis for the hierarchical DRO system.
"""

import argparse
import torch
from pathlib import Path
from datetime import datetime

from ablation_analysis import AblationStudyRunner, EfficiencyAnalyzer, run_ablation_study, run_efficiency_analysis

def parse_args():
    parser = argparse.ArgumentParser(description='Run comprehensive analysis for CIFAR OOD detection')
    
    # Dataset and model configuration
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--backbone', default='resnet18', 
                       choices=['resnet18', 'resnet50', 'vit_base_patch16_224', 'efficientnet_b0'])
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--data-dir', default='/home/tanmoy/research/data', help='Data directory')
    
    # Analysis configuration
    parser.add_argument('--analysis-type', default='both', choices=['ablation', 'efficiency', 'both'],
                       help='Type of analysis to run')
    parser.add_argument('--output-dir', default='comprehensive_analysis_results', 
                       help='Output directory for results')
    parser.add_argument('--quick-mode', action='store_true', 
                       help='Run in quick mode with reduced parameters')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for ablation studies')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    return parser.parse_args()

def create_config(args):
    """Create configuration dictionary from arguments."""
    num_classes = 100 if args.dataset == 'cifar100' else 10
    
    config = {
        'dataset': args.dataset,
        'backbone': args.backbone,
        'pretrained': args.pretrained,
        'data_dir': args.data_dir,
        'num_classes': num_classes,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_epochs': args.epochs,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Adjust parameters for quick mode
    if args.quick_mode:
        config.update({
            'num_epochs': 3,
            'batch_size': 16,
            'quick_mode': True
        })
    
    return config

def print_analysis_summary():
    """Print what analyses will be performed."""
    print("🔬 COMPREHENSIVE CIFAR OOD DETECTION ANALYSIS")
    print("=" * 60)
    print("\n📊 ABLATION STUDY COMPONENTS:")
    print("  1. Virtual Sample Strategy Analysis")
    print("     • No Virtual Samples (ERM baseline)")
    print("     • Pixel-level Virtual Samples Only")
    print("     • Feature-level Virtual Samples Only")
    print("     • Combined Basic (simple combination)")
    print("     • Combined Full (hierarchical approach)")
    print()
    print("  2. Progressive Training Analysis")
    print("     • Standard ERM training")
    print("     • Pixel-only progressive training")
    print("     • Full progressive training")
    print("     • Simultaneous multi-level training")
    print()
    print("  3. Hyperparameter Sensitivity Analysis")
    print("     • Pixel perturbation radius")
    print("     • Feature perturbation radius")
    print("     • Loss weighting parameters")
    print("     • Learning rate sensitivity")
    print()
    print("  4. Architecture Generalization")
    print("     • ResNet-18/50 (CNN architectures)")
    print("     • Vision Transformer (ViT)")
    print("     • EfficientNet (compound scaling)")
    print()
    print("⚡ EFFICIENCY ANALYSIS COMPONENTS:")
    print("  1. Memory Usage Analysis")
    print("     • Training memory vs batch size")
    print("     • Inference memory requirements")
    print("     • Memory efficiency metrics")
    print()
    print("  2. Training Time Breakdown")
    print("     • Component-wise timing analysis")
    print("     • Throughput measurements")
    print("     • Bottleneck identification")
    print()
    print("  3. Scalability Analysis")
    print("     • Training time vs dataset size")
    print("     • Linear/sublinear scaling analysis")
    print("     • Resource utilization trends")
    print()
    print("  4. Batch Size Optimization")
    print("     • Throughput vs batch size")
    print("     • Memory efficiency analysis")
    print("     • Optimal batch size identification")
    print()
    print("  5. Baseline Method Comparison")
    print("     • Hierarchical DRO vs ERM")
    print("     • Virtual sample methods")
    print("     • Energy-based approaches")
    print()

def main():
    args = parse_args()
    
    # Print analysis overview
    print_analysis_summary()
    
    # Create configuration
    config = create_config(args)
    
    print(f"📋 CONFIGURATION:")
    print(f"  Dataset: {config['dataset'].upper()}")
    print(f"  Backbone: {config['backbone']}")
    print(f"  Pretrained: {config['pretrained']}")
    print(f"  Quick Mode: {args.quick_mode}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Device: {config['device']}")
    print()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = Path(args.output_dir) / f"{config['dataset']}_{config['backbone']}_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Run ablation study
    if args.analysis_type in ['ablation', 'both']:
        print("🚀 STARTING ABLATION STUDY...")
        print("-" * 40)
        
        try:
            ablation_dir = output_base / "ablation_study"
            runner = AblationStudyRunner(
                base_config=config,
                output_dir=str(ablation_dir),
                device=config['device']
            )
            
            ablation_results = runner.run_ablation_study()
            results['ablation'] = ablation_results
            
            print("✅ Ablation study completed successfully!")
            print(f"   Results saved to: {ablation_dir}")
            
            # Print quick summary
            if 'virtual_strategies' in ablation_results:
                print("\n📈 VIRTUAL STRATEGY RESULTS PREVIEW:")
                for strategy, res in ablation_results['virtual_strategies'].items():
                    if 'error' not in res and 'evaluation_metrics' in res:
                        auroc = res['evaluation_metrics'].get('avg_auroc', 0)
                        print(f"   {strategy:15s}: AUROC = {auroc:.4f}")
            
        except Exception as e:
            print(f"❌ Ablation study failed: {str(e)}")
            results['ablation_error'] = str(e)
    
    # Run efficiency analysis
    if args.analysis_type in ['efficiency', 'both']:
        print("\n⚡ STARTING EFFICIENCY ANALYSIS...")
        print("-" * 40)
        
        try:
            efficiency_dir = output_base / "efficiency_analysis"
            analyzer = EfficiencyAnalyzer(
                base_config=config,
                output_dir=str(efficiency_dir),
                device=config['device']
            )
            
            efficiency_results = analyzer.run_efficiency_analysis()
            results['efficiency'] = efficiency_results
            
            print("✅ Efficiency analysis completed successfully!")
            print(f"   Results saved to: {efficiency_dir}")
            
            # Print quick summary
            if 'memory_analysis' in efficiency_results:
                print("\n💾 MEMORY USAGE PREVIEW:")
                for batch_size, mem_data in efficiency_results['memory_analysis'].items():
                    if 'error' not in mem_data:
                        train_mem = mem_data.get('training_memory_mb', 0)
                        print(f"   Batch {batch_size:3s}: {train_mem:6.1f} MB training memory")
            
        except Exception as e:
            print(f"❌ Efficiency analysis failed: {str(e)}")
            results['efficiency_error'] = str(e)
    
    # Save combined results
    combined_results_file = output_base / "combined_results_summary.json"
    with open(combined_results_file, 'w') as f:
        import json
        json.dump(results, f, indent=4, default=str)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
    print("=" * 60)
    print(f"📁 All results saved to: {output_base}")
    print()
    print("📊 Generated Files:")
    print("   • Detailed analysis results (JSON)")
    print("   • Performance visualization plots (PNG)")
    print("   • LaTeX tables for papers (TEX)")
    print("   • CSV data for further analysis")
    print("   • Comprehensive logs")
    print()
    print("🔍 Key Insights Available:")
    print("   • Best virtual sample strategy")
    print("   • Optimal hyperparameter ranges")
    print("   • Architecture performance trade-offs")
    print("   • Memory and timing bottlenecks")
    print("   • Scalability characteristics")
    print("   • Baseline method comparisons")
    print()
    print("Next Steps:")
    print("1. Review generated plots in plots/ directories")
    print("2. Use LaTeX tables in your paper from tables/ directories")
    print("3. Analyze detailed results in JSON files")
    print("4. Run focused experiments based on insights")

if __name__ == "__main__":
    main() 